"""The YABBE MCP server: every feature of the YABBE video editor as tools an AI
agent can use to edit videos (upload → look → plan → edit → preview → export)."""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote, urlparse

import httpx
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from . import analysis, presets
from .client import Yabbe, YabbeError
from .timeline import Edit, EditError, end, new_clip, uid

INSTRUCTIONS = """\
You are editing videos in YABBE, a browser video editor, through its MCP tools.

Workflow:
1. open_project (or create_project) - it becomes the current project, so later
   tools can leave project_id out.
2. Get footage in: upload_media (local files), import_media_from_url, or
   search_music + add_music for songs. Media is processed in the background
   (upload waits for it by default).
3. Understand it before cutting: look_at_media (frames of a video at chosen
   times), listen_to_media (loudness, silences, onsets, tempo, beats).
4. Edit: add_clip / add_text / add_shape, then move_clip, trim_clip,
   split_clip, delete_clips (ripple to close gaps), set_speed, update_clip
   (transform, opacity, blend, volume, fades, crop, text style ...),
   set_keyframes (animation), set_transition, effects (chroma key, roto,
   motion tracking, stabilize), add_lyrics for word-by-word song lyrics.
5. Check your work: get_timeline, preview_frame / preview_frames.
6. export_video - optionally download_to a local path.

Times are seconds. Timeline time = position in the sequence; a clip shows
its source at in_point + (t - start) * speed. Tracks are listed top layer first;
video tracks hold pictures, audio tracks sound. Positions (x, y) are pixels
from the canvas centre; scale 1 = fit inside the canvas. Every edit can be
reverted with undo. If the user has the project open in the YABBE editor,
tell them to reload the page to see your changes (and not to edit there at the
same time - the editor would save over them).
"""

mcp = MCPServer(
    name="yabbe",
    title="YABBE video editor",
    description="Edit videos with YABBE: media, timeline, effects, lyrics, previews and exports.",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)

_api: Optional[Yabbe] = None
_current: dict[str, Any] = {"project": None, "sequence": {}}
_undo: dict[str, list[dict]] = {}
UNDO_DEPTH = 50


def api() -> Yabbe:
    global _api
    if _api is None:
        _api = Yabbe(os.environ.get("YABBE_URL", "http://localhost:8000"))
    return _api


def tool(read_only: bool = False, destructive: bool = False, title: Optional[str] = None):
    """Register a tool; errors meant for the agent become clean tool errors."""

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except (EditError, YabbeError, ValueError, FileNotFoundError) as exc:
                raise ToolError(str(exc)) from exc

        wrapper.__signature__ = inspect.signature(fn, eval_str=True)  # type: ignore[attr-defined]
        return mcp.tool(title=title, annotations=ToolAnnotations(
            read_only_hint=read_only, destructive_hint=destructive, open_world_hint=False))(wrapper)

    return deco


def _pid(project_id: Optional[str]) -> str:
    pid = project_id or _current["project"]
    if not pid:
        raise EditError("No project open: call open_project or create_project first (or pass project_id)")
    return pid


async def _progress(ctx: Optional[Context], frac: float, msg: str) -> None:
    if ctx is not None:
        try:
            await ctx.report_progress(round(frac * 100, 1), 100, msg)
        except Exception:  # noqa: BLE001 - progress is best effort
            pass


def _reporter(ctx: Optional[Context]):
    async def report(frac: float, msg: str) -> None:
        await _progress(ctx, frac, msg)

    return report


async def _load(project_id: Optional[str], sequence_id: Optional[str] = None) -> Edit:
    pid = _pid(project_id)
    p = await api().project(pid)
    _current["project"] = pid
    return Edit(p, sequence_id or _current["sequence"].get(pid))


async def _save(e: Edit, before: dict) -> None:
    stack = _undo.setdefault(e.project["id"], [])
    stack.append({"sequences": before["sequences"], "main_sequence_id": before["main_sequence_id"]})
    del stack[:-UNDO_DEPTH]
    saved = await api().save_sequences(e.project)
    _current["sequence"][e.project["id"]] = e.sequence_id
    e.project = saved


async def _edit(project_id: Optional[str], fn, sequence_id: Optional[str] = None):
    """Load → change → save (with an undo snapshot). Returns (fn's result, edit)."""
    e = await _load(project_id, sequence_id)
    before = copy.deepcopy(e.project)
    result = fn(e)
    await _save(e, before)
    return result, e


def _src_time(c: dict, at: Optional[float]) -> float:
    """Source time shown at timeline time ``at`` (the clip's first frame if None)."""
    if at is None:
        return c["in_point"]
    if not (c["start"] - 1e-6 <= at <= end(c) + 1e-6):
        raise EditError(f"{at:.2f}s is outside the clip ({c['start']:.2f}–{end(c):.2f}s)")
    return c["in_point"] + (at - c["start"]) * c["speed"]


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


# =============================================================================== server / projects

@tool(read_only=True)
async def yabbe_status() -> str:
    """Is YABBE reachable, how much storage is left, and which optional features
    (roto brush, lyrics, music search) this server has."""
    a = api()
    health = await a.get("/api/health")
    out: dict[str, Any] = {"url": a.base, "health": health, "storage": await a.get("/api/storage")}
    for name, path in (("roto_brush", "/api/roto/info"), ("lyrics", "/api/lyrics/info"), ("music_search", "/api/music/info")):
        try:
            out[name] = (await a.get(path)).get("available")
        except YabbeError:
            out[name] = False
    out["current_project"] = _current["project"]
    return _json(out)


@tool(read_only=True)
async def list_projects() -> str:
    """All projects (id, name, length, size, media count), newest first."""
    rows = await api().get("/api/projects")
    return _json([{k: r[k] for k in ("id", "name", "duration", "width", "height", "asset_count")} for r in rows])


@tool()
async def create_project(name: str, preset: Optional[str] = None, width: Optional[int] = None,
                         height: Optional[int] = None, fps: Optional[float] = None,
                         background: str = "#000000") -> str:
    """Create a project and make it the current one.

    preset: youtube (1920×1080), youtube_4k, 720p, vertical (1080×1920 Reels /
    TikTok / Shorts), square (1080×1080), portrait_4_5 (1080×1350), cinema
    (2.35:1, 24 fps). width / height / fps override it."""
    size = dict(presets.CANVAS.get(preset or "youtube") or {})
    if preset and preset not in presets.CANVAS:
        raise EditError(f"Unknown preset {preset!r}: {', '.join(presets.CANVAS)}")
    size.update({k: v for k, v in (("width", width), ("height", height), ("fps", fps)) if v})
    p = await api().post("/api/projects", {"name": name, "background": background, **size})
    _current["project"] = p["id"]
    return _json(await _overview(Edit(p)))


async def _overview(e: Edit) -> dict:
    return {
        "project": {"id": e.project["id"], "name": e.project["name"]},
        "sequences": [{"id": s["id"], "name": s["name"], "main": s["id"] == e.project["main_sequence_id"],
                       "working": s["id"] == e.sequence_id, **s["settings"],
                       "duration": round(max((end(c) for c in s["clips"]), default=0), 3)}
                      for s in e.project["sequences"]],
        "media": [_media_row(a) for a in e.project["assets"] if not a.get("hidden")],
        "timeline": e.describe(),
    }


def _media_row(a: dict) -> dict:
    row = {k: a.get(k) for k in ("id", "kind", "original_name", "status", "duration") if a.get(k) is not None}
    if a.get("width"):
        row["size"] = f"{a['width']}×{a['height']}"
    if a.get("fps"):
        row["fps"] = a["fps"]
    if a["kind"] == "video":
        row["has_audio"] = a.get("has_audio")
    if a.get("song_title"):
        row["song"] = f"{a.get('song_artist') or '?'} – {a['song_title']}"
    if a.get("error"):
        row["error"] = a["error"]
    return row


@tool(read_only=True)
async def open_project(project_id: str, sequence_id: Optional[str] = None) -> str:
    """Make a project current and show everything about it: its sequences, media
    and the timeline of the working sequence (the main one unless given)."""
    e = await _load(project_id, sequence_id)
    _current["sequence"][e.project["id"]] = e.sequence_id
    return _json(await _overview(e))


@tool()
async def rename_project(name: str, project_id: Optional[str] = None) -> str:
    """Rename a project."""
    p = await api().put(f"/api/projects/{quote(_pid(project_id))}", {"name": name})
    return f"Renamed to “{p['name']}”"


@tool()
async def duplicate_project(project_id: Optional[str] = None) -> str:
    """Copy a project (with its media); the copy becomes current."""
    p = await api().post(f"/api/projects/{quote(_pid(project_id))}/duplicate")
    _current["project"] = p["id"]
    return _json({"id": p["id"], "name": p["name"]})


@tool(destructive=True)
async def delete_project(project_id: str, confirm: bool = False) -> str:
    """Delete a project and all its media and exports. Only with confirm=true,
    after the user agreed."""
    if not confirm:
        raise EditError("Deleting a project can't be undone: ask the user, then call again with confirm=true")
    await api().delete(f"/api/projects/{quote(project_id)}")
    if _current["project"] == project_id:
        _current["project"] = None
    return "Deleted"


@tool()
async def set_canvas(preset: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None,
                     fps: Optional[float] = None, background: Optional[str] = None,
                     project_id: Optional[str] = None) -> str:
    """Change the working sequence's resolution / frame rate / background colour
    (preset as in create_project)."""
    if preset and preset not in presets.CANVAS:
        raise EditError(f"Unknown preset {preset!r}: {', '.join(presets.CANVAS)}")

    def fn(e: Edit):
        s = e.settings
        s.update(presets.CANVAS.get(preset or "", {}))
        for k, v in (("width", width), ("height", height), ("fps", fps), ("background", background)):
            if v is not None:
                s[k] = v
        s["width"] -= s["width"] % 2
        s["height"] -= s["height"] % 2
        return dict(s)

    s, _ = await _edit(project_id, fn)
    return _json(s)


# =============================================================================== sequences

@tool()
async def create_sequence(name: str, preset: Optional[str] = None, switch_to: bool = True,
                          project_id: Optional[str] = None) -> str:
    """Add an empty sequence (another timeline), e.g. a vertical cut-down or a
    part to nest. switch_to makes it the working sequence for later tools."""

    def fn(e: Edit):
        seq = e.new_sequence(name, presets.CANVAS.get(preset or ""))
        if switch_to:
            e.sequence_id = seq["id"]
        return seq

    seq, e = await _edit(project_id, fn)
    return _json({"id": seq["id"], "name": seq["name"], "working": e.sequence_id == seq["id"]})


@tool(read_only=True)
async def switch_sequence(sequence_id: str, project_id: Optional[str] = None) -> str:
    """Work on another sequence: later timeline tools edit it."""
    e = await _load(project_id, sequence_id)
    if e.sequence_id != sequence_id:
        raise EditError(f"No sequence {sequence_id!r}")
    _current["sequence"][e.project["id"]] = sequence_id
    return _json(e.describe())


@tool()
async def set_main_sequence(sequence_id: str, project_id: Optional[str] = None) -> str:
    """The main sequence is what export uses by default and the dashboard shows."""

    def fn(e: Edit):
        if not e._seq(sequence_id):
            raise EditError(f"No sequence {sequence_id!r}")
        e.project["main_sequence_id"] = sequence_id

    await _edit(project_id, fn)
    return "Done"


@tool()
async def rename_sequence(sequence_id: str, name: str, project_id: Optional[str] = None) -> str:
    """Rename a sequence."""

    def fn(e: Edit):
        s = e._seq(sequence_id)
        if not s:
            raise EditError(f"No sequence {sequence_id!r}")
        s["name"] = name

    await _edit(project_id, fn)
    return "Done"


@tool(destructive=True)
async def delete_sequence(sequence_id: str, project_id: Optional[str] = None) -> str:
    """Delete a sequence (not one that's nested somewhere)."""
    await _edit(project_id, lambda e: e.delete_sequence(sequence_id))
    return "Deleted (undo brings it back)"


@tool()
async def nest_clips(clip_ids: list[str], name: Optional[str] = None, project_id: Optional[str] = None) -> str:
    """Pre-compose: move clips into a new sequence and put that sequence in their
    place as one clip (to transform / fade / time them together)."""
    (seq, clip), _ = await _edit(project_id, lambda e: e.nest(clip_ids, name))
    return _json({"sequence_id": seq["id"], "sequence_name": seq["name"], "nested_clip_id": clip["id"]})


@tool()
async def add_sequence_clip(sequence_id: str, start: Optional[float] = None, track: Optional[str] = None,
                            project_id: Optional[str] = None) -> str:
    """Use another sequence as a clip in the working sequence (nesting)."""
    c, e = await _edit(project_id, lambda e: e.add_sequence_clip(sequence_id, start, track))
    return _json(e.describe_clip(c))


# =============================================================================== media

@tool()
async def upload_media(paths: list[str], wait: bool = True, project_id: Optional[str] = None,
                       ctx: Optional[Context] = None) -> str:
    """Upload local video / audio / photo files (paths on this machine) to the
    project. Waits until they're processed unless wait=false."""
    pid = _pid(project_id)
    added = []
    for i, raw in enumerate(paths):
        p = Path(os.path.expanduser(raw))
        if not p.is_file():
            raise FileNotFoundError(f"No file at {p}")

        async def prog(f: float, msg: str, i=i):
            await _progress(ctx, (i + f) / len(paths) * (0.7 if wait else 1), msg)

        added.append(await api().upload(pid, p, prog))
    if wait:
        rows = await api().wait_media(pid, {a["id"] for a in added}, progress=_reporter(ctx))
    else:
        rows = added
    return _json([_media_row(a) for a in rows])


@tool()
async def import_media_from_url(url: str, filename: Optional[str] = None, wait: bool = True,
                                project_id: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    """Download a media file from a direct http(s) link and add it to the project.
    (For YouTube songs use search_music / add_music.)"""
    pid = _pid(project_id)
    if urlparse(url).scheme not in ("http", "https"):
        raise EditError("Only http(s) links")
    name = filename or Path(urlparse(url).path).name or "download"
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / name
        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as h:
            async with h.stream("GET", url) as r:
                if r.status_code >= 400:
                    raise EditError(f"Download failed: {r.status_code}")
                total = int(r.headers.get("content-length") or 0)
                got = 0
                with open(dest, "wb") as fh:
                    async for chunk in r.aiter_bytes(1 << 20):
                        fh.write(chunk)
                        got += len(chunk)
                        if total:
                            await _progress(ctx, got / total * 0.4, f"Downloading {name}")
        asset = await api().upload(pid, dest)
    rows = await api().wait_media(pid, {asset["id"]}) if wait else [asset]
    return _json([_media_row(a) for a in rows])


@tool(read_only=True)
async def list_media(project_id: Optional[str] = None) -> str:
    """The project's media: id, kind, name, length, size, processing status."""
    e = await _load(project_id)
    used: dict[str, int] = {}
    for s in e.project["sequences"]:
        for c in s["clips"]:
            if c.get("asset_id"):
                used[c["asset_id"]] = used.get(c["asset_id"], 0) + 1
    return _json([{**_media_row(a), "used_by_clips": used.get(a["id"], 0)}
                  for a in e.project["assets"] if not a.get("hidden")])


@tool(destructive=True)
async def delete_media(asset_id: str, project_id: Optional[str] = None) -> str:
    """Delete a media file and every clip that uses it (can't be undone)."""
    pid = _pid(project_id)
    await api().delete(f"/api/projects/{quote(pid)}/media/{quote(asset_id)}")
    return "Deleted"


async def _asset_frame(pid: str, asset: dict, src_t: float, height: int = 360) -> bytes:
    """One frame of a media file at source time ``src_t`` (rendered by YABBE)."""
    w, h = int(asset.get("width") or 1920), int(asset.get("height") or 1080)
    w, h = w - w % 2, h - h % 2
    track = {"id": "t_look", "kind": "video", "name": "V"}
    if asset["kind"] == "image":
        clip = new_clip(type="image", asset_id=asset["id"], track_id="t_look", duration=1)
    else:
        clip = new_clip(type="video", asset_id=asset["id"], track_id="t_look", duration=1,
                        in_point=max(0.0, min(src_t, float(asset.get("duration") or 0) - 0.05)))
    body = {"t": 0, "height": min(height, h), "timeline": {
        "settings": {"width": w, "height": h, "fps": asset.get("fps") or 30, "background": "#000000"},
        "tracks": [track], "clips": [clip]}}
    return await api().raw("POST", f"/api/projects/{quote(pid)}/frame", json=body)


@tool(read_only=True)
async def look_at_media(asset_id: str, times: Optional[list[float]] = None, count: int = 6, height: int = 300,
                        project_id: Optional[str] = None) -> list:
    """See a video or photo: frames at the given source times (seconds into the
    file), or ``count`` frames spread over it, as one labelled contact sheet.
    For audio files this describes the sound instead (see listen_to_media)."""
    e = await _load(project_id)
    a = e.asset(asset_id)
    if a is None:
        raise EditError(f"No media {asset_id!r}")
    if a["kind"] == "audio":
        return [await listen_to_media(asset_id, project_id=project_id)]
    if a.get("status") != "ready":
        raise EditError(f"“{a['original_name']}” is still {a.get('status')}")
    if a["kind"] == "image":
        frame = await _asset_frame(e.project["id"], a, 0, max(height, 480))
        return [f"{a['original_name']} ({a['width']}×{a['height']})", Image(data=frame, format="jpeg")]
    dur = float(a.get("duration") or 0)
    ts = times or [dur * (i + 0.5) / max(1, count) for i in range(max(1, min(count, 16)))]
    frames = [(await _asset_frame(e.project["id"], a, t, height), f"{t:.2f}s") for t in ts[:16]]
    sheet = analysis.contact_sheet(frames)
    return [f"{a['original_name']}: {dur:.2f}s, {a.get('width')}×{a.get('height')}, {a.get('fps')} fps, "
            f"{'with' if a.get('has_audio') else 'no'} sound. Frames at {', '.join(f'{t:.2f}s' for t in ts[:16])}:",
            Image(data=sheet, format="jpeg")]


@tool(read_only=True)
async def listen_to_media(asset_id: str, start: float = 0.0, end_time: Optional[float] = None,
                          project_id: Optional[str] = None) -> str:
    """Hear a file's sound, roughly: loudness every half second (0..1), silent
    stretches, onsets (hits / cuts), and for music an estimated tempo with beat
    times - handy for cutting to the beat or trimming silences. Times are
    seconds into the file."""
    pid = _pid(project_id)
    e = await _load(pid)
    a = e.asset(asset_id)
    if a is None:
        raise EditError(f"No media {asset_id!r}")
    if not a.get("has_audio"):
        raise EditError(f"“{a['original_name']}” has no sound")
    peaks = await api().raw("GET", f"/api/projects/{quote(pid)}/media/{quote(asset_id)}/waveform")
    out = analysis.analyse_peaks(peaks, start, end_time)
    out = {"file": a["original_name"], "duration": a.get("duration"), **out}
    if a.get("song_title"):
        out["song"] = f"{a.get('song_artist') or '?'} – {a['song_title']}"
    return _json(out)


@tool(read_only=True)
async def search_music(query: str) -> str:
    """Search YouTube for music (for add_music)."""
    rows = await api().get("/api/music/search", params={"q": query})
    return _json(rows)


@tool()
async def add_music(video_id: str, project_id: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    """Add a YouTube video's audio to the project as a high quality MP3 (only
    music the user may use). Returns the new media."""
    pid = _pid(project_id)
    job = await api().post(f"/api/projects/{quote(pid)}/media/youtube", {"video_id": video_id})
    done = await api().wait_job(job["id"], _reporter(ctx))
    aid = (done.get("result") or {}).get("asset_id")
    rows = await api().wait_media(pid, {aid}) if aid else []
    return _json([_media_row(a) for a in rows])


# =============================================================================== looking at the edit

@tool(read_only=True)
async def get_timeline(project_id: Optional[str] = None, sequence_id: Optional[str] = None) -> str:
    """The working sequence: settings, length, and every track (top layer first)
    with its clips (ids, times, source range, transform, effects, ...)."""
    e = await _load(project_id, sequence_id)
    return _json(e.describe())


@tool(read_only=True)
async def get_clip(clip_id: str, project_id: Optional[str] = None) -> str:
    """Everything about one clip (the full JSON YABBE stores)."""
    e = await _load(project_id)
    return _json(e.clip(clip_id))


async def _frame(e: Edit, t: float, height: int) -> bytes:
    body = {"t": max(0.0, t), "height": height, "timeline": {"sequence_id": e.sequence_id}}
    return await api().raw("POST", f"/api/projects/{quote(e.project['id'])}/frame", json=body)


@tool(read_only=True)
async def preview_frame(t: float, height: int = 540, project_id: Optional[str] = None) -> list:
    """Render the edit at timeline time t (what the export will look like there)."""
    e = await _load(project_id)
    return [f"{e.seq['name']} at {t:.2f}s", Image(data=await _frame(e, t, height), format="jpeg")]


@tool(read_only=True)
async def preview_frames(times: Optional[list[float]] = None, start: float = 0.0, end_time: Optional[float] = None,
                         count: int = 8, height: int = 300, project_id: Optional[str] = None) -> list:
    """Render several moments of the edit as one contact sheet: the given times,
    or ``count`` evenly spread between start and end_time (default: all of it)."""
    e = await _load(project_id)
    stop = end_time if end_time is not None else e.duration()
    if not times:
        n = max(1, min(count, 16))
        times = [start + (stop - start) * (i + 0.5) / n for i in range(n)]
    frames = [(await _frame(e, t, height), f"{t:.2f}s") for t in times[:16]]
    return [f"{e.seq['name']}: {len(frames)} frames", Image(data=analysis.contact_sheet(frames), format="jpeg")]


# =============================================================================== adding clips

@tool()
async def add_clip(asset_id: str, start: Optional[float] = None, track: Optional[str] = None,
                   in_point: float = 0.0, duration: Optional[float] = None,
                   project_id: Optional[str] = None) -> str:
    """Put media on the timeline. start: timeline seconds (default: after the last
    clip of its kind, i.e. append). track: id, name or number (default: a free
    track of the right kind, made if needed). in_point: where in the file to
    start; duration: how long (default: to the end of the file; photos 5s)."""
    c, e = await _edit(project_id, lambda e: e.add_media(asset_id, start, track, in_point, duration))
    return _json(e.describe_clip(c))


@tool()
async def add_text(content: str, start: float = 0.0, duration: float = 5.0, track: Optional[str] = None,
                   font: Optional[str] = None, size: Optional[int] = None, color: Optional[str] = None,
                   background: Optional[str] = None, stroke_color: Optional[str] = None,
                   stroke_width: Optional[int] = None, bold: Optional[bool] = None, italic: Optional[bool] = None,
                   align: Optional[Literal["left", "center", "right"]] = None, x: float = 0.0, y: float = 0.0,
                   project_id: Optional[str] = None) -> str:
    """Add a text clip (titles, captions, labels). Colours are #rrggbb or
    #rrggbbaa; background puts a box behind it; x / y move it from the centre
    (pixels). See list_fonts. Animate it with set_keyframes."""
    style = {k: v for k, v in dict(content=content, font=font, size=size, color=color, background=background,
                                   stroke_color=stroke_color, stroke_width=stroke_width, bold=bold, italic=italic,
                                   align=align).items() if v is not None}

    def fn(e: Edit):
        c = e.add_text(style, start, duration, track)
        c["transform"].update(x=x, y=y)
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool()
async def add_shape(kind: Literal["rectangle", "ellipse", "triangle", "polygon", "star", "line", "arrow"],
                    start: float = 0.0, duration: float = 5.0, width: float = 400, height: float = 400,
                    fill: Optional[str] = "#7c5cff", stroke: Optional[str] = None, stroke_width: float = 0,
                    radius: float = 0, sides: int = 6, x: float = 0.0, y: float = 0.0, track: Optional[str] = None,
                    project_id: Optional[str] = None) -> str:
    """Add a vector shape (boxes behind text, highlights, arrows, frames).
    fill / stroke: #rrggbb[aa] or null; radius: rectangle corner rounding
    (0..0.5); sides: polygon sides / star points. Lines and arrows use stroke."""
    style = dict(kind=kind, width=width, height=height, fill=fill, stroke=stroke, stroke_width=stroke_width,
                 radius=radius, sides=sides)
    if kind in ("line", "arrow") and not stroke:
        style.update(stroke=fill or "#ffffff", fill=None, stroke_width=stroke_width or 8)

    def fn(e: Edit):
        c = e.add_shape(style, start, duration, track)
        c["transform"].update(x=x, y=y)
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool(read_only=True)
async def list_fonts() -> str:
    """Fonts available for text and lyrics."""
    return _json(await api().get("/api/fonts"))


# =============================================================================== changing clips

@tool()
async def move_clip(clip_id: str, start: Optional[float] = None, track: Optional[str] = None,
                    with_linked: bool = True, project_id: Optional[str] = None) -> str:
    """Move a clip to another time and/or track (linked clips, e.g. a video's
    separated audio or a song's lyrics, move along)."""
    ids, e = await _edit(project_id, lambda e: e.move(clip_id, start, track, with_linked))
    return _json([e.describe_clip(e.clip(i)) for i in ids])


@tool()
async def trim_clip(clip_id: str, start: Optional[float] = None, end_time: Optional[float] = None,
                    with_linked: bool = True, project_id: Optional[str] = None) -> str:
    """Move a clip's edges on the timeline: start = new start time (trims or
    extends the beginning; the footage stays in place), end_time = new end time.
    Can't extend past the source footage."""
    ids, e = await _edit(project_id, lambda e: e.trim(clip_id, start, end_time, with_linked))
    return _json([e.describe_clip(e.clip(i)) for i in ids])


@tool()
async def set_clip_source(clip_id: str, in_point: float, project_id: Optional[str] = None) -> str:
    """Slip: show a different part of the footage in the same slot (in_point =
    source seconds at the clip's start)."""

    def fn(e: Edit):
        c = e.editable(e.clip(clip_id))
        if c["type"] not in ("video", "audio", "sequence", "lyrics"):
            raise EditError("Only video / audio / nested clips have footage to slip")
        if in_point < 0 or in_point + c["duration"] * c["speed"] > e.source_length(c) + 1e-3:
            raise EditError(f"The footage is {e.source_length(c):.2f}s long: in_point must be 0–"
                            f"{e.source_length(c) - c['duration'] * c['speed']:.2f}")
        for i in e.linked([clip_id]):
            x = e.clip(i)
            if x.get("asset_id") == c.get("asset_id"):
                x["in_point"] = in_point
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool()
async def split_clip(clip_id: str, at: float, with_linked: bool = True, project_id: Optional[str] = None) -> str:
    """Cut a clip in two at timeline time ``at``. Returns the left and right ids."""
    pairs, _ = await _edit(project_id, lambda e: e.split(clip_id, at, with_linked))
    return _json([{"left": a, "right": b} for a, b in pairs])


@tool(destructive=True)
async def delete_clips(clip_ids: list[str], ripple: bool = False, with_linked: bool = True,
                       project_id: Optional[str] = None) -> str:
    """Remove clips. ripple=true closes the gaps (later clips on those tracks
    move earlier)."""
    gone, _ = await _edit(project_id, lambda e: e.delete(clip_ids, ripple, with_linked))
    return f"Removed {len(gone)} clip(s)"


@tool()
async def duplicate_clip(clip_id: str, start: Optional[float] = None, track: Optional[str] = None,
                         project_id: Optional[str] = None) -> str:
    """Copy a clip (with all its settings) to another time / track (default:
    right after it)."""
    c, e = await _edit(project_id, lambda e: e.duplicate(clip_id, start, track))
    return _json(e.describe_clip(c))


@tool()
async def set_speed(clip_id: str, speed: Optional[float] = None, duration: Optional[float] = None,
                    project_id: Optional[str] = None) -> str:
    """Slow motion / fast forward (0.25–4×): give a speed, or the duration the
    clip's footage should play in. Keyframes and markers stay on the same moments."""

    def fn(e: Edit):
        c = e.clip(clip_id)
        s = speed if speed is not None else (c["duration"] * c["speed"] / duration if duration else None)
        if not s:
            raise EditError("Give speed or duration")
        return e.set_speed(clip_id, s)

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool()
async def update_clip(clip_id: str, changes: dict, project_id: Optional[str] = None) -> str:
    """Change any clip setting by merging ``changes`` into it. Common ones:

    transform: {x, y (pixels from centre), scale (1 = fit), rotation (degrees),
      opacity (0..1), flip_h, flip_v}
    volume (0..4), muted, fade_in / fade_out (seconds; sound and picture),
    blend: normal, multiply, screen, overlay, soft_light, add, darken, lighten,
      difference, color_dodge, color_burn, hard_light, ... (Photoshop names)
    crop: {left, top, right, bottom} fractions 0..1
    text (text clips): {content, font, size, color, background, padding,
      stroke_color, stroke_width, align, bold, italic, line_spacing}
    shape (shape clips): {kind, width, height, fill, stroke, stroke_width, radius, sides, inner}
    markers: [{t, label, color}]

    A property with keyframes ignores its static value: use set_keyframes or
    clear them first. Times / positions: see move_clip, trim_clip, set_speed."""

    def fn(e: Edit):
        c = e.patch(clip_id, changes)
        cr = c.get("crop") or {}
        if "crop" in changes and any(cr.get(k) for k in ("left", "top", "right", "bottom")):
            e.add_effect(c, "crop")
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool()
async def set_keyframes(clip_id: str, property: str, keyframes: list[dict], project_id: Optional[str] = None) -> str:
    """Animate a property. keyframes: [{t, v, ease}] (t = seconds from the
    clip's start; colour properties use "c": "#rrggbb" instead of v). An empty
    list removes the animation.

    properties: x, y, scale, rotation, opacity, volume; text: text_size,
    text_stroke_width, text_padding, text_line_spacing, text_color,
    text_stroke_color, text_background; shapes: shape_width, shape_height,
    shape_stroke_width, shape_radius, shape_fill, shape_stroke.
    ease (shape of the move to the next key): linear, hold, ease_in, ease_out,
    ease_in_out, back_in/out/in_out (overshoot), elastic_in/out/in_out,
    bounce_in/out/in_out."""
    allowed = {"x", "y", "scale", "rotation", "opacity", "volume", "text_size", "text_stroke_width",
               "text_padding", "text_line_spacing", "text_color", "text_stroke_color", "text_background",
               "shape_width", "shape_height", "shape_stroke_width", "shape_radius", "shape_fill", "shape_stroke"}
    if property not in allowed:
        raise EditError(f"Can't animate {property!r}; one of: {', '.join(sorted(allowed))}")

    def fn(e: Edit):
        c = e.editable(e.clip(clip_id))
        kf = c.setdefault("keyframes", {})
        clean = []
        for k in keyframes:
            item = {"t": float(k["t"]), "ease": k.get("ease", "linear")}
            if "c" in k:
                item["c"] = k["c"]
            else:
                item["v"] = float(k["v"])
            clean.append(item)
        if clean:
            kf[property] = sorted(clean, key=lambda k: k["t"])
        else:
            kf.pop(property, None)
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool(read_only=True)
async def list_transitions() -> str:
    """Transition kinds (ids and categories) for set_transition."""
    cat = await api().get("/api/transitions")
    by: dict[str, list[str]] = {}
    for t in cat["transitions"]:
        by.setdefault(t["category"], []).append(t["id"])
    return _json({"by_category": by, "default_duration": cat["default_duration"]})


@tool()
async def set_transition(clip_id: str, kind: Optional[str] = "mix", duration: float = 0.5, audio: bool = True,
                         project_id: Optional[str] = None) -> str:
    """Transition from this clip into the next clip that touches it on the same
    video track, centred on the cut (kind=null removes it). audio crossfades
    the two clips' sound too. See list_transitions (mix = crossfade)."""
    tr, _ = await _edit(project_id, lambda e: e.set_transition(clip_id, kind, duration, audio))
    return _json(tr) if tr else "Removed"


@tool()
async def add_marker(clip_id: str, at: float, label: str = "", color: str = "#f2b84b",
                     project_id: Optional[str] = None) -> str:
    """Mark a moment on a clip (at = timeline time). Markers stay on the footage
    through moves, trims and speed changes."""

    def fn(e: Edit):
        c = e.clip(clip_id)
        m = {"id": uid("m_"), "t": round(at - c["start"], 4), "label": label, "color": color}
        c.setdefault("markers", []).append(m)
        return m

    m, _ = await _edit(project_id, fn)
    return _json(m)


@tool()
async def remove_marker(clip_id: str, marker_id: str, project_id: Optional[str] = None) -> str:
    """Remove a marker."""

    def fn(e: Edit):
        c = e.clip(clip_id)
        c["markers"] = [m for m in c.get("markers") or [] if m["id"] != marker_id]

    await _edit(project_id, fn)
    return "Removed"


# =============================================================================== tracks / links / audio

@tool()
async def add_track(kind: Literal["video", "audio"], name: Optional[str] = None,
                    position: Literal["top", "bottom"] = "top", project_id: Optional[str] = None) -> str:
    """Add a track. Video tracks higher in the list draw on top."""
    t, _ = await _edit(project_id, lambda e: e.add_track(kind, name, position))
    return _json(t)


@tool()
async def update_track(track: str, name: Optional[str] = None, muted: Optional[bool] = None,
                       hidden: Optional[bool] = None, locked: Optional[bool] = None,
                       project_id: Optional[str] = None) -> str:
    """Rename / mute / hide / lock a track (id, name or number)."""

    def fn(e: Edit):
        t = e.track(track)
        for k, v in (("name", name), ("muted", muted), ("hidden", hidden), ("locked", locked)):
            if v is not None:
                t[k] = v
        return t

    t, _ = await _edit(project_id, fn)
    return _json(t)


@tool(destructive=True)
async def remove_track(track: str, project_id: Optional[str] = None) -> str:
    """Remove a track and the clips on it."""

    def fn(e: Edit):
        t = e.track(track)
        e.delete([c["id"] for c in e.clips if c["track_id"] == t["id"]], with_linked=False)
        e.seq["tracks"] = [x for x in e.tracks if x["id"] != t["id"]]

    await _edit(project_id, fn)
    return "Removed"


@tool()
async def move_track(track: str, to_position: int, project_id: Optional[str] = None) -> str:
    """Reorder tracks: to_position 1 = top (drawn above everything)."""

    def fn(e: Edit):
        t = e.track(track)
        tracks = [x for x in e.tracks if x["id"] != t["id"]]
        tracks.insert(max(0, min(len(tracks), to_position - 1)), t)
        e.seq["tracks"] = tracks
        return [x["name"] for x in tracks]

    order, _ = await _edit(project_id, fn)
    return _json(order)


@tool()
async def link_clips(clip_ids: list[str], project_id: Optional[str] = None) -> str:
    """Link clips so they move / trim / split / delete together."""
    link, _ = await _edit(project_id, lambda e: e.link(clip_ids))
    return f"Linked ({link})"


@tool()
async def unlink_clips(clip_ids: list[str], project_id: Optional[str] = None) -> str:
    """Unlink clips (each can then be edited on its own)."""
    await _edit(project_id, lambda e: e.unlink(clip_ids))
    return "Unlinked"


@tool()
async def separate_audio(clip_id: str, project_id: Optional[str] = None) -> str:
    """Put a video clip's sound on its own audio clip (linked), e.g. to trim or
    fade it separately (J/L cuts) or delete it."""
    c, e = await _edit(project_id, lambda e: e.separate_audio(clip_id))
    return _json(e.describe_clip(c))


@tool()
async def freeze_frame(clip_id: str, at: float, duration: float = 2.0, project_id: Optional[str] = None) -> str:
    """Hold the frame at timeline time ``at`` for ``duration`` seconds: the clip
    is split there and a still of that frame goes between the halves (the rest
    of the track moves later)."""
    e = await _load(project_id)
    c = e.clip(clip_id)
    if c["type"] != "video" or c.get("hold"):
        raise EditError("Freeze frames come from video clips")
    src = _src_time(c, at)
    img = await api().post(f"/api/projects/{quote(e.project['id'])}/media/freeze",
                           {"asset_id": c["asset_id"], "t": src})
    await api().wait_media(e.project["id"], {img["id"]})
    still, e = await _edit(project_id, lambda e: e.freeze(clip_id, at, img["id"], duration))
    return _json(e.describe_clip(still))


# =============================================================================== effects

@tool()
async def set_chroma_key(clip_id: str, enabled: bool = True, color: str = "#00b140", clip_black: float = 0.15,
                         clip_white: float = 0.9, spill: float = 0.6, choke: float = 0.0, feather: float = 0.0,
                         project_id: Optional[str] = None) -> str:
    """Remove a green / blue screen (color = the screen colour; pick it from a
    frame). clip_black up = cleaner background, clip_white down = more solid
    subject, spill removes the screen's colour cast, choke / feather tidy edges.
    enabled=false removes the effect."""

    def fn(e: Edit):
        c = e.editable(e.clip(clip_id))
        if c["type"] not in ("video", "image", "sequence"):
            raise EditError("Chroma key works on video, photo and nested clips")
        if not enabled:
            c["chroma_key"] = None
            e.remove_effect(c, "chroma_key")
        else:
            c["chroma_key"] = {"enabled": True, "color": color, "clip_black": clip_black, "clip_white": clip_white,
                               "spill": spill, "choke": choke, "feather": feather}
            e.add_effect(c, "chroma_key")
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


def _prompt(c: dict, at: Optional[float], box: Optional[list[float]], points: Optional[list[list[float]]]) -> dict:
    if not box and not points:
        raise EditError("Give a box [x0, y0, x1, y1] and/or points [[x, y, 1=object | 0=not]] (0..1 of the frame)")
    return {"t": _src_time(c, at), "box": box, "points": points or []}


@tool(read_only=True)
async def roto_preview(clip_id: str, at: Optional[float] = None, box: Optional[list[float]] = None,
                       points: Optional[list[list[float]]] = None, project_id: Optional[str] = None) -> list:
    """Check a roto brush selection on one frame before cutting it out: shows the
    frame with the selected object tinted. box = [x0, y0, x1, y1] and points =
    [[x, y, 1 (object) or 0 (not)]] as fractions of the frame (0..1); at =
    timeline time (default: the clip's first frame). Use look_at_media first to
    find the object."""
    e = await _load(project_id)
    c = e.clip(clip_id)
    a = e.asset(c.get("asset_id"))
    if c["type"] != "video" or not a:
        raise EditError("The roto brush works on video clips")
    pr = _prompt(c, at, box, points)
    mask = await api().raw("POST", f"/api/projects/{quote(e.project['id'])}/roto/preview",
                           json={"asset_id": a["id"], "prompt": pr})
    frame = await _asset_frame(e.project["id"], a, pr["t"], 480)
    return [f"Selection at source {pr['t']:.2f}s (tinted pink):", Image(data=analysis.overlay_mask(frame, mask), format="jpeg")]


@tool()
async def roto_cut_out(clip_id: str, at: Optional[float] = None, box: Optional[list[float]] = None,
                       points: Optional[list[list[float]]] = None, invert: bool = False, choke: float = 0.0,
                       feather: float = 0.0, wait: bool = True, project_id: Optional[str] = None,
                       ctx: Optional[Context] = None) -> str:
    """Roto brush: cut a person / object out of a video clip (AI segmentation,
    followed through the whole clip) - the rest becomes transparent, so what's
    below shows through. Select it like roto_preview. invert keeps everything
    except the object. Takes about a second per frame."""

    def fn(e: Edit):
        c = e.editable(e.clip(clip_id))
        if c["type"] != "video" or c.get("hold"):
            raise EditError("The roto brush works on video clips")
        c["roto"] = {"enabled": True, "prompts": [_prompt(c, at, box, points)], "start": c["in_point"],
                     "end": c["in_point"] + c["duration"] * c["speed"], "invert": invert, "refine": True,
                     "choke": choke, "feather": feather}
        e.add_effect(c, "roto")
        return c

    c, e = await _edit(project_id, fn)
    st = await api().post(f"/api/projects/{quote(e.project['id'])}/roto/run",
                          {"clip_id": clip_id, "asset_id": c["asset_id"], "roto": c["roto"]})
    if wait and st["state"] in ("queued", "tracking"):
        while st["state"] in ("queued", "tracking"):
            await _progress(ctx, st.get("progress", 0), "Following the object")
            await asyncio.sleep(2)
            [st] = await api().post(f"/api/projects/{quote(e.project['id'])}/roto/status",
                                    [{"clip_id": clip_id, "asset_id": c["asset_id"], "roto": c["roto"]}])
    if st["state"] == "error":
        raise EditError(f"Roto failed: {st.get('error')}")
    return f"Roto brush {st['state']} on {clip_id}" + ("" if st["state"] == "done" else " (still working; it shows when done)")


@tool()
async def remove_effect(clip_id: str, effect: Literal["crop", "speed", "chroma_key", "roto", "tracking", "stabilize"],
                        project_id: Optional[str] = None) -> str:
    """Take an effect off a clip (resets its settings)."""

    def fn(e: Edit):
        c = e.editable(e.clip(clip_id))
        if effect == "crop":
            c["crop"] = {"left": 0, "top": 0, "right": 0, "bottom": 0, "enabled": True}
        elif effect == "speed":
            e.set_speed(clip_id, 1.0)
        elif effect == "chroma_key":
            c["chroma_key"] = None
        elif effect == "roto":
            c["roto"] = None
        elif effect in ("tracking", "stabilize"):
            gone = {t["id"] for t in c.get("trackers") or [] if (t["kind"] == "stabilize") == (effect == "stabilize")}
            c["trackers"] = [t for t in c.get("trackers") or [] if t["id"] not in gone]
            if effect == "stabilize":
                c["stabilize"] = None
            for x in e.clips:
                for k in ("follow", "pin"):
                    if (x.get(k) or {}).get("clip_id") == clip_id and x[k]["tracker_id"] in gone:
                        x[k] = None
        e.remove_effect(c, effect)
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


async def _run_tracker(e: Edit, c: dict, tracker: dict, wait: bool, ctx: Optional[Context]) -> dict:
    a = api()
    pid = e.project["id"]
    body = {"asset_id": c["asset_id"], "tracker": tracker}
    st = await a.post(f"/api/projects/{quote(pid)}/tracking/run", body)
    if wait:
        while st["state"] in ("queued", "tracking"):
            await _progress(ctx, st.get("progress", 0), "Tracking")
            await asyncio.sleep(1.5)
            [st] = await a.post(f"/api/projects/{quote(pid)}/tracking/status", [body])
    if st["state"] == "error":
        raise EditError(f"Tracking failed: {st.get('error')}")
    return st


@tool()
async def track_motion(clip_id: str, kind: Literal["point", "transform", "corner_pin"] = "point",
                       at: Optional[float] = None, box: Optional[list[float]] = None,
                       quad: Optional[list[list[float]]] = None, quality: Literal["fast", "precise"] = "fast",
                       name: str = "", wait: bool = True, project_id: Optional[str] = None,
                       ctx: Optional[Context] = None) -> str:
    """Motion tracking on a video clip, to make other clips follow something.

    kind: point (position), transform (position + rotation + scale), corner_pin
    (a flat surface's four corners - for screen / sign replacements).
    box = [cx, cy, w, h] around the thing to follow (fractions 0..1 of the
    frame, centre + size); corner_pin uses quad = 4 corners [[x, y], ...]
    (top-left, top-right, bottom-right, bottom-left). at = timeline time of
    the frame those are on. Then use attach_to_tracker / pin_to_tracker."""
    e = await _load(project_id)
    c = e.clip(clip_id)
    if c["type"] != "video" or c.get("hold"):
        raise EditError("Motion tracking works on video clips")
    if kind == "corner_pin" and not quad:
        raise EditError("corner_pin needs quad (four corners)")
    if kind != "corner_pin" and not box:
        raise EditError("Give box = [cx, cy, w, h] around what to follow")
    n = sum(1 for t in c.get("trackers") or [] if t["kind"] == kind) + 1
    tracker = {"id": uid("k_"), "name": name or f"{kind.replace('_', ' ').title()} {n}", "kind": kind,
               "ref": _src_time(c, at), "box": box or [0.5, 0.5, 0.08, 0.08],
               "quad": quad or [[0.35, 0.3], [0.65, 0.3], [0.65, 0.7], [0.35, 0.7]],
               "start": c["in_point"], "end": c["in_point"] + c["duration"] * c["speed"], "quality": quality}

    def fn(e: Edit):
        x = e.editable(e.clip(clip_id))
        x.setdefault("trackers", []).append(tracker)
        e.add_effect(x, "tracking")
        return x

    c, e = await _edit(project_id, fn)
    st = await _run_tracker(e, c, tracker, wait, ctx)
    return _json({"tracker_id": tracker["id"], "name": tracker["name"], "state": st["state"]})


@tool()
async def attach_to_tracker(clip_id: str, tracked_clip_id: str, tracker_id: str, position: bool = True,
                            rotation: bool = False, scale: bool = False, project_id: Optional[str] = None) -> str:
    """Make a clip (text, sticker, shape, video) follow a point / transform
    tracker. Its current transform is where it sits on the tracker's reference
    frame. Pass tracker_id="" to detach."""

    def fn(e: Edit):
        c = e.editable(e.clip(clip_id))
        if not tracker_id:
            c["follow"] = None
            return c
        src = e.clip(tracked_clip_id)
        t = next((t for t in src.get("trackers") or [] if t["id"] == tracker_id), None)
        if t is None or t["kind"] not in ("point", "transform"):
            raise EditError("That isn't a point / transform tracker on that clip")
        c.update(follow={"clip_id": tracked_clip_id, "tracker_id": tracker_id, "position": position,
                         "rotation": rotation, "scale": scale}, pin=None)
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool()
async def pin_to_tracker(clip_id: str, tracked_clip_id: str, tracker_id: str,
                         project_id: Optional[str] = None) -> str:
    """Corner-pin a clip (e.g. a screen recording or logo) onto a corner_pin
    tracker's surface, with perspective. tracker_id="" unpins."""

    def fn(e: Edit):
        c = e.editable(e.clip(clip_id))
        if not tracker_id:
            c["pin"] = None
            return c
        src = e.clip(tracked_clip_id)
        if not any(t["id"] == tracker_id and t["kind"] == "corner_pin" for t in src.get("trackers") or []):
            raise EditError("That isn't a corner_pin tracker on that clip")
        c.update(pin={"clip_id": tracked_clip_id, "tracker_id": tracker_id}, follow=None)
        return c

    c, e = await _edit(project_id, fn)
    return _json(e.describe_clip(c))


@tool()
async def stabilize(clip_id: str, mode: Literal["smooth", "lock"] = "smooth", smoothness: float = 1.0,
                    rotation: bool = True, scale: bool = False, auto_zoom: bool = True, wait: bool = True,
                    project_id: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    """Remove camera shake. smooth keeps the camera's move but smooths it
    (smoothness = seconds averaged); lock makes the shot look like a tripod.
    auto_zoom scales up just enough to hide the moving edges."""
    e = await _load(project_id)
    c = e.clip(clip_id)
    if c["type"] != "video" or c.get("hold"):
        raise EditError("Stabilize works on video clips")
    a = e.asset(c["asset_id"]) or {}
    aspect = (a.get("width") or 16) / (a.get("height") or 9)
    tracker = next((t for t in c.get("trackers") or [] if t["kind"] == "stabilize"), None) or {
        "id": uid("k_"), "name": "Stabilize", "kind": "stabilize", "ref": c["in_point"],
        "box": [0.5, 0.5, 0.08, min(0.5, 0.08 * aspect)], "quad": [[0.35, 0.3], [0.65, 0.3], [0.65, 0.7], [0.35, 0.7]],
        "start": c["in_point"], "end": c["in_point"] + c["duration"] * c["speed"], "quality": "fast"}

    def fn(e: Edit):
        x = e.editable(e.clip(clip_id))
        if not any(t["id"] == tracker["id"] for t in x.get("trackers") or []):
            x.setdefault("trackers", []).append(tracker)
        x["stabilize"] = {"tracker_id": tracker["id"], "enabled": True, "mode": mode, "smoothness": smoothness,
                          "rotation": rotation, "scale": scale, "auto_zoom": auto_zoom}
        e.add_effect(x, "stabilize")
        return x

    c, e = await _edit(project_id, fn)
    st = await _run_tracker(e, c, tracker, wait, ctx)
    return f"Stabilize {st['state']} on {clip_id}"


# =============================================================================== lyrics

@tool(read_only=True)
async def search_lyrics(query: str, duration: float = 0) -> str:
    """Search LRCLIB for a song's lyrics (to pick a specific version for
    add_lyrics: pass its id as lrclib_id). duration helps rank (seconds)."""
    return _json(await api().get("/api/lyrics/search", params={"q": query, "duration": duration}))


@tool()
async def add_lyrics(song_clip_id: str, preset: str = "karaoke", style: Optional[dict] = None,
                     lrclib_id: Optional[int] = None, lyrics_text: Optional[str] = None, isolate_vocals: bool = False,
                     wait: bool = True, project_id: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    """Show a song's lyrics word by word on the video (like Instagram's lyrics
    stickers). The lyrics are found automatically (or give lrclib_id from
    search_lyrics, or lyrics_text: plain or LRC), every word is timed to the
    song (first time: a minute or two), and a lyrics clip linked to the song is
    added on a video track.

    preset: karaoke, highlight, typewriter, pop, bounce, big, scroll, stack,
    verse, neon, boxed, subtitle. style overrides any setting - see
    set_lyrics_style. isolate_vocals = more accurate on loud mixes, slower."""
    style_dict = presets.lyrics_style(preset, style)

    def fn(e: Edit):
        song = e.editable(e.clip(song_clip_id))
        a = e.asset(song.get("asset_id"))
        if song["type"] not in ("audio", "video") or not a or not a.get("has_audio") or song.get("audio_detached"):
            raise EditError("Pick a clip with sound (a song or a video with sound)")
        link = song.get("link") or uid("l_")
        song["link"] = link
        old = [c for c in e.clips if c["type"] == "lyrics" and c.get("link") == link and c.get("asset_id") == a["id"]]
        if old:
            return old[0]
        # lyrics sit on top: the top track if it's free there, else a new one
        top = e.tracks[0] if e.tracks else None
        if top and top["kind"] == "video" and not top.get("locked") and \
                not e.overlaps(top["id"], song["start"], song["duration"]):
            track = top
        else:
            track = e.add_track("video", "Lyrics", "top")
        c = new_clip(type="lyrics", asset_id=a["id"], track_id=track["id"], start=song["start"],
                     duration=song["duration"], in_point=song["in_point"], speed=song["speed"], link=link,
                     lyrics={"lines": [], "style": style_dict, "source": None, "isolate": isolate_vocals})
        e.clips.append(c)
        return c

    lc, e = await _edit(project_id, fn)
    if lc.get("lyrics", {}).get("lines") and lrclib_id is None and lyrics_text is None:
        return f"This song already has lyrics: {lc['id']} (use set_lyrics_style / remove_lyrics)"
    return await _compute_lyrics(e, lc, isolate_vocals, lrclib_id, lyrics_text, wait, ctx)


async def _compute_lyrics(e: Edit, lc: dict, isolate: bool, lrclib_id: Optional[int], text: Optional[str],
                          wait: bool, ctx: Optional[Context]) -> str:
    pid = e.project["id"]
    body = {"isolate": isolate, "lrclib_id": lrclib_id, "text": text}
    job = await api().post(f"/api/projects/{quote(pid)}/lyrics/{quote(lc['asset_id'])}", body)
    if not wait:
        return _json({"lyrics_clip_id": lc["id"], "job_id": job["id"],
                      "note": "Timing in the background; call finish_lyrics with the job id when it's done"})
    done = await api().wait_job(job["id"], _reporter(ctx))
    return await _apply_lyrics(pid, lc["id"], done.get("result") or {}, isolate)


async def _apply_lyrics(pid: str, clip_id: str, result: dict, isolate: bool) -> str:
    def fn(e: Edit):
        c = e.clip(clip_id)
        c["lyrics"].update(lines=result.get("lines") or [], source=result.get("source"), isolate=isolate)
        return c

    c, e = await _edit(pid, fn)
    lines = c["lyrics"]["lines"]
    guessed = sum(1 for ln in lines if ln["words"] and all(not w.get("conf") for w in ln["words"]))
    return _json({"lyrics_clip_id": clip_id, "source": c["lyrics"]["source"], "lines": len(lines),
                  "words": sum(len(ln["words"]) for ln in lines), "lines_timed_by_line_only": guessed,
                  "first_lines": [f"{ln['start']:.2f}s {ln['text']}" for ln in lines[:4]]})


@tool()
async def finish_lyrics(lyrics_clip_id: str, job_id: str, project_id: Optional[str] = None,
                        ctx: Optional[Context] = None) -> str:
    """Wait for a lyrics job started with wait=false and put the result on the clip."""
    done = await api().wait_job(job_id, _reporter(ctx))
    e = await _load(project_id)
    return await _apply_lyrics(e.project["id"], lyrics_clip_id, done.get("result") or {},
                               bool(e.clip(lyrics_clip_id)["lyrics"].get("isolate")))


@tool()
async def set_lyrics_style(lyrics_clip_id: str, preset: Optional[str] = None, style: Optional[dict] = None,
                           project_id: Optional[str] = None) -> str:
    """Restyle lyrics: a preset (karaoke, highlight, typewriter, pop, bounce, big,
    scroll, stack, verse, neon, boxed, subtitle) and/or any settings:

    layout: line | stack (line + next) | scroll | words (a few at a time) | page (verse)
    lines_shown, words_shown; word_fx (sung word): fill | highlight | reveal |
    fade | pop | bounce | glow | none; line_in / line_out: none | fade |
    slide_up | slide_down | slide_left | drop | zoom | blur; transition_ms;
    font, size, bold, italic, case (as_is | upper | lower | title),
    letter_spacing, line_spacing; colours #rrggbb[aa]: color (sung),
    upcoming_color, past_color, active_color, outline_color, shadow_color,
    box_color, active_box_color; other_lines_opacity, outline, shadow, blur;
    box: none | line | words | active (the sung word), box_padding;
    active_scale; align: left | center | right; anchor: top | middle | bottom;
    margin (0..0.5 of the height), width (0.1..1 of the canvas width);
    offset_ms (shift all lyrics), lead_ms (show early), hold_ms (keep after).
    Move / scale / rotate the whole block with update_clip transform."""

    def fn(e: Edit):
        c = e.editable(e.clip(lyrics_clip_id))
        if c["type"] != "lyrics":
            raise EditError("That isn't a lyrics clip")
        cur = c["lyrics"].get("style") or {}
        c["lyrics"]["style"] = presets.lyrics_style(preset, style, keep=cur) if preset else {**cur, **(style or {})}
        return c

    c, _ = await _edit(project_id, fn)
    return _json(c["lyrics"]["style"])


@tool()
async def edit_lyric_line(lyrics_clip_id: str, line: int, text: Optional[str] = None, shift: float = 0.0,
                          delete: bool = False, project_id: Optional[str] = None) -> str:
    """Fix one line of lyrics (line = its number, from 1, as get_lyrics lists):
    new text (its words are re-timed to the song), shift it earlier / later
    (seconds), or delete it."""
    e = await _load(project_id)
    c = e.clip(lyrics_clip_id)
    lines = c["lyrics"]["lines"]
    if not 1 <= line <= len(lines):
        raise EditError(f"Line {line} doesn't exist (1–{len(lines)})")
    i = line - 1
    ln = lines[i]
    new = None
    if text is not None and not delete:
        a = max(lines[i - 1]["end"] - 0.1 if i else 0.0, ln["start"] - 0.6)
        b = min(lines[i + 1]["start"] + 0.1, ln["end"] + 1.5) if i + 1 < len(lines) else ln["end"] + 1.5
        new = await api().post(f"/api/projects/{quote(e.project['id'])}/lyrics/{quote(c['asset_id'])}/retime",
                               {"text": text, "start": max(0.0, a), "end": b})

    def fn(e: Edit):
        cl = e.clip(lyrics_clip_id)
        ls = cl["lyrics"]["lines"]
        if delete:
            ls.pop(i)
            return None
        cur = new or ls[i]
        if shift:
            cur = {**cur, "start": cur["start"] + shift, "end": cur["end"] + shift,
                   "words": [{**w, "start": w["start"] + shift, "end": w["end"] + shift} for w in cur["words"]]}
        ls[i] = cur
        return cur

    res, _ = await _edit(project_id, fn)
    return _json(res) if res else "Deleted"


@tool(read_only=True)
async def get_lyrics(lyrics_clip_id: str, project_id: Optional[str] = None) -> str:
    """A lyrics clip's lines (numbered) with their times, and its style."""
    e = await _load(project_id)
    c = e.clip(lyrics_clip_id)
    ly = c.get("lyrics") or {}
    return _json({"source": ly.get("source"), "style": ly.get("style"),
                  "lines": [{"n": i + 1, "start": round(ln["start"], 2), "end": round(ln["end"], 2), "text": ln["text"],
                             "timed_by_line_only": bool(ln["words"]) and all(not w.get("conf") for w in ln["words"])}
                            for i, ln in enumerate(ly.get("lines") or [])]})


@tool()
async def redo_lyrics(lyrics_clip_id: str, lrclib_id: Optional[int] = None, lyrics_text: Optional[str] = None,
                      isolate_vocals: Optional[bool] = None, project_id: Optional[str] = None,
                      ctx: Optional[Context] = None) -> str:
    """Get the lyrics again: another LRCLIB match, your own text, or with the
    vocals isolated (more accurate on loud mixes)."""
    e = await _load(project_id)
    c = e.clip(lyrics_clip_id)
    iso = bool(c["lyrics"].get("isolate")) if isolate_vocals is None else isolate_vocals
    return await _compute_lyrics(e, c, iso, lrclib_id, lyrics_text, True, ctx)


# =============================================================================== export / jobs / undo

@tool()
async def export_video(quality: Literal["high", "medium", "low"] = "medium", height: Optional[int] = None,
                       sequence_id: Optional[str] = None, wait: bool = True, download_to: Optional[str] = None,
                       project_id: Optional[str] = None, ctx: Optional[Context] = None) -> str:
    """Render the edit to an MP4 (the main sequence unless sequence_id). height
    scales it (e.g. 720); download_to = a local file or folder path to save it."""
    pid = _pid(project_id)
    body = {"quality": quality, "height": height, "sequence_id": sequence_id}
    r = await api().post(f"/api/projects/{quote(pid)}/exports", body)
    rec = r["export"]
    url = api().url(f"/api/projects/{quote(pid)}/exports/{rec['id']}/download")
    if not wait:
        return _json({"export_id": rec["id"], "job_id": r["job"]["id"], "download_url": url})
    await api().wait_job(r["job"]["id"], _reporter(ctx))
    out: dict[str, Any] = {"export_id": rec["id"], "download_url": url}
    if download_to:
        out["saved_to"] = await _download(pid, rec["id"], download_to)
    return _json(out)


async def _download(pid: str, export_id: str, dest: str) -> str:
    rows = await api().get(f"/api/projects/{quote(pid)}/exports")
    rec = next((x for x in rows if x["id"] == export_id), None)
    if rec is None or rec["status"] != "done":
        raise EditError("That export isn't finished")
    path = Path(os.path.expanduser(dest))
    if path.is_dir() or dest.endswith(("/", os.sep)):
        path.mkdir(parents=True, exist_ok=True)
        path = path / rec["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    async with api().http.stream("GET", f"/api/projects/{quote(pid)}/exports/{export_id}/download") as r:
        if r.status_code >= 400:
            raise EditError(f"Download failed: {r.status_code}")
        with open(path, "wb") as fh:
            async for chunk in r.aiter_bytes(1 << 20):
                fh.write(chunk)
    return str(path)


@tool(read_only=True)
async def list_exports(project_id: Optional[str] = None) -> str:
    """Finished and running exports of the project, with download links."""
    pid = _pid(project_id)
    rows = await api().get(f"/api/projects/{quote(pid)}/exports")
    return _json([{**{k: x.get(k) for k in ("id", "name", "status", "width", "height", "duration", "size", "error")},
                   "download_url": api().url(f"/api/projects/{quote(pid)}/exports/{x['id']}/download")}
                  for x in rows])


@tool()
async def download_export(export_id: str, download_to: str, project_id: Optional[str] = None) -> str:
    """Save a finished export to a local file or folder."""
    return await _download(_pid(project_id), export_id, download_to)


@tool(read_only=True)
async def get_job(job_id: str) -> str:
    """Progress of a background job (export, upload processing, lyrics, ...)."""
    return _json(await api().get(f"/api/jobs/{job_id}"))


@tool()
async def cancel_job(job_id: str) -> str:
    """Stop a background job."""
    await api().post(f"/api/jobs/{job_id}/cancel")
    return "Cancelled"


@tool()
async def undo(steps: int = 1, project_id: Optional[str] = None) -> str:
    """Undo the last timeline changes made through these tools (this session)."""
    pid = _pid(project_id)
    stack = _undo.get(pid) or []
    if not stack:
        raise EditError("Nothing to undo")
    snap = None
    for _ in range(max(1, min(steps, len(stack)))):
        snap = stack.pop()
    assert snap is not None
    await api().put(f"/api/projects/{quote(pid)}", snap)
    return f"Undone ({len(stack)} more step(s) can be undone)"


# =============================================================================== resources / prompts

@mcp.resource("yabbe://guide", name="editing-guide", title="How to edit with YABBE", mime_type="text/markdown")
def guide() -> str:
    """How YABBE works and how to use these tools well."""
    return INSTRUCTIONS + """
Tips:
* Look before you cut: look_at_media for shots, listen_to_media for beats
  and silences. Cut on beats for music edits (beats are in file seconds; a
  clip at start S with in_point I hits beat B at S + (B - I) / speed).
* Build the main story on one video track; put overlays (text, logos,
  picture-in-picture) on tracks above; music on its own audio track.
* After a batch of edits, preview_frames over the changed range.
* Vertical videos: create_project(preset="vertical"); fill the frame with
  update_clip transform.scale (a 16:9 clip needs about 3.16 to fill 9:16).
"""


@mcp.prompt(name="edit_video", title="Edit a video from my footage")
def edit_video_prompt(brief: str, files: str = "") -> str:
    """Start an AI edit: the brief (what to make) and optionally files to upload."""
    up = f"Upload these files first: {files}\n" if files else ""
    return (f"Edit a video in YABBE.\n{up}Brief: {brief}\n\n"
            "Steps: open or create a project with a fitting canvas, upload the media, look at every video and "
            "listen to the music, write a short edit plan (shots, order, timing, text, music, effects), build it, "
            "check it with preview_frames, fix what looks wrong, then export it and give the download link.")
