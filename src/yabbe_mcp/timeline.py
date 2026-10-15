"""Timeline edits on a YABBE project document (the JSON the YABBE API serves).

YABBE's server stores whatever timeline the editor sends; the editing rules
(splitting, retiming, linking, nesting, ...) live in the editor. This module
re-implements them the same way, on plain dicts, so the MCP server can edit a
project and save it back. Everything here is pure (no I/O), which keeps it easy
to test.

Conventions (same as YABBE):
* ``tracks[0]`` is the top layer; video tracks hold pictures, audio tracks sound.
* A clip plays source time ``in_point + (t - start) * speed`` at timeline time ``t``.
* Keyframe and marker times are seconds from the clip's start.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

MIN_CLIP = 1 / 30
EPS = 1e-6
DEFAULT_STILL = 5.0  # seconds for photos / text / shapes
MIN_SPEED, MAX_SPEED = 0.25, 4.0

VISUAL = {"video", "image", "text", "sequence", "shape", "lyrics"}


class EditError(ValueError):
    """An edit that can't be done (message is for the user / agent)."""


def uid(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:12]


def end(c: dict) -> float:
    return c["start"] + c["duration"]


def default_transform() -> dict:
    return {"x": 0.0, "y": 0.0, "scale": 1.0, "rotation": 0.0, "opacity": 1.0, "flip_h": False, "flip_v": False}


def new_clip(**fields: Any) -> dict:
    c = {
        "id": uid("c_"), "track_id": "", "type": "video", "asset_id": None, "start": 0.0, "duration": DEFAULT_STILL,
        "in_point": 0.0, "speed": 1.0, "volume": 1.0, "muted": False, "fade_in": 0.0, "fade_out": 0.0,
        "transform": default_transform(), "crop": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        "text": None, "keyframes": {}, "markers": [], "transition": None, "blend": "normal",
    }
    c.update(fields)
    return c


def new_track(kind: str, name: str) -> dict:
    return {"id": uid("t_"), "kind": kind, "name": name, "muted": False, "hidden": False, "locked": False}


@dataclass
class Edit:
    """A project being edited: one sequence of it is the one being worked on."""

    project: dict
    sequence_id: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.project = copy.deepcopy(self.project)
        if not self.sequence_id or not self._seq(self.sequence_id):
            self.sequence_id = self.project.get("main_sequence_id") or self.project["sequences"][0]["id"]

    # -- access ---------------------------------------------------------------------------------

    def _seq(self, sid: str) -> Optional[dict]:
        return next((s for s in self.project["sequences"] if s["id"] == sid), None)

    @property
    def seq(self) -> dict:
        s = self._seq(self.sequence_id)  # type: ignore[arg-type]
        assert s is not None
        return s

    @property
    def clips(self) -> list[dict]:
        return self.seq["clips"]

    @property
    def tracks(self) -> list[dict]:
        return self.seq["tracks"]

    @property
    def settings(self) -> dict:
        return self.seq["settings"]

    def asset(self, asset_id: Optional[str]) -> Optional[dict]:
        return next((a for a in self.project.get("assets", []) if a["id"] == asset_id), None)

    def clip(self, clip_id: str) -> dict:
        c = next((c for c in self.clips if c["id"] == clip_id), None)
        if c is None:
            raise EditError(f"No clip {clip_id!r} in sequence “{self.seq['name']}” (see get_timeline)")
        return c

    def track(self, ref: str) -> dict:
        """A track by id, by name (case-insensitive) or by number as listed (1 = top)."""
        for t in self.tracks:
            if t["id"] == ref:
                return t
        low = ref.strip().lower()
        for t in self.tracks:
            if t["name"].lower() == low:
                return t
        if low.isdigit() and 1 <= int(low) <= len(self.tracks):
            return self.tracks[int(low) - 1]
        raise EditError(f"No track {ref!r}; tracks are: " + ", ".join(f"{t['name']} ({t['id']})" for t in self.tracks))

    def locked(self, c: dict) -> bool:
        t = next((t for t in self.tracks if t["id"] == c["track_id"]), None)
        return bool(t and t.get("locked"))

    def editable(self, c: dict) -> dict:
        if self.locked(c):
            raise EditError(f"Clip {c['id']} is on a locked track")
        return c

    def duration(self) -> float:
        return max((end(c) for c in self.clips), default=0.0)

    def source_length(self, c: dict) -> float:
        """How long the clip's source is (inf for stills / text / shapes)."""
        if c["type"] in ("text", "shape", "image") or c.get("hold"):
            return float("inf")
        if c["type"] == "sequence":
            s = self._seq(c.get("sequence_id") or "")
            return max((end(x) for x in s["clips"]), default=0.0) if s else float("inf")
        a = self.asset(c.get("asset_id"))
        return float(a["duration"]) if a and a.get("duration") else float("inf")

    def overlaps(self, track_id: str, start: float, duration: float, ignore: Iterable[str] = ()) -> bool:
        skip = set(ignore)
        e = start + duration
        return any(
            c["track_id"] == track_id and c["id"] not in skip and c["start"] < e - EPS and end(c) > start + EPS
            for c in self.clips
        )

    def track_end(self, track_id: str) -> float:
        return max((end(c) for c in self.clips if c["track_id"] == track_id), default=0.0)

    def linked(self, ids: Iterable[str]) -> list[str]:
        ids = list(ids)
        links = {c.get("link") for c in self.clips if c["id"] in ids and c.get("link")}
        out = list(ids)
        for c in self.clips:
            if c.get("link") in links and c["id"] not in out:
                out.append(c["id"])
        return out

    # -- tracks -------------------------------------------------------------------------------

    def add_track(self, kind: str, name: Optional[str] = None, position: str = "auto") -> dict:
        n = sum(1 for t in self.tracks if t["kind"] == kind) + 1
        t = new_track(kind, name or f"{'Video' if kind == 'video' else 'Audio'} {n}")
        if position == "top" or (position == "auto" and kind == "video"):
            self.tracks.insert(0, t)
        else:
            self.tracks.append(t)
        return t

    def pick_track(self, kind: str, start: float, duration: float, ignore: Iterable[str] = ()) -> dict:
        """The lowest unlocked track of ``kind`` that's free over [start, start+duration]
        (a new one if none is)."""
        for t in reversed(self.tracks) if kind == "video" else self.tracks:
            if t["kind"] == kind and not t.get("locked") and not self.overlaps(t["id"], start, duration, ignore):
                return t
        return self.add_track(kind)

    # -- adding ---------------------------------------------------------------------------------

    def place(self, c: dict, track: Optional[str], start: Optional[float], kind: str) -> dict:
        """Put a new clip on ``track`` (or a free one) at ``start`` (or the end of that track)."""
        if track:
            t = self.track(track)
            if t["kind"] != kind:
                raise EditError(f"Track “{t['name']}” is an {t['kind']} track; this clip needs a {kind} track")
            if t.get("locked"):
                raise EditError(f"Track “{t['name']}” is locked")
            c["track_id"] = t["id"]
            c["start"] = self.track_end(t["id"]) if start is None else max(0.0, start)
            if self.overlaps(t["id"], c["start"], c["duration"]):
                raise EditError(f"Something is already on “{t['name']}” between {c['start']:.2f}s and {end(c):.2f}s "
                                "(pick another time or track, or leave the track out to use a free one)")
        else:
            if start is None:
                # append after everything of this kind
                start = max((end(x) for x in self.clips if self._kind_of(x) == kind), default=0.0)
            c["start"] = max(0.0, start)
            c["track_id"] = self.pick_track(kind, c["start"], c["duration"])["id"]
        self.clips.append(c)
        return c

    def _kind_of(self, c: dict) -> str:
        t = next((t for t in self.tracks if t["id"] == c["track_id"]), None)
        return t["kind"] if t else "video"

    def add_media(self, asset_id: str, start: Optional[float] = None, track: Optional[str] = None,
                  in_point: float = 0.0, duration: Optional[float] = None) -> dict:
        a = self.asset(asset_id)
        if a is None:
            raise EditError(f"No media {asset_id!r} in this project (see list_media)")
        if a.get("status") == "error":
            raise EditError(f"“{a['original_name']}” failed to process: {a.get('error')}")
        kind = a["kind"]
        if kind == "image":
            length = duration or DEFAULT_STILL
            in_point = 0.0
        else:
            avail = max(0.0, float(a.get("duration") or 0) - in_point)
            if avail <= MIN_CLIP:
                raise EditError(f"in_point {in_point}s is past the end of “{a['original_name']}” ({a.get('duration', 0):.2f}s)")
            length = min(duration, avail) if duration else avail
        c = new_clip(type=kind, asset_id=asset_id, in_point=in_point,
                     duration=max(MIN_CLIP, length))
        return self.place(c, track, start, "audio" if kind == "audio" else "video")

    def add_text(self, style: dict, start: Optional[float], duration: float, track: Optional[str]) -> dict:
        base = {"content": "Text", "font": "DejaVu Sans", "size": 96, "color": "#ffffff", "background": None,
                "padding": 16, "stroke_color": "#000000", "stroke_width": 0, "align": "center", "bold": False,
                "italic": False, "line_spacing": 1.2}
        c = new_clip(type="text", text={**base, **style}, duration=max(MIN_CLIP, duration))
        return self.place(c, track, start if start is not None else 0.0, "video")

    def add_shape(self, style: dict, start: Optional[float], duration: float, track: Optional[str]) -> dict:
        base = {"kind": "rectangle", "width": 400, "height": 400, "fill": "#7c5cff", "stroke": None, "stroke_width": 0,
                "radius": 0, "sides": 6, "inner": 0.45}
        c = new_clip(type="shape", shape={**base, **style}, duration=max(MIN_CLIP, duration))
        return self.place(c, track, start if start is not None else 0.0, "video")

    def add_sequence_clip(self, sequence_id: str, start: Optional[float], track: Optional[str]) -> dict:
        child = self._seq(sequence_id)
        if child is None:
            raise EditError(f"No sequence {sequence_id!r}")
        if child["id"] == self.sequence_id or self._contains(child["id"], self.sequence_id):  # type: ignore[arg-type]
            raise EditError("A sequence can't be put inside itself")
        length = max((end(x) for x in child["clips"]), default=0.0)
        if length <= 0:
            raise EditError(f"Sequence “{child['name']}” is empty")
        c = new_clip(type="sequence", sequence_id=sequence_id, duration=length)
        return self.place(c, track, start, "video")

    def _contains(self, outer: str, inner: str) -> bool:
        seen, todo = set(), [outer]
        while todo:
            s = self._seq(todo.pop())
            for c in (s or {}).get("clips", []):
                sid = c.get("sequence_id") if c["type"] == "sequence" else None
                if sid == inner:
                    return True
                if sid and sid not in seen:
                    seen.add(sid)
                    todo.append(sid)
        return False

    # -- changing ---------------------------------------------------------------------------------

    def move(self, clip_id: str, start: Optional[float] = None, track: Optional[str] = None,
             with_linked: bool = True) -> list[str]:
        c = self.editable(self.clip(clip_id))
        dt = 0.0 if start is None else max(0.0, start) - c["start"]
        group = self.linked([clip_id]) if with_linked else [clip_id]
        group_clips = [self.clip(i) for i in group]
        if any(self.locked(x) for x in group_clips):
            raise EditError("A linked clip is on a locked track (move with with_linked=false)")
        # don't move anything before 0
        dt = max(dt, -min(x["start"] for x in group_clips))
        target = self.track(track) if track else None
        if target and target["kind"] != self._kind_of(c):
            raise EditError(f"“{target['name']}” is an {target['kind']} track")
        ids = {x["id"] for x in group_clips}
        for x in group_clips:
            tid = target["id"] if (target and x["id"] == clip_id) else x["track_id"]
            if self.overlaps(tid, x["start"] + dt, x["duration"], ids):
                name = next(t["name"] for t in self.tracks if t["id"] == tid)
                raise EditError(f"That would overlap another clip on “{name}” (at {x['start'] + dt:.2f}s)")
        for x in group_clips:
            x["start"] = round(x["start"] + dt, 6)
            if target and x["id"] == clip_id:
                x["track_id"] = target["id"]
        return list(ids)

    def trim(self, clip_id: str, new_start: Optional[float] = None, new_end: Optional[float] = None,
             with_linked: bool = True) -> list[str]:
        """Move a clip's start and/or end edge on the timeline (the footage stays put:
        trimming the start moves the in point too)."""
        group = self.linked([clip_id]) if with_linked else [clip_id]
        c0 = self.clip(clip_id)
        old_start, old_end = c0["start"], end(c0)
        for cid in group:
            c = self.editable(self.clip(cid))
            # linked clips trim by the same amounts where they share the edge
            s = c["start"] + ((new_start - old_start) if new_start is not None else 0.0)
            e = end(c) + ((new_end - old_end) if new_end is not None else 0.0)
            if cid != clip_id and abs(c["start"] - old_start) > 1e-3:
                s = c["start"]
            if cid != clip_id and abs(end(c) - old_end) > 1e-3:
                e = end(c)
            self._trim_one(c, s, e)
        return group

    def _trim_one(self, c: dict, s: float, e: float) -> None:
        s = max(0.0, s)
        ds = s - c["start"]
        in_point = c["in_point"] + (0.0 if c.get("hold") else ds * c["speed"])
        if in_point < -EPS:
            s -= in_point / c["speed"]
            ds = s - c["start"]
            in_point = 0.0
        src = self.source_length(c)
        max_len = (src - in_point) / c["speed"]
        e = min(e, s + max_len)
        if e - s < MIN_CLIP:
            raise EditError("That would make the clip too short (or there's no footage there)")
        others = [x for x in self.clips if x["track_id"] == c["track_id"] and x["id"] != c["id"]]
        if any(x["start"] < e - EPS and end(x) > s + EPS for x in others):
            raise EditError("That would overlap the neighbouring clip on the same track")
        c["keyframes"] = shift_keyframes(c.get("keyframes") or {}, -ds)
        c["markers"] = [{**m, "t": m["t"] - ds} for m in c.get("markers") or []]
        c["start"], c["duration"], c["in_point"] = round(s, 6), round(e - s, 6), max(0.0, round(in_point, 6))
        c["fade_in"] = min(c.get("fade_in", 0), c["duration"])
        c["fade_out"] = min(c.get("fade_out", 0), c["duration"])

    def split(self, clip_id: str, at: float, with_linked: bool = True) -> list[tuple[str, str]]:
        """Cut clips at timeline time ``at``. Returns (left id, right id) pairs."""
        group = self.linked([clip_id]) if with_linked else [clip_id]
        targets = [self.clip(i) for i in group]
        targets = [c for c in targets if c["start"] + MIN_CLIP < at < end(c) - MIN_CLIP and not self.locked(c)]
        if not targets:
            raise EditError(f"{at:.2f}s isn't inside the clip (it runs {self.clip(clip_id)['start']:.2f}–"
                            f"{end(self.clip(clip_id)):.2f}s)")
        relink: dict[str, str] = {}
        pairs = []
        for c in targets:
            left = at - c["start"]
            b = copy.deepcopy(c)
            b.update(
                id=uid("c_"), start=round(at, 6), duration=round(c["duration"] - left, 6),
                in_point=c["in_point"] if c.get("hold") else round(c["in_point"] + left * c["speed"], 6),
                fade_in=0.0, keyframes=shift_keyframes(c.get("keyframes") or {}, -left),
                markers=[{**m, "t": m["t"] - left} for m in c.get("markers") or [] if m["t"] >= left - 1e-6],
                link=(relink.setdefault(c["link"], uid("l_")) if c.get("link") else None),
            )
            c.update(duration=round(left, 6), fade_out=0.0, transition=None,
                     markers=[m for m in c.get("markers") or [] if m["t"] < left - 1e-6])
            self.clips.append(b)
            pairs.append((c["id"], b["id"]))
        return pairs

    def delete(self, ids: Iterable[str], ripple: bool = False, with_linked: bool = True) -> list[str]:
        ids = list(ids)
        for i in ids:
            self.clip(i)
        group = self.linked(ids) if with_linked else ids
        gone = [self.clip(i) for i in group if not self.locked(self.clip(i))]
        gone_ids = {c["id"] for c in gone}
        if ripple:
            # close each gap on its own track
            for c in sorted(gone, key=lambda x: -x["start"]):
                for x in self.clips:
                    if x["track_id"] == c["track_id"] and x["id"] not in gone_ids and x["start"] >= end(c) - EPS:
                        x["start"] = round(x["start"] - c["duration"], 6)
        self.seq["clips"] = [c for c in self.clips if c["id"] not in gone_ids]
        # followers / pins pointing at removed clips
        for c in self.clips:
            if (c.get("follow") or {}).get("clip_id") in gone_ids:
                c["follow"] = None
            if (c.get("pin") or {}).get("clip_id") in gone_ids:
                c["pin"] = None
        return list(gone_ids)

    def duplicate(self, clip_id: str, start: Optional[float] = None, track: Optional[str] = None) -> dict:
        c = self.clip(clip_id)
        d = copy.deepcopy(c)
        d.update(id=uid("c_"), link=None, transition=None)
        if start is None:
            start = end(c)
        if track is None and not self.overlaps(c["track_id"], start, d["duration"]):
            track = c["track_id"]
        return self.place(d, track, start, self._kind_of(c))

    def set_speed(self, clip_id: str, speed: float) -> dict:
        c = self.editable(self.clip(clip_id))
        if c["type"] not in ("video", "audio", "sequence", "lyrics") or c.get("hold"):
            raise EditError("Only video, audio and nested-sequence clips have a speed")
        speed = max(MIN_SPEED, min(MAX_SPEED, speed))
        footage = c["duration"] * c["speed"]
        duration = max(MIN_CLIP, footage / speed)
        if self.overlaps(c["track_id"], c["start"], duration, [c["id"]]):
            raise EditError("At that speed the clip would run into the next clip on its track")
        k = duration / c["duration"]
        c["keyframes"] = {p: [{**f, "t": f["t"] * k,
                               "hi": [f["hi"][0] * k, f["hi"][1]] if f.get("hi") else f.get("hi"),
                               "ho": [f["ho"][0] * k, f["ho"][1]] if f.get("ho") else f.get("ho")}
                              for f in frames] for p, frames in (c.get("keyframes") or {}).items()}
        c["markers"] = [{**m, "t": m["t"] * k} for m in c.get("markers") or []]
        c.update(speed=speed, duration=round(duration, 6))
        c["fade_in"] = min(c.get("fade_in", 0), duration)
        c["fade_out"] = min(c.get("fade_out", 0), duration)
        if abs(speed - 1) > 1e-6:
            self._add_effect(c, "speed")
        return c

    def patch(self, clip_id: str, changes: dict) -> dict:
        """Deep-merge ``changes`` into a clip (nested dicts merge; lists / values replace)."""
        c = self.editable(self.clip(clip_id))
        protected = {"id", "track_id", "start", "duration", "in_point", "speed", "type", "asset_id", "sequence_id"}
        bad = protected & set(changes)
        if bad:
            raise EditError(f"Use move_clip / trim_clip / set_speed to change {', '.join(sorted(bad))}")
        _merge(c, changes)
        return c

    def _add_effect(self, c: dict, effect: str) -> None:
        if c.get("effects") is None:
            c["effects"] = infer_effects(c)
        if effect not in c["effects"]:
            c["effects"].append(effect)

    def remove_effect(self, c: dict, effect: str) -> None:
        if c.get("effects") is None:
            c["effects"] = infer_effects(c)
        c["effects"] = [e for e in c["effects"] if e != effect]

    def add_effect(self, c: dict, effect: str) -> None:
        self._add_effect(c, effect)

    # -- links / audio / transitions -----------------------------------------------------------

    def link(self, ids: list[str]) -> str:
        if len(ids) < 2:
            raise EditError("Link at least two clips")
        link = uid("l_")
        for i in self.linked(ids):
            self.clip(i)["link"] = link
        return link

    def unlink(self, ids: list[str]) -> None:
        for i in ids:
            self.clip(i)["link"] = None

    def separate_audio(self, clip_id: str) -> dict:
        c = self.editable(self.clip(clip_id))
        a = self.asset(c.get("asset_id"))
        if c["type"] != "video" or not a or not a.get("has_audio") or c.get("audio_detached"):
            raise EditError("Only a video clip with sound (not already separated) can have its audio separated")
        track = self.pick_track("audio", c["start"], c["duration"])
        link = c.get("link") or uid("l_")
        kf = dict(c.get("keyframes") or {})
        vol = kf.pop("volume", None)
        audio = new_clip(type="audio", asset_id=c["asset_id"], track_id=track["id"], start=c["start"],
                         duration=c["duration"], in_point=c["in_point"], speed=c["speed"], volume=c.get("volume", 1),
                         muted=c.get("muted", False), fade_in=c.get("fade_in", 0), fade_out=c.get("fade_out", 0),
                         keyframes={"volume": vol} if vol else {}, link=link)
        c.update(link=link, audio_detached=True, volume=1.0, muted=False, fade_in=0.0, fade_out=0.0, keyframes=kf)
        self.clips.append(audio)
        return audio

    def next_touching(self, c: dict) -> Optional[dict]:
        return next((x for x in self.clips if x["track_id"] == c["track_id"] and x["id"] != c["id"]
                     and abs(x["start"] - end(c)) < 1.5 / max(1.0, float(self.settings.get("fps", 30)))), None)

    def set_transition(self, clip_id: str, kind: Optional[str], duration: float = 0.5, audio: bool = True) -> Optional[dict]:
        c = self.editable(self.clip(clip_id))
        if kind is None:
            c["transition"] = None
            return None
        nxt = self.next_touching(c)
        if nxt is None:
            raise EditError("A transition goes from a clip into the next clip that touches it on the same track: "
                            "nothing starts right where this clip ends")
        if self._kind_of(c) != "video":
            raise EditError("Transitions are for clips on video tracks (fade audio with fade_in / fade_out)")
        longest = min(c["duration"], nxt["duration"])
        duration = max(0.04, min(duration, 5.0, longest))
        c["transition"] = {"kind": kind, "duration": round(duration, 3), "audio": audio}
        return c["transition"]

    # -- freeze frame ------------------------------------------------------------------------------

    def freeze(self, clip_id: str, at: float, image_asset_id: str, duration: float) -> dict:
        """Split the clip at ``at`` and put the still (an image asset) between the halves,
        pushing the rest of that track (and linked clips) later by ``duration``."""
        c = self.editable(self.clip(clip_id))
        fps = float(self.settings.get("fps", 30))
        at = round(at * fps) / fps
        group = [self.clip(i) for i in self.linked([clip_id])]
        group = [x for x in group if x["start"] <= at + EPS and end(x) > at + EPS and not self.locked(x)]
        tracks = {x["track_id"] for x in group}
        ids = {x["id"] for x in group}
        for x in self.clips:  # ripple: make room
            if x["track_id"] in tracks and x["id"] not in ids and x["start"] >= at - EPS:
                x["start"] = round(x["start"] + duration, 6)
        still = None
        for x in group:
            left = at - x["start"]
            right = copy.deepcopy(x)
            right.update(id=uid("c_") if left >= MIN_CLIP else x["id"], start=round(at + duration, 6),
                         duration=round(x["duration"] - left, 6), in_point=round(x["in_point"] + left * x["speed"], 6),
                         fade_in=0.0, keyframes=shift_keyframes(x.get("keyframes") or {}, -left),
                         markers=[{**m, "t": m["t"] - left} for m in x.get("markers") or []])
            if x["id"] == clip_id:
                still = new_clip(type="image", asset_id=image_asset_id, track_id=x["track_id"], start=at,
                                 duration=duration, transform=copy.deepcopy(x["transform"]),
                                 crop=copy.deepcopy(x.get("crop")), blend=x.get("blend", "normal"),
                                 chroma_key=copy.deepcopy(x.get("chroma_key")),
                                 effects=[e for e in (x.get("effects") or []) if e in ("crop", "chroma_key")] or None)
            if left >= MIN_CLIP:
                x.update(duration=round(left, 6), fade_out=0.0, transition=None)
                self.clips.append(right)
            else:
                x.update(right)
        assert still is not None
        self.clips.append(still)
        return still

    # -- nesting / sequences -------------------------------------------------------------------------

    def nest(self, ids: list[str], name: Optional[str]) -> tuple[dict, dict]:
        """Pre-compose clips into a new sequence and put it in their place."""
        picked = [self.clip(i) for i in ids]
        picked = [c for c in picked if not self.locked(c)]
        if not picked:
            raise EditError("Nothing to nest")
        t0 = min(c["start"] for c in picked)
        t1 = max(end(c) for c in picked)
        picked_ids = {c["id"] for c in picked}
        used = [t for t in self.tracks if any(c["track_id"] == t["id"] for c in picked)]
        tmap = {t["id"]: uid("t_") for t in used}
        tracks = [{**t, "id": tmap[t["id"]], "locked": False} for t in used]
        if not any(t["kind"] == "video" for t in tracks):
            tracks.insert(0, new_track("video", "Video 1"))
        if not any(t["kind"] == "audio" for t in tracks):
            tracks.append(new_track("audio", "Audio 1"))
        relink: dict[str, str] = {}
        moved = []
        for c in picked:
            nxt = self.next_touching(c)
            m = copy.deepcopy(c)
            m.update(id=uid("c_"), start=round(c["start"] - t0, 6), track_id=tmap[c["track_id"]],
                     link=relink.setdefault(c["link"], uid("l_")) if c.get("link") else None,
                     transition=c.get("transition") if nxt and nxt["id"] in picked_ids else None)
            moved.append(m)
        names = {s["name"] for s in self.project["sequences"]}
        n = 1
        while f"Nested sequence {n}" in names:
            n += 1
        seq = {"id": uid("s_"), "name": (name or "").strip() or f"Nested sequence {n}",
               "settings": dict(self.settings), "tracks": tracks, "clips": moved}
        self.seq["clips"] = [c for c in self.clips if c["id"] not in picked_ids]
        vis = [t for t in used if t["kind"] == "video"]
        track = vis[-1] if vis else next((t for t in reversed(self.tracks) if t["kind"] == "video" and not t.get("locked")), None)
        if track is None or self.overlaps(track["id"], t0, t1 - t0):
            at = self.tracks.index(track) if track else 0
            track = new_track("video", f"Video {sum(1 for t in self.tracks if t['kind'] == 'video') + 1}")
            self.tracks.insert(at, track)
        clip = new_clip(type="sequence", sequence_id=seq["id"], track_id=track["id"], start=t0, duration=t1 - t0)
        self.project["sequences"].append(seq)
        self.clips.append(clip)
        return seq, clip

    def new_sequence(self, name: str, settings: Optional[dict]) -> dict:
        seq = {"id": uid("s_"), "name": name or "Sequence", "settings": {**self.settings, **(settings or {})},
               "tracks": [new_track("video", "Video 2"), new_track("video", "Video 1"), new_track("audio", "Audio 1")],
               "clips": []}
        self.project["sequences"].append(seq)
        return seq

    def delete_sequence(self, sequence_id: str) -> None:
        if len(self.project["sequences"]) <= 1:
            raise EditError("A project needs at least one sequence")
        if not self._seq(sequence_id):
            raise EditError(f"No sequence {sequence_id!r}")
        users = [s["name"] for s in self.project["sequences"]
                 if any(c.get("sequence_id") == sequence_id for c in s["clips"])]
        if users:
            raise EditError(f"It's used inside {', '.join(users)}: remove those nested clips first")
        self.project["sequences"] = [s for s in self.project["sequences"] if s["id"] != sequence_id]
        if self.project.get("main_sequence_id") == sequence_id:
            self.project["main_sequence_id"] = self.project["sequences"][0]["id"]
        if self.sequence_id == sequence_id:
            self.sequence_id = self.project["main_sequence_id"]

    # -- describing -------------------------------------------------------------------------------------

    def clip_name(self, c: dict) -> str:
        t = c["type"]
        if t == "text":
            return "Text: " + ((c.get("text") or {}).get("content") or "").split("\n")[0][:60]
        if t == "shape":
            return "Shape: " + (c.get("shape") or {}).get("kind", "shape")
        if t == "sequence":
            s = self._seq(c.get("sequence_id") or "")
            return "Nested: " + (s["name"] if s else "missing sequence")
        a = self.asset(c.get("asset_id"))
        name = a["original_name"] if a else "missing media"
        if t == "lyrics":
            return f"Lyrics of {name}"
        return ("Freeze of " if c.get("hold") else "") + name

    def describe(self) -> dict:
        """A compact, readable view of the sequence for an AI to reason about."""
        tracks = []
        for i, t in enumerate(self.tracks, 1):
            clips = sorted((c for c in self.clips if c["track_id"] == t["id"]), key=lambda c: c["start"])
            tracks.append({
                "n": i, "id": t["id"], "name": t["name"], "kind": t["kind"],
                **{k: True for k in ("muted", "hidden", "locked") if t.get(k)},
                "clips": [self.describe_clip(c) for c in clips],
            })
        return {
            "sequence": {"id": self.seq["id"], "name": self.seq["name"],
                         "main": self.seq["id"] == self.project.get("main_sequence_id"), **self.settings},
            "duration": round(self.duration(), 3),
            "tracks_top_to_bottom": tracks,
        }

    def describe_clip(self, c: dict) -> dict:
        d: dict[str, Any] = {"id": c["id"], "type": c["type"], "name": self.clip_name(c),
                             "start": round(c["start"], 3), "end": round(end(c), 3)}
        if c["type"] in ("video", "audio", "sequence", "lyrics"):
            d["source"] = [round(c["in_point"], 3), round(c["in_point"] + c["duration"] * c["speed"], 3)]
        if c.get("asset_id"):
            d["asset_id"] = c["asset_id"]
        if abs(c.get("speed", 1) - 1) > 1e-6:
            d["speed"] = c["speed"]
        tr = c.get("transform") or {}
        changed = {k: v for k, v in tr.items() if v != default_transform().get(k)}
        if changed:
            d["transform"] = changed
        for k in ("volume",):
            if c.get(k, 1) != 1:
                d[k] = c[k]
        for k in ("muted", "audio_detached", "hold"):
            if c.get(k):
                d[k] = True
        for k in ("fade_in", "fade_out"):
            if c.get(k):
                d[k] = c[k]
        if c.get("blend", "normal") != "normal":
            d["blend"] = c["blend"]
        if c.get("transition"):
            d["transition_out"] = c["transition"]
        if c.get("link"):
            d["link"] = c["link"]
        if c.get("keyframes"):
            d["animated"] = sorted(p for p, f in c["keyframes"].items() if f)
        fx = c.get("effects") if c.get("effects") is not None else infer_effects(c)
        if fx:
            d["effects"] = fx
        if c.get("markers"):
            d["markers"] = [{"t": m["t"], "label": m.get("label", "")} for m in c["markers"]]
        if c["type"] == "text":
            t = c.get("text") or {}
            d["text"] = {k: t.get(k) for k in ("content", "font", "size", "color") if k in t}
        if c["type"] == "lyrics":
            ly = c.get("lyrics") or {}
            d["lyrics"] = {"lines": len(ly.get("lines") or []), "preset": (ly.get("style") or {}).get("preset"),
                           "source": ly.get("source")}
        if c.get("follow"):
            d["follows"] = c["follow"]
        if c.get("pin"):
            d["pinned_to"] = c["pin"]
        return d


def shift_keyframes(kf: dict, dt: float) -> dict:
    if not dt:
        return kf
    return {p: [{**f, "t": f["t"] + dt} for f in frames] for p, frames in kf.items()}


def infer_effects(c: dict) -> list[str]:
    """The editor's Effects list for a clip that doesn't record one."""
    out = []
    cr = c.get("crop") or {}
    if any(cr.get(k) for k in ("left", "top", "right", "bottom")):
        out.append("crop")
    if abs(c.get("speed", 1) - 1) > 1e-6:
        out.append("speed")
    if c.get("chroma_key"):
        out.append("chroma_key")
    if c.get("roto"):
        out.append("roto")
    trackers = c.get("trackers") or []
    if any(t.get("kind") != "stabilize" for t in trackers):
        out.append("tracking")
    if c.get("stabilize") or any(t.get("kind") == "stabilize" for t in trackers):
        out.append("stabilize")
    return out


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
