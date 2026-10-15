"""End to end: a real YABBE server (from YABBE_BACKEND, default ../YABBE/backend)
on a temporary data folder, and this MCP server driven by a real MCP client over
stdio - an edit from upload to exported MP4."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = Path(os.environ.get("YABBE_BACKEND", ROOT.parent / "YABBE" / "backend"))
PY = BACKEND / ".venv" / "bin" / "python"

pytestmark = pytest.mark.skipif(not PY.is_file() or not shutil.which("ffmpeg"),
                                reason="needs a YABBE checkout with its venv (YABBE_BACKEND) and ffmpeg")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def yabbe(tmp_path_factory):
    data = tmp_path_factory.mktemp("yabbe-data")
    port = _free_port()
    env = {**os.environ, "YABBE_DATA_DIR": str(data), "YABBE_STATIC_DIR": str(data / "nostatic"),
           "YABBE_AUTO_PRERENDER": "0"}
    proc = subprocess.Popen([str(PY), "-m", "uvicorn", "app.main:app", "--port", str(port)], cwd=BACKEND, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.25)
    else:
        proc.kill()
        pytest.fail("YABBE didn't start")
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run([*ff, "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=4", "-f", "lavfi", "-i", "sine=f=440:d=4",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "shot1.mp4")], check=True)
    subprocess.run([*ff, "-f", "lavfi", "-i", "mandelbrot=s=640x360:r=25", "-t", "4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(d / "shot2.mp4")], check=True)
    # a click every 0.5 s = 120 bpm
    subprocess.run([*ff, "-f", "lavfi", "-i", "sine=f=1000:d=12",
                    "-af", "volume='if(lt(mod(t,0.5),0.04),1,0)':eval=frame", str(d / "beat.wav")], check=True)
    subprocess.run([*ff, "-f", "lavfi", "-i", "color=c=red:s=200x200:d=1", "-frames:v", "1", str(d / "logo.png")],
                   check=True)
    return d


def test_full_edit(yabbe, media, tmp_path):
    import anyio
    from mcp import Client, StdioServerParameters

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "yabbe_mcp"],
                                       env={**os.environ, "YABBE_URL": yabbe})
        async with Client(params, read_timeout_seconds=600) as client:
            async def call(tool, **args):
                r = await client.call_tool(tool, args)
                texts = [c.text for c in r.content if c.type == "text"]
                images = [c for c in r.content if c.type == "image"]
                if r.is_error:
                    raise AssertionError(f"{tool}: {texts}")
                return (json.loads(texts[0]) if texts and texts[0][:1] in "[{" else (texts[0] if texts else None)), images

            async def fails(tool, **args):
                r = await client.call_tool(tool, args)
                assert r.is_error, tool
                return r.content[0].text

            names = {t.name for t in (await client.list_tools()).tools}
            assert {"upload_media", "add_clip", "export_video", "add_lyrics", "roto_cut_out"} <= names

            status, _ = await call("yabbe_status")
            assert status["health"]
            proj, _ = await call("create_project", name="MCP test", preset="720p")
            assert proj["sequences"][0]["width"] == 1280

            up, _ = await call("upload_media", paths=[str(media / f) for f in ("shot1.mp4", "shot2.mp4", "beat.wav", "logo.png")])
            ids = {m["original_name"]: m["id"] for m in up}
            assert all(m["status"] == "ready" for m in up)

            text, imgs = await call("look_at_media", asset_id=ids["shot1.mp4"], count=4)
            assert imgs and "4.00s" in text
            hear, _ = await call("listen_to_media", asset_id=ids["beat.wav"])
            assert abs(hear["tempo_bpm"] - 120) < 3 and len(hear["onsets"]) > 15

            a, _ = await call("add_clip", asset_id=ids["shot1.mp4"])
            b, _ = await call("add_clip", asset_id=ids["shot2.mp4"])
            assert b["start"] == pytest.approx(4)
            tr, _ = await call("set_transition", clip_id=a["id"], kind="wipe_left", duration=0.5)
            assert tr["kind"] == "wipe_left"
            music, _ = await call("add_clip", asset_id=ids["beat.wav"], duration=8)
            await call("update_clip", clip_id=music["id"], changes={"volume": 0.5, "fade_out": 1})
            title, _ = await call("add_text", content="Hello MCP", start=0.5, duration=3, size=72, color="#ffcc00")
            await call("set_keyframes", clip_id=title["id"], property="opacity",
                       keyframes=[{"t": 0, "v": 0}, {"t": 0.5, "v": 1, "ease": "ease_out"}])
            logo, _ = await call("add_clip", asset_id=ids["logo.png"], start=1, duration=2)
            await call("update_clip", clip_id=logo["id"], changes={"transform": {"x": 500, "y": -250, "scale": 0.3}})
            shape, _ = await call("add_shape", kind="star", width=120, height=120, fill="#00ccff", start=5, duration=2)

            pairs, _ = await call("split_clip", clip_id=b["id"], at=6)
            await call("set_speed", clip_id=pairs[0]["right"], speed=2)
            await call("separate_audio", clip_id=a["id"])
            still, _ = await call("freeze_frame", clip_id=a["id"], at=2, duration=1)
            assert still["type"] == "image"

            tl, _ = await call("get_timeline")
            kinds = [c["type"] for t in tl["tracks_top_to_bottom"] for c in t["clips"]]
            assert {"video", "image", "text", "shape", "audio"} <= set(kinds)

            # undo the last change (the freeze) and do it again
            await call("undo")
            tl2, _ = await call("get_timeline")
            assert not any(c.get("type") == "image" and c["start"] == 2 for t in tl2["tracks_top_to_bottom"] for c in t["clips"])

            text, frame = await call("preview_frame", t=1.0, height=360)
            assert frame and frame[0].mime_type == "image/jpeg"
            _, sheet = await call("preview_frames", count=4)
            assert sheet

            err = await fails("add_clip", asset_id="nope")
            assert "No media" in err
            err = await fails("delete_project", project_id=proj["project"]["id"])
            assert "confirm" in err

            out, _ = await call("export_video", quality="low", height=360, download_to=str(tmp_path) + "/")
            path = Path(out["saved_to"])
            assert path.is_file() and path.stat().st_size > 10_000
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(path)], capture_output=True, text=True).stdout
            assert float(probe) == pytest.approx(tl2["duration"], abs=0.2)

    anyio.run(run)
