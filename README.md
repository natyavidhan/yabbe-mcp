# YABBE MCP

Let an AI edit your videos. This is an [MCP](https://modelcontextprotocol.io) server for
[YABBE](https://github.com/natyavidhan/YABBE), the self-hosted browser video editor. Connect it to
Claude, Codex, OpenCode, Cursor, Gemini CLI or any other MCP client and give it your footage and a
brief, like "a 30 second vertical reel from these clips, cut to the music, with a title". It builds the
edit in YABBE, where you can open it, adjust it by hand, and export it.

Every feature of the editor is available as a tool:

| | |
|---|---|
| **Projects & sequences** | create, open, rename, duplicate, delete; canvas presets: YouTube, 4K, vertical 9:16, square, 4:5, cinema; several sequences per project, nesting / pre-composing |
| **Media** | upload local files, import from a link, YouTube music search → MP3, list, delete |
| **Seeing & hearing** | `look_at_media` (contact sheet of frames at chosen times), `listen_to_media` (loudness, silences, onsets, tempo and beat times), `preview_frame` / `preview_frames` of the edit itself |
| **Timeline** | add media, text and shapes, move, trim, slip, split, delete (with ripple), duplicate, speed (0.25-4×), transform, opacity, blend modes, volume, fades, crop, keyframe animation with easing, 70+ transitions, markers, tracks (add / reorder / mute / hide / lock), linking, separate audio, freeze frames |
| **Effects** | chroma key, roto brush (AI cut-out of people / objects, with a preview of the selection), motion tracking (point, transform, corner pin), attaching / pinning clips to trackers, stabilize |
| **Lyrics** | word-by-word synced song lyrics in 12 styles, all restylable, with line fixes; found on LRCLIB, timed to the recording, 1100+ languages |
| **Output** | export MP4 (quality, size, any sequence), download to a local folder, job progress / cancel |
| **Safety** | `undo` for every edit made through the tools; deleting a project needs explicit confirmation |

## Setup

You need:
- a running YABBE (see its README: `docker run -d -p 8000:8000 -v yabbe-data:/data ghcr.io/natyavidhan/yabbe`);
- [uv](https://docs.astral.sh/uv/).

The MCP server runs on the machine that has your files. YABBE can be anywhere (`YABBE_URL`).

Every client runs this command:

```sh
uvx --from git+https://github.com/natyavidhan/yabbe-mcp yabbe-mcp
```

Set `YABBE_URL` if YABBE isn't at `http://localhost:8000`.

### Claude Code

```sh
claude mcp add yabbe -e YABBE_URL=http://localhost:8000 -- uvx --from git+https://github.com/natyavidhan/yabbe-mcp yabbe-mcp
```

### Claude Desktop

`claude_desktop_config.json` (Settings → Developer → Edit config):

```json
{
  "mcpServers": {
    "yabbe": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/natyavidhan/yabbe-mcp", "yabbe-mcp"],
      "env": { "YABBE_URL": "http://localhost:8000" }
    }
  }
}
```

### Codex CLI

`~/.codex/config.toml`:

```toml
[mcp_servers.yabbe]
command = "uvx"
args = ["--from", "git+https://github.com/natyavidhan/yabbe-mcp", "yabbe-mcp"]
env = { YABBE_URL = "http://localhost:8000" }
startup_timeout_sec = 60
tool_timeout_sec = 3600   # exports and lyrics can take minutes
```

### OpenCode

`opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "yabbe": {
      "type": "local",
      "command": ["uvx", "--from", "git+https://github.com/natyavidhan/yabbe-mcp", "yabbe-mcp"],
      "environment": { "YABBE_URL": "http://localhost:8000" },
      "enabled": true
    }
  }
}
```

### Cursor / Gemini CLI / VS Code

Cursor (`.cursor/mcp.json`) and Gemini CLI (`~/.gemini/settings.json`) use the same `mcpServers`
block as Claude Desktop. VS Code (`.vscode/mcp.json`) uses `"servers"` instead of `"mcpServers"`.

### Over HTTP

For clients that connect to a URL, or to run it next to YABBE on a server:

```sh
YABBE_URL=http://localhost:8000 uvx --from git+https://github.com/natyavidhan/yabbe-mcp yabbe-mcp --http --port 8765
# → http://127.0.0.1:8765/mcp   (--host 0.0.0.0 to listen on the network; it has no login of its own)
```

Over HTTP, `upload_media` reads paths on the machine running the MCP server.

## Using it

Just ask, for example:

> Make a 20 second vertical reel from ~/Videos/trip/*.mp4 with the song "Blinding Lights", cut on the beat,
> with the lyrics in the Pop style and a title "Lisbon '26" at the start. Export it to ~/Desktop.

The server also gives clients:
- an **`edit_video` prompt** (a brief plus files);
- a **`yabbe://guide`** resource with editing tips.

A typical AI run:
1. Create the project and upload the media.
2. `look_at_media` on every clip and `listen_to_media` on the song.
3. Plan the edit.
4. Build it with `add_clip`, `trim_clip`, `set_transition`, `add_text`, `add_lyrics` and the other tools.
5. Check it with `preview_frames`.
6. `export_video`.

**Watch it live:** keep the project open in the YABBE editor while the AI works. Its changes appear
as they're made (the header shows "AI editing…"), and you can keep editing too: edits made at the same
moment are merged, not overwritten. The AI re-applies its change on top of yours, and the editor merges
yours on top of the AI's. Ctrl+Z in the editor undoes the AI's last change. This needs a YABBE from
October 2026 or later; older versions don't have live sync.

## Settings

| Variable / flag | Default | |
|---|---|---|
| `YABBE_URL` / `--url` | `http://localhost:8000` | Where YABBE is |
| `YABBE_TOKEN` | — | Sent as `Authorization: Bearer …`, for a YABBE behind an auth proxy |
| `--http`, `--host`, `--port` | stdio | Serve over streamable HTTP |

## Development

```sh
uv sync
uv run pytest                       # see below
uv run mcp dev src/yabbe_mcp/server.py   # MCP Inspector
```

`uv run pytest` runs the unit tests. It also runs the end-to-end test when `../YABBE/backend` (or
`YABBE_BACKEND`) has its venv: that test starts a YABBE and edits through a real MCP client, from upload
to exported MP4.

The timeline editing rules (splitting, retiming, linking, nesting, freeze frames, ...) mirror the YABBE
editor's, in `src/yabbe_mcp/timeline.py`. For each edit the server loads the project, applies the change
and saves it back through YABBE's API.
