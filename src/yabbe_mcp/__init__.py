"""YABBE MCP: edit videos in YABBE from any MCP client (Claude, Codex, OpenCode, ...)."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="yabbe-mcp", description="MCP server for the YABBE video editor")
    ap.add_argument("--url", help="YABBE address (default: $YABBE_URL or http://localhost:8000)")
    ap.add_argument("--http", action="store_true", help="serve over streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.url:
        os.environ["YABBE_URL"] = args.url
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise

    from .server import mcp

    if args.http:
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()
