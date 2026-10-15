"""A small async client for the YABBE HTTP API."""

from __future__ import annotations

import asyncio
import mimetypes
import os
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
from urllib.parse import quote

import httpx

Progress = Callable[[float, str], Awaitable[None]]


class YabbeError(Exception):
    pass


class Yabbe:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        headers = {"User-Agent": "yabbe-mcp"}
        token = os.environ.get("YABBE_TOKEN")  # for a YABBE behind an auth proxy
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.AsyncClient(base_url=self.base, timeout=timeout, headers=headers, follow_redirects=True)

    def url(self, path: str) -> str:
        return self.base + path

    async def _check(self, r: httpx.Response) -> httpx.Response:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
                if isinstance(detail, list):
                    detail = "; ".join(d.get("msg", str(d)) for d in detail)
            except Exception:  # noqa: BLE001
                detail = r.text[:300]
            raise YabbeError(f"YABBE said {r.status_code}: {detail}")
        return r

    async def request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = await self.http.request(method, path, **kw)
        except httpx.ConnectError:
            raise YabbeError(f"Can't reach YABBE at {self.base}. Is it running? (set YABBE_URL)") from None
        return await self._check(r)

    async def get(self, path: str, **kw: Any) -> Any:
        return (await self.request("GET", path, **kw)).json()

    async def post(self, path: str, body: Any = None, **kw: Any) -> Any:
        return (await self.request("POST", path, json=body, **kw)).json()

    async def put(self, path: str, body: Any) -> Any:
        return (await self.request("PUT", path, json=body)).json()

    async def delete(self, path: str) -> Any:
        return (await self.request("DELETE", path)).json()

    async def raw(self, method: str, path: str, **kw: Any) -> bytes:
        return (await self.request(method, path, **kw)).content

    # -- projects ------------------------------------------------------------------------------

    async def project(self, pid: str) -> dict:
        return await self.get(f"/api/projects/{quote(pid)}")

    async def save_sequences(self, project: dict) -> dict:
        return await self.put(f"/api/projects/{quote(project['id'])}",
                              {"sequences": project["sequences"], "main_sequence_id": project["main_sequence_id"]})

    # -- media -------------------------------------------------------------------------------------

    async def upload(self, pid: str, path: Path, progress: Optional[Progress] = None) -> dict:
        size = path.stat().st_size
        sent = 0

        async def body() -> AsyncIterator[bytes]:
            nonlocal sent
            with open(path, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    sent += len(chunk)
                    if progress:
                        await progress(sent / max(1, size), f"Uploading {path.name}")
                    yield chunk

        headers = {"Content-Type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                   "Content-Length": str(size)}
        r = await self.request("POST", f"/api/projects/{quote(pid)}/media?filename={quote(path.name)}",
                               content=body(), headers=headers, timeout=None)
        return r.json()

    async def wait_media(self, pid: str, ids: Optional[set[str]] = None, timeout: float = 1800,
                         progress: Optional[Progress] = None) -> list[dict]:
        """Wait until media is processed (proxies, thumbnails, waveforms)."""
        t0 = asyncio.get_running_loop().time()
        while True:
            assets = (await self.project(pid))["assets"]
            mine = [a for a in assets if ids is None or a["id"] in ids]
            busy = [a for a in mine if a["status"] == "processing"]
            if not busy:
                return mine
            if progress:
                await progress(1 - len(busy) / max(1, len(mine)), f"Processing {busy[0]['original_name']}")
            if asyncio.get_running_loop().time() - t0 > timeout:
                raise YabbeError("Media is still processing; try again in a bit")
            await asyncio.sleep(1.5)

    # -- jobs ------------------------------------------------------------------------------------------

    async def wait_job(self, job_id: str, progress: Optional[Progress] = None, timeout: float = 7200) -> dict:
        t0 = asyncio.get_running_loop().time()
        while True:
            job = await self.get(f"/api/jobs/{job_id}")
            if job["status"] not in ("queued", "running"):
                if job["status"] == "error":
                    raise YabbeError(job.get("error") or "The job failed")
                if job["status"] == "cancelled":
                    raise YabbeError("The job was cancelled")
                return job
            if progress:
                await progress(float(job.get("progress") or 0), job.get("message") or job.get("label") or "Working")
            if asyncio.get_running_loop().time() - t0 > timeout:
                raise YabbeError(f"Still running after {timeout:.0f}s (job {job_id}); check again with get_job")
            await asyncio.sleep(1.0)

    async def close(self) -> None:
        await self.http.aclose()
