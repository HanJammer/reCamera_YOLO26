#!/usr/bin/env python3
"""Upload a ready .cvimodel to reCamera firmware API.

This bypasses the firmware ONNX converter. Use only with an already converted
.cvimodel. The script mirrors the web UI's /api/deviceMgr/uploadModel behavior
and can use 512 KiB chunked upload for larger model files.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import pathlib
import sys
from typing import Any

import requests

DEFAULT_CHUNK = 512 * 1024


def read_classes(path: pathlib.Path | None) -> list[str]:
    if path is None:
        path = pathlib.Path(__file__).resolve().parents[1] / "data" / "coco80.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def post_single(base: str, model: pathlib.Path, info: dict[str, Any], headers: dict[str, str]) -> requests.Response:
    with model.open("rb") as fh:
        files = {
            "model_file": (model.name, fh, mimetypes.guess_type(model.name)[0] or "application/octet-stream"),
        }
        data = {"model_info": json.dumps(info, ensure_ascii=False)}
        return requests.post(f"{base}/api/deviceMgr/uploadModel", files=files, data=data, headers=headers, timeout=120)


def post_chunked(base: str, model: pathlib.Path, info: dict[str, Any], headers: dict[str, str], chunk_size: int) -> requests.Response:
    total = model.stat().st_size
    offset = 0
    last_resp: requests.Response | None = None
    with model.open("rb") as fh:
        while offset < total:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            end = offset + len(chunk)
            files = {
                "model_file": (model.name, chunk, "application/octet-stream"),
            }
            data = {
                "offset": str(offset),
                "size": str(total),
            }
            if end >= total:
                data["model_info"] = json.dumps(info, ensure_ascii=False)
            last_resp = requests.post(f"{base}/api/deviceMgr/uploadModel", files=files, data=data, headers=headers, timeout=120)
            print(f"uploaded {end}/{total} ({end * 100 / total:.1f}%) status={last_resp.status_code}")
            if not last_resp.ok:
                return last_resp
            offset = end
    if last_resp is None:
        raise RuntimeError("no data uploaded")
    return last_resp


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload .cvimodel to reCamera /api/deviceMgr/uploadModel")
    ap.add_argument("model", type=pathlib.Path, help="Path to .cvimodel")
    ap.add_argument("--host", required=True, help="reCamera host/IP, e.g. 192.168.1.50")
    ap.add_argument("--model-name", default=None, help="Display model name; default is file stem")
    ap.add_argument("--model-id", default="0")
    ap.add_argument("--classes", type=pathlib.Path, help="Class list text file, one class per line; default data/coco80.txt")
    ap.add_argument("--description", default="YOLO model converted locally for Seeed reCamera CV181x")
    ap.add_argument("--author", default="local TPU-MLIR converter")
    ap.add_argument("--single", action="store_true", help="Use single multipart upload instead of chunked upload")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--authorization", default=None, help="Optional Authorization header value copied from browser session")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model = args.model
    if not model.exists():
        print(f"missing model: {model}", file=sys.stderr)
        return 2
    if model.suffix.lower() != ".cvimodel":
        print("warning: file does not end with .cvimodel", file=sys.stderr)

    base = f"http://{args.host}" if not args.host.startswith(("http://", "https://")) else args.host.rstrip("/")
    info = {
        "model_id": str(args.model_id),
        "model_name": args.model_name or model.stem,
        "model_format": "cvimodel",
        "ai_framwork": 6,
        "description": args.description,
        "author": args.author,
        "classes": read_classes(args.classes),
    }
    headers: dict[str, str] = {}
    if args.authorization:
        headers["Authorization"] = args.authorization

    print(json.dumps({"base": base, "model": str(model), "size": model.stat().st_size, "model_info": info}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    resp = post_single(base, model, info, headers) if args.single else post_chunked(base, model, info, headers, args.chunk_size)
    print("response status:", resp.status_code)
    print(resp.text[:2000])
    return 0 if resp.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
