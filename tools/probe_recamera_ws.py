#!/usr/bin/env python3
"""Probe reCamera WebSocket 8090 and print raw detection metadata.

Useful because the browser preview can make different models look identical.
This script reads a few WebSocket frames and prints code/boxes/classes/confidence
without relying on the UI overlay.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
from typing import Any


def read_frame(sock: socket.socket) -> tuple[bool, int, bytes]:
    hdr = sock.recv(2)
    if len(hdr) != 2:
        raise EOFError("short websocket header")
    b1, b2 = hdr
    fin = bool(b1 & 0x80)
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", sock.recv(8))[0]
    mask = sock.recv(4) if masked else b""
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(65536, length - len(data)))
        if not chunk:
            raise EOFError("socket closed while reading frame")
        data += chunk
    if masked:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    return fin, b1 & 0x0F, data


def read_message(sock: socket.socket) -> tuple[int, bytes]:
    fin, opcode, data = read_frame(sock)
    chunks = [data]
    first_opcode = opcode
    while not fin:
        fin, opcode, data = read_frame(sock)
        chunks.append(data)
    return first_opcode, b"".join(chunks)


def summarize_json(obj: dict[str, Any]) -> dict[str, Any]:
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    boxes = data.get("boxes") or []
    labels = data.get("labels") or []
    keypoints = data.get("keypoints") or []
    return {
        "code": obj.get("code"),
        "msg": obj.get("msg"),
        "box_count": len(boxes),
        "boxes_head": boxes[:5],
        "label_count": len(labels),
        "labels_head": labels[:5],
        "keypoint_count": len(keypoints),
        "keypoints_head": keypoints[:2],
        "perf": data.get("perf"),
        "resolution": data.get("resolution"),
        "data_keys": sorted(data.keys()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.21.37.9")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--raw", action="store_true", help="Print raw JSON payloads instead of summaries")
    args = ap.parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
    sock.settimeout(args.timeout)
    key = base64.b64encode(os.urandom(16)).decode()
    request = (
        f"GET / HTTP/1.1\r\n"
        f"Host: {args.host}:{args.port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Origin: http://{args.host}\r\n"
        "\r\n"
    )
    sock.sendall(request.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(1024)
    print(resp.decode(errors="replace").split("\r\n")[0])

    for idx in range(args.frames):
        opcode, payload = read_message(sock)
        print(f"frame={idx} opcode={opcode} bytes={len(payload)}")
        if not payload.startswith(b"{"):
            print(payload[:120])
            continue
        try:
            obj = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            print(payload[:500])
            continue
        if args.raw:
            print(json.dumps(obj, ensure_ascii=False)[:4000])
        else:
            print(json.dumps(summarize_json(obj), ensure_ascii=False, indent=2))
    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
