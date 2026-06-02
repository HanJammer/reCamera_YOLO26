#!/usr/bin/env python3
"""Generate a reCamera `model_transform` command using actual ONNX tensor names.

For YOLO detect models, Seeed avoids final `output0` and asks TPU-MLIR to emit the
six tensors before the YOLO head: three regression branch outputs (`cv2.*`) and
three class branch outputs (`cv3.*`). The exact names are exporter-dependent.

This script searches the ONNX graph for those six Conv outputs and prints a
`model_transform` command. It intentionally refuses to guess if it cannot find
exactly one ordered set.
"""
from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import onnx

BRANCH_RE = re.compile(
    r"(?P<prefix>.*/model\.(?P<model_idx>\d+))/(?:(?:one2one_)?cv(?P<branch>[23])\.(?P<scale>\d+))/(?:(?:one2one_)?cv[23]\.\d+\.2)/(?:Conv_output_0)$"
)
LOOSE_RE = re.compile(r"/model\.(?P<model_idx>\d+)/(?:one2one_)?cv(?P<branch>[23])\.(?P<scale>\d+)/.*Conv", re.I)


def all_tensor_names(model: onnx.ModelProto) -> list[str]:
    names: list[str] = []
    for node in model.graph.node:
        names.extend(node.output)
    names.extend(o.name for o in model.graph.output)
    return names


def derive_outputs(names: list[str]) -> list[str]:
    exact: dict[tuple[int, int, int], str] = {}
    loose: dict[tuple[int, int, int], str] = {}

    for name in names:
        m = BRANCH_RE.match(name)
        if m:
            key = (int(m.group("model_idx")), int(m.group("scale")), int(m.group("branch")))
            exact[key] = name
            continue
        m = LOOSE_RE.search(name)
        if m and name.endswith("Conv_output_0"):
            key = (int(m.group("model_idx")), int(m.group("scale")), int(m.group("branch")))
            loose.setdefault(key, name)

    for source in (exact, loose):
        if not source:
            continue
        model_indices = sorted({k[0] for k in source})
        # Prefer the highest model index; YOLO Detect is usually the final module.
        for idx in reversed(model_indices):
            keys = [(idx, scale, branch) for scale in range(3) for branch in (2, 3)]
            if all(k in source for k in keys):
                return [source[k] for k in keys]

    return []


def shell_join(parts: list[str]) -> str:
    return " \\\n  ".join(shlex.quote(p) for p in parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--input-shapes", default="[[1,3,640,640]]")
    ap.add_argument("--mean", default="0.0,0.0,0.0")
    ap.add_argument("--scale", default="0.0039216,0.0039216,0.0039216")
    ap.add_argument("--pixel-format", default="rgb")
    ap.add_argument("--test-input", default="../image/dog.jpg")
    ap.add_argument("--test-result")
    ap.add_argument("--mlir")
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    outputs = derive_outputs(all_tensor_names(model))

    if len(outputs) != 6:
        print("ERROR: could not derive exactly six YOLO pre-head outputs.")
        print("Run: python tools/inspect_onnx.py", args.onnx)
        print("Then inspect candidates manually in Netron.")
        return 2

    test_result = args.test_result or f"{args.model_name}_top_outputs.npz"
    mlir = args.mlir or f"{args.model_name}.mlir"
    cmd = [
        "model_transform",
        "--model_name", args.model_name,
        "--model_def", str(args.onnx),
        "--input_shapes", args.input_shapes,
        "--mean", args.mean,
        "--scale", args.scale,
        "--keep_aspect_ratio",
        "--pixel_format", args.pixel_format,
        "--output_names", ",".join(outputs),
        "--test_input", args.test_input,
        "--test_result", test_result,
        "--mlir", mlir,
    ]

    print("# Derived output_names:")
    for out in outputs:
        print("#  ", out)
    print()
    print(shell_join(cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
