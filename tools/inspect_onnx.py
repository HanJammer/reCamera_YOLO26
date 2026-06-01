#!/usr/bin/env python3
"""Inspect ONNX tensor names relevant to Seeed reCamera conversion.

The Seeed wiki command hardcodes internal YOLO11 tensor names. This tool prints
actual graph outputs and candidate YOLO head tensors so the conversion command can
be adapted to the real ONNX file instead of guessing.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import onnx


def tensor_shape(value_info: onnx.ValueInfoProto) -> str:
    try:
        dims = value_info.type.tensor_type.shape.dim
        out = []
        for dim in dims:
            if dim.dim_value:
                out.append(str(dim.dim_value))
            elif dim.dim_param:
                out.append(dim.dim_param)
            else:
                out.append("?")
        return "[" + ",".join(out) + "]"
    except Exception:
        return "[?]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--grep", default=r"model\.|cv2|cv3|Detect|dfl|Concat|Conv")
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    graph = model.graph

    print(f"file: {args.onnx}")
    print(f"ir_version: {model.ir_version}")
    print("opsets:", ", ".join(f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import))

    print("\n== inputs ==")
    for x in graph.input:
        print(f"{x.name}\t{tensor_shape(x)}")

    print("\n== graph outputs ==")
    for x in graph.output:
        print(f"{x.name}\t{tensor_shape(x)}")

    pattern = re.compile(args.grep, re.I)
    print("\n== candidate nodes/tensors ==")
    for i, node in enumerate(graph.node):
        haystack = " ".join([node.name, node.op_type, *node.input, *node.output])
        if pattern.search(haystack):
            outs = ", ".join(node.output)
            print(f"{i:04d}\t{node.op_type}\t{node.name}\t=> {outs}")

    print("\n== exact Seeed YOLO11 detection names present? ==")
    names = {out for node in graph.node for out in node.output} | {o.name for o in graph.output}
    seeed = [
        "/model.23/cv2.0/cv2.0.2/Conv_output_0",
        "/model.23/cv3.0/cv3.0.2/Conv_output_0",
        "/model.23/cv2.1/cv2.1.2/Conv_output_0",
        "/model.23/cv3.1/cv3.1.2/Conv_output_0",
        "/model.23/cv2.2/cv2.2.2/Conv_output_0",
        "/model.23/cv3.2/cv3.2.2/Conv_output_0",
    ]
    for name in seeed:
        print(("YES" if name in names else "NO ") + "\t" + name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
