#!/usr/bin/env python3
"""Benchmark raw MPP low-bit operations at exact H3 projection shapes."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from functools import partial
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.mpp_low_bit import mpp_low_bit_linear
from minimax_h3_mlx.projection import MPPTile, mpp_bf16_linear

PROJECTIONS = (
    ("attn.qkv_proj", 5_376, 21_504, MPPTile()),
    ("attn.out_proj", 7_168, 5_376, MPPTile()),
    ("mlp.fc1", 5_376, 28_672, MPPTile()),
    ("mlp.fc2", 14_336, 5_376, MPPTile(64, 128, 8)),
)


def _measure(operation, warmups: int, repetitions: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        mx.eval(operation())
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(operation())
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def _mpp_bf16(source, weight, *, tile):
    return mpp_bf16_linear(source, weight, tile=tile)


def _mpp_low_bit(source, weight, *, bits, output_dim, tile):
    return mpp_low_bit_linear(source, weight, bits=bits, output_dim=output_dim, tile=tile)


def _case(rows, name, input_dim, output_dim, tile, bits, warmups, repetitions):
    source = mx.full((rows, input_dim), 0.015625, dtype=mx.bfloat16)
    if bits == 16:
        weight = mx.full((output_dim, input_dim), 0.03125, dtype=mx.bfloat16)
        operation = partial(_mpp_bf16, source, weight, tile=tile)
        expected = input_dim * 0.015625 * 0.03125
        storage_bytes = weight.nbytes
        label = "bf16"
    elif bits == 8:
        weight = mx.ones((output_dim, input_dim), dtype=mx.int8)
        operation = partial(
            _mpp_low_bit,
            source,
            weight,
            bits=8,
            output_dim=output_dim,
            tile=tile,
        )
        expected = input_dim * 0.015625
        storage_bytes = weight.nbytes
        label = "int8"
    else:
        weight = mx.full((output_dim, input_dim // 2), 0x11, dtype=mx.uint8)
        operation = partial(
            _mpp_low_bit,
            source,
            weight,
            bits=4,
            output_dim=output_dim,
            tile=tile,
        )
        expected = input_dim * 0.015625
        storage_bytes = weight.nbytes
        label = "packed_int4"
    mx.eval(source, weight)
    median, samples = _measure(operation, warmups, repetitions)
    result = operation()
    sample = result[0, 0]
    mx.eval(sample)
    record = {
        "rows": rows,
        "projection": name,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "format": label,
        "tile_m": tile.rows,
        "tile_n": tile.columns,
        "simdgroups": tile.simdgroups,
        "median_seconds": median,
        "samples_seconds": samples,
        "tflops_equivalent": 2.0 * rows * input_dim * output_dim / median / 1e12,
        "weight_storage_bytes": storage_bytes,
        "explicit_tensor_bytes": source.nbytes + storage_bytes + result.nbytes,
        "output_sample": float(sample.item()),
        "expected_sample": expected,
        "sample_absolute_error": abs(float(sample.item()) - expected),
    }
    del result, operation, source, weight
    gc.collect()
    mx.clear_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[9_477, 25_138])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    if min(args.rows) < 1 or min(args.warmups, args.repetitions) < 1:
        parser.error("rows, warmups, and repetitions must be positive")

    records = []
    for rows in args.rows:
        for projection in PROJECTIONS:
            for bits in (16, 8, 4):
                records.append(
                    _case(
                        rows,
                        *projection,
                        bits,
                        args.warmups,
                        args.repetitions,
                    )
                )

    payload = {"device": mx.device_info(), "records": records}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [key for key in records[0] if key != "samples_seconds"]
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in records)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
