"""Verified inference-only VDN operations; reference arithmetic stays available."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


def scan_addmm(transitions, injections, start):
    state = start
    prefix = []
    for frame in range(transitions.shape[0]):
        state = mx.addmm(injections[frame], state, transitions[frame])
        prefix.append(state)
    state = start
    suffix = [None] * transitions.shape[0]
    for frame in range(transitions.shape[0] - 1, -1, -1):
        state = mx.addmm(injections[frame], state, transitions[frame])
        suffix[frame] = state
    return mx.stack(prefix), mx.stack(suffix)


@lru_cache(maxsize=16)
def gather_indices(bounds, frames):
    lower = [max(lo, 0) for lo, _ in bounds]
    upper = [min(hi + 1, frames) for _, hi in bounds]
    return mx.array(lower), mx.array(upper), mx.arange(frames)


def gather_state(prefix, suffix, alpha, text_state, lower, upper, frames):
    # Include virtual text-only boundary states, avoiding separate per-frame branches.
    before_bank = mx.concatenate((text_state[None], prefix), axis=0)
    after_bank = mx.concatenate((suffix, text_state[None]), axis=0)
    logs = mx.concatenate(
        (mx.zeros_like(alpha[:1]), mx.cumsum(mx.log(mx.maximum(alpha, 1e-12)), axis=0))
    )
    before_decay = mx.exp(logs[frames + 1] - logs[lower])
    after_decay = mx.exp(logs[upper] - logs[frames])
    return (
        before_bank[lower] * before_decay[:, :, None, :]
        + after_bank[upper] * after_decay[:, :, None, :]
    )


class VDNInferenceKernels:
    """Latch exactness per operation/geometry, and fall back on errors or differences."""

    def __init__(self, *, enabled=True, mpp=False):
        self.enabled = enabled
        self.mpp = mpp
        self.verdicts = {}
        self.fallbacks = {}
        self.calls = {}
        self.reference_calls = {}
        self.layers = {}
        self.scan_fn = mx.compile(scan_addmm)
        self.gather_fn = mx.compile(gather_state)

    def begin_run(self):
        self.calls.clear()
        self.reference_calls.clear()

    def checked(self, name, args, candidate, reference):
        key = (name, tuple((tuple(a.shape), a.dtype) for a in args))
        if self.enabled and self.verdicts.get(key) is not False:
            try:
                output = candidate(*args)
                if key not in self.verdicts:
                    expected = reference(*args)
                    outputs = output if isinstance(output, tuple) else (output,)
                    references = expected if isinstance(expected, tuple) else (expected,)
                    valid = [
                        mx.array_equal(a, b) & mx.all(mx.isfinite(a))
                        for a, b in zip(outputs, references, strict=True)
                    ]
                    mx.eval(valid)
                    if not all(v.item() for v in valid):
                        raise ValueError("bitwise_mismatch")
                    self.verdicts[key] = True
                    output = expected
                self.calls[name] = self.calls.get(name, 0) + 1
                return output
            except (RuntimeError, ValueError) as exc:
                self.verdicts[key] = False
                self.fallbacks[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
        self.reference_calls[name] = self.reference_calls.get(name, 0) + 1
        return reference(*args)

    def scan(self, transitions, injections, start, reference):
        return self.checked("scan", (transitions, injections, start), self.scan_fn, reference)

    def gather(self, prefix, suffix, alpha, text_state, bounds, reference):
        indices = gather_indices(tuple(bounds), int(alpha.shape[0]))
        return self.checked(
            "state_gather",
            (prefix, suffix, alpha, text_state, *indices),
            self.gather_fn,
            lambda p, s, a, t, *_: reference(p, s, a, t, bounds),
        )

    def project(self, value, weights, name):
        from .projection import MPPLinear, MPPTile

        weight = weights[name]
        eligible = (
            self.enabled
            and self.mpp
            and weight.dtype == mx.bfloat16
            and name
            in {
                "to_out_linear.weight",
                "linear_attention.output_gate.down.weight",
            }
        )
        if not eligible:
            return value.astype(weight.dtype) @ weight.T
        key = (id(weight), name)
        if key not in self.layers:
            base = nn.Linear(1, 1, bias=False)
            base.weight = weight
            tile = MPPTile(64, 64, 4) if name.endswith("down.weight") else MPPTile()
            self.layers[key] = MPPLinear(base, tile)
        return self.layers[key](value.astype(weight.dtype))

    def report(self):
        return {
            "enabled": self.enabled,
            "optimized_calls": dict(self.calls),
            "reference_calls": dict(self.reference_calls),
            "fallback_reasons": dict(self.fallbacks),
            "verified_geometries": sum(self.verdicts.values()),
            "auxiliary_mpp_projections": len(self.layers),
        }
