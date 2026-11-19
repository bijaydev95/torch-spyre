# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Run one built example kernel through spyre launch and the spyre_cli SDK.

Usage: python check.py <name> [kernel_dir]    (default: build/<name>)

Two checks, each reported on its own line:

- cli: ``spyre launch`` with the example's -i/-o arguments must exit 0 with no
  device error in its output.
- sdk: ``spyre_cli.launch`` with random inputs must match the CPU reference.

The CLI runs first, in its own process, before this one touches the device.
Run this script in a fresh process per example: a device fault leaves the
stream in an error state for the rest of the process.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import rednorm_examples
    from rednorm_examples import EXAMPLES as REDNORM_EXAMPLES
except ImportError:
    REDNORM_EXAMPLES = {}
    rednorm_examples = None

from examples import EXAMPLES, cli_args, expected, make_inputs, torch_dtype

DEVICE_ERRORS = ("RAS::", "StreamInErrorState", "DtException")
ARG_COUNT_SIGNATURE = "does not match number of inputs expected"

EXIT_XFAIL = 77  # expected limitation; reason starts with "XFAIL:"
ROWSUM_TOL = 0.01  # 1% tolerance for FP16 probability row-sum accumulation
FP16_MIN_NORMAL = 2.0**-14


def _xfail(message):
    print(f"XFAIL: {message}")
    sys.exit(EXIT_XFAIL)


def _first_line(exc):
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else ""


def check_cli(name, path):
    spyre = shutil.which("spyre")
    if spyre is None:
        return False, "spyre CLI not on PATH"
    result = subprocess.run(
        [spyre, "launch", *cli_args(name), str(path)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    output = result.stdout + result.stderr
    errors = [err for err in DEVICE_ERRORS if err in output]
    if result.returncode != 0 or errors:
        return False, f"exit={result.returncode} errors={errors}\n{output}"
    return True, "exit=0"


def check_sdk(name, path):
    if name in REDNORM_EXAMPLES:
        ex = REDNORM_EXAMPLES[name]
        inputs = rednorm_examples.make_inputs(name)
        plan = rednorm_examples.read_launch_plan(path, name)
        launch_fn = lambda: rednorm_examples.launch_tensors(name, plan, inputs)
        evaluate_fn = lambda got: evaluate(
            name,
            ex,
            inputs,
            got,
            ops_dict=rednorm_examples.ops(),
            planes_fn=rednorm_examples.planes,
            dtype=rednorm_examples.torch_dtype(name),
        )
        try:
            passed, detail = check_sdk_custom(path, launch_fn, evaluate_fn)
        except Exception as exc:  # noqa: BLE001
            if ARG_COUNT_SIGNATURE in str(exc):
                _xfail(
                    f"{name}: kernel takes arguments the recorded launch plan does not "
                    f"supply (known limitation): {_first_line(exc)}"
                )
            raise
        if getattr(ex, "xfail_run", "") and not passed:
            _xfail(f"{name}: {ex.xfail_run} ({detail})")
        return passed, detail

    import spyre_cli
    import torch

    ex = EXAMPLES[name]
    inputs = make_inputs(name)
    out = torch.empty(ex.output, dtype=torch_dtype(name), device="spyre")

    runner = spyre_cli.launch(*[t.to("spyre") for t in inputs], out, path=path)
    got = out.cpu()
    del runner

    want = expected(name, inputs)
    max_diff = (got.float() - want.float()).abs().max().item()
    ok = torch.allclose(got, want, atol=ex.atol, rtol=ex.rtol)
    return ok, f"max|diff|={max_diff}"


def evaluate(name, ex, inputs, got, ops_dict=None, planes_fn=None, dtype=None):
    """Run the example's invariants on one output. Returns (failures, notes)."""
    import torch

    fails, notes = [], []
    if tuple(got.shape) != tuple(ex.output):
        fails.append(f"shape {tuple(got.shape)} != expected {tuple(ex.output)}")
        return fails, notes

    # The output buffer starts zeroed, so all-zero means the kernel wrote nothing
    # (or silently ran with a mismatched shape). A constant-input LayerNorm with a
    # zero bias legitimately gives zeros, so it is not flagged there.
    init = getattr(ex, "init", "")
    if torch.count_nonzero(got) == 0 and not init.startswith("const"):
        fails.append("output is all-zero: kernel wrote nothing / silent shape mismatch")

    checks = getattr(ex, "checks", ("finite", "golden"))
    if "finite" in checks:
        n_nan = torch.isnan(got).sum().item()
        n_inf = torch.isinf(got).sum().item()
        if n_nan or n_inf:
            fails.append(f"{n_nan} NaN / {n_inf} Inf in output")

    if ops_dict is not None:
        fn = ops_dict[getattr(ex, "ref_op", None) or ex.op]
    else:
        from examples import references
        fn = references()[name]

    if dtype is None:
        if hasattr(ex, "dtype"):
            dtypes = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
            dt = dtypes.get(ex.dtype, got.dtype)
        else:
            dt = got.dtype
    else:
        dt = dtype

    plane_iter = planes_fn(inputs, got) if planes_fn is not None else [([t for t in inputs], got)]

    max_err = max_rowsum = 0.0
    n_bad = n_underflow = 0
    for xs, g in plane_iter:
        ref = fn(*[t.float() for t in xs])  # FP32 reference
        w = ref.to(dt)  # cast to the example dtype
        gf = g.float()
        max_err = max(max_err, (gf - w.float()).abs().max().item())
        n_bad += (~torch.isclose(g, w, atol=ex.atol, rtol=ex.rtol)).sum().item()
        if "rowsum" in checks:
            max_rowsum = max(max_rowsum, (gf.sum(dim=-1) - 1.0).abs().max().item())
        if "underflow" in checks:
            n_underflow += ((gf == 0) & (ref >= FP16_MIN_NORMAL)).sum().item()

    notes.append(f"max|diff|={max_err:.2e}" if max_err < 1e-2 else f"max|diff|={max_err}")
    if "golden" in checks and n_bad:
        fails.append(f"{n_bad} elements outside atol={ex.atol} rtol={ex.rtol} of the CPU reference")
    if "rowsum" in checks:
        notes.append(f"max|rowsum-1|={max_rowsum:.2e}")
        if not max_rowsum <= ROWSUM_TOL:
            fails.append(f"row sum deviates by {max_rowsum:.2e} (> {ROWSUM_TOL})")
    if "underflow" in checks and n_underflow:
        fails.append(f"{n_underflow} probabilities >= 2^-14 underflowed to exactly 0")
    if "bias" in checks:
        roles = getattr(ex, "roles", ())
        if "b" in roles:
            bias = inputs[roles.index("b")].float()
            err = (got.float() - bias).abs().max().item()
            notes.append(f"max|out-bias|={err:.2e}")
            if not err <= ex.atol:
                fails.append(f"zero-variance output differs from bias by {err:.2e} (> {ex.atol})")
    return fails, notes


def check_sdk_custom(path, launch_fn, evaluate_fn):
    """Run SDK launch once with custom launch arguments and evaluate invariants."""
    import spyre_cli

    launch_args, out = launch_fn()
    runner = spyre_cli.launch(*launch_args, path=path)
    got = out.cpu()
    del runner

    fails, notes = evaluate_fn(got)
    return not fails, "; ".join(notes + fails)


def main():
    name = sys.argv[1]
    if len(sys.argv) > 2:
        path = Path(sys.argv[2])
    else:
        path = Path(__file__).parent / "build" / name

    if name in REDNORM_EXAMPLES:
        passed, detail = check_sdk(name, path)
        print(f"{name} sdk: {'PASS' if passed else 'FAIL'} {detail}")
        sys.exit(0 if passed else 1)

    ok = True
    for label, check in (("cli", check_cli), ("sdk", check_sdk)):
        passed, detail = check(name, path)
        print(f"{name} {label}: {'PASS' if passed else 'FAIL'} {detail}")
        ok = ok and passed
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
