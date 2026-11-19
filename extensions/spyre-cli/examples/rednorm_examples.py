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


"""spyre-cli workload examples for Reductions & Normalizations.

Covers extended test scenarios for RMSNorm, LayerNorm, and Softmax.
Each example pushes one op through ``torch.compile`` and the ``spyre_cli`` SDK.

    =========================================  ==========  ====================  =======
    example                                    workload    category              dtype
    =========================================  ==========  ====================  =======
    rmsnorm_bf16                               RMSNorm     bf16                  bf16
    rmsnorm_variance_{tiny,huge,mixed}_bf16    RMSNorm     variance_stability    bf16
    rmsnorm_native_bf16                        RMSNorm     native                bf16
    layernorm_manual_variance_fp16             LayerNorm   manual_variance       fp16
    layernorm_tensor_var_fp16  (blocked)       LayerNorm   tensor_var            fp16
    layernorm_native_fp16                      LayerNorm   native                fp16
    layernorm_fp32_accumulate_fp16             LayerNorm   fp32_accumulate       fp16
    layernorm_zero_variance_*_fp16 (4)         LayerNorm   zero_variance         fp16
    softmax_spec_fp16                          Softmax     large_shape           fp16
    softmax_fp32_accumulate_{spec,small}_fp16  Softmax     fp32_accumulate       fp16
    softmax_sum_to_one_fp16                    Softmax     rowsum                fp16
    softmax_underflow{,_fp32_accumulate}_fp16  Softmax     underflow             fp16
    =========================================  ==========  ====================  =======

Several examples reuse the kernel of another one (field ``bundle``) and only
change the input data, so they share one compile. Importing this module does
not import torch, so tests can be collected without a Spyre device.

Run as a script to compile example kernels on the current toolchain:

    python rednorm_examples.py [--out DIR] [name ...]    (default: every example)
    python rednorm_examples.py list

Kernels are not checked in, since they only load on the toolchain that built
them. Each kernel is compiled in a fresh process with its own Inductor cache, so
exactly one Spyre kernel directory is produced, and it is copied to DIR/<name>
(default build/<name>). Examples that compile to more than one kernel are
reported as an expected limitation (exit 77), since spyre launch runs one.
"""

import argparse
import dataclasses
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "build"

EXIT_XFAIL = 77  # expected limitation; the reason is on a line starting "XFAIL:"
PLAN_FILE = "launch_plan.json"  # written next to spyreCodeDir when a kernel is built

# (atol, rtol). 16-bit floats keep 8 (bf16) or 11 (fp16) significant bits, so a
# one-ulp difference against the fp32 CPU reference can reach ~0.8% in bf16.
TOL = (0.02, 0.02)
EPS = 1e-5  # value of the "eps" tensor input
ROWSUM_TOL = 0.01  # 1% tolerance for FP16 probability row-sum accumulation
FP16_MIN_NORMAL = 2.0**-14  # a probability at or above this must not flush to 0
DEFAULT_CHECKS = ("finite", "golden")


@dataclasses.dataclass(frozen=True)
class Example:
    roles: tuple  # role of each input, in kernel argument order: x, w, b or eps
    inputs: tuple  # input shapes, same order
    output: tuple  # output shape
    workload: str  # workload name, e.g. "RMSNorm", "LayerNorm", "Softmax"
    category: str  # test category / feature
    title: str  # human-readable description
    dtype: str  # a spyre launch dtype: fp16, fp32 or bf16
    op: str  # key into ops()
    ref_op: str = ""  # op used for the CPU reference, if it differs from ``op``
    # How the "x" input is drawn (see make_inputs):
    #   "randn"        N(0, 1)
    #   "randn:S"      N(0, S^2), e.g. randn:1e-3 gives variance 1e-6
    #   "mixed"        row r has std-dev 10**(-3 + 6 r / (rows - 1))
    #   "const:V"      every element is V (zero-variance rows)
    init: str = "randn"
    bundle: str = ""  # reuse the compiled kernel of this example; "" builds its own
    checks: tuple = DEFAULT_CHECKS
    tol: tuple = TOL  # (atol, rtol) against the CPU reference
    xfail_build: str = ""  # reason, when the op is known not to compile for Spyre
    xfail_run: str = ""  # reason, when the op is expected to deviate at runtime
    timeout: int = 600  # seconds for the CLI launch; expiry is reported as a lockup

    @property
    def atol(self):
        return self.tol[0]

    @property
    def rtol(self):
        return self.tol[1]


EXAMPLES: dict = {}


def add(name, workload, category, title, dtype, op, roles, inputs, output, **kw):
    EXAMPLES[name] = Example(tuple(roles), tuple(inputs), tuple(output), workload, category, title, dtype, op, **kw)


def norm(name, workload, category, title, dtype, op, roles, shape, **kw):
    """A normalization over the last axis of ``shape``; the output has the same shape."""
    d = shape[-1]
    table = {"x": shape, "w": (d,), "b": (d,), "eps": shape[:-1] + (1,)}
    add(name, workload, category, title, dtype, op, roles, [table[r] for r in roles], shape, **kw)


def _register():
    # ----------------------------------------------------------------------
    # RMSNorm, shape [B=1, S=2048, D=4096], BF16
    # ----------------------------------------------------------------------
    spec = (1, 2048, 4096)
    norm("rmsnorm_bf16", "RMSNorm", "bf16", "RMSNorm, BF16", "bf16", "rms_manual", ("x", "eps"), spec,
         checks=("finite", "golden"))
    # Variance-reduction stability. Same kernel, std-dev 1e-3 / 1e3 / mixed rows
    # (variance 1e-6 .. 1e6). At the tiny scale eps dominates the variance; the CPU
    # reference does the same, so the comparison stays valid.
    for tag, init, title in (
        ("tiny", "randn:1e-3", "variance 1e-6"),
        ("huge", "randn:1e3", "variance 1e6"),
        ("mixed", "mixed", "rows from variance 1e-6 to 1e6"),
    ):
        norm(f"rmsnorm_variance_{tag}_bf16", "RMSNorm", "variance_stability", f"RMSNorm stability, {title}", "bf16",
             "rms_manual", ("x", "eps"), spec, init=init, bundle="rmsnorm_bf16")
    # Native F.rms_norm at D=4096 (eps is a scalar kernel argument)
    norm("rmsnorm_native_bf16", "RMSNorm", "native", "F.rms_norm at D=4096", "bf16", "rms_native", ("x",), spec)

    # ----------------------------------------------------------------------
    # LayerNorm, shape [B=1, S=512, D=768], FP16 I/O
    # ----------------------------------------------------------------------
    spec = (1, 512, 768)
    affine = ("x", "w", "b", "eps")
    var_gap = "Tensor.var has no Spyre kernel (#1688)"
    # Variance as mean((x - mean)^2), the flow that works today, and the
    # Tensor.var flow that the existing tests mark xfail
    norm("layernorm_manual_variance_fp16", "LayerNorm", "manual_variance", "LayerNorm, manual variance", "fp16",
         "ln_manual", affine, spec, checks=("finite", "golden"))
    norm("layernorm_tensor_var_fp16", "LayerNorm", "tensor_var", "LayerNorm via Tensor.var (blocked)", "fp16",
         "ln_var", affine, spec, ref_op="ln_manual", xfail_build=var_gap)
    # Shape with FP32 accumulate and FP16 I/O, and the native F.layer_norm
    norm("layernorm_native_fp16", "LayerNorm", "native", "F.layer_norm at [1,512,768]", "fp16",
         "ln_native", ("x", "w", "b"), spec)
    norm("layernorm_fp32_accumulate_fp16", "LayerNorm", "fp32_accumulate", "LayerNorm, FP32 accumulate", "fp16",
         "ln_fp32acc", affine, spec, ref_op="ln_manual", checks=("finite", "golden"))
    # Zero-variance rows must not give NaN. The normalized term is 0, so the
    # output must equal the bias. 2.0 sums exactly in FP16 (768 * 2.0), so the mean
    # has no rounding error.
    for kernel in ("manual_variance", "fp32_accumulate"):
        for tag, init in (("const2", "const:2.0"), ("zeros", "const:0.0")):
            xfail_run = ""
            if kernel == "manual_variance" and tag == "const2":
                xfail_run = "FP16 mean reduction precision residual amplified by subnormal eps (use FP32 accumulation)"
            norm(f"layernorm_zero_variance_{tag}_{kernel}_fp16", "LayerNorm", "zero_variance",
                 f"LayerNorm zero variance ({tag}), {kernel.replace('_', ' ')}", "fp16",
                 "ln_manual" if kernel == "manual_variance" else "ln_fp32acc", affine, spec,
                 ref_op="ln_manual", init=init, bundle=f"layernorm_{kernel}_fp16", checks=("finite", "bias"),
                 xfail_run=xfail_run)

    # ----------------------------------------------------------------------
    # Softmax, shapes [4, 32, 2048, 2048] and [1, 8, 512, 512], FP16 I/O
    # ----------------------------------------------------------------------
    big, small = (4, 32, 2048, 2048), (1, 8, 512, 512)
    rows = ("finite", "golden", "rowsum")
    add("softmax_spec_fp16", "Softmax", "large_shape", "Softmax at [4,32,2048,2048]", "fp16", "softmax_last",
        ("x",), (big,), big, checks=rows)
    # FP32 accumulate with FP16 I/O
    add("softmax_fp32_accumulate_spec_fp16", "Softmax", "fp32_accumulate", "Softmax FP32 accumulate, large shape", "fp16",
        "softmax_fp32acc", ("x",), (big,), big, ref_op="softmax_last", checks=rows)
    add("softmax_fp32_accumulate_small_fp16", "Softmax", "fp32_accumulate", "Softmax FP32 accumulate, [1,8,512,512]",
        "fp16", "softmax_fp32acc", ("x",), (small,), small, ref_op="softmax_last", checks=rows)
    # Row sums: sum p_i = 1.0 +/- 1e-4
    add("softmax_sum_to_one_fp16", "Softmax", "rowsum", "Softmax row sums, [1,8,512,512]", "fp16",
        "softmax_last", ("x",), (small,), small, checks=("finite", "rowsum"))
    # Logits with std-dev 15 span about +/-45; any probability at or above the
    # FP16 normal minimum that reads back as 0 is an underflow
    under = ("finite", "rowsum", "underflow")
    add("softmax_underflow_fp16", "Softmax", "underflow", "Softmax underflow, wide logits", "fp16",
        "softmax_last", ("x",), (small,), small, init="randn:15", bundle="softmax_sum_to_one_fp16", checks=under)
    add("softmax_underflow_fp32_accumulate_fp16", "Softmax", "underflow", "Softmax underflow, FP32 accumulate",
        "fp16", "softmax_fp32acc", ("x",), (small,), small, ref_op="softmax_last", init="randn:15",
        bundle="softmax_fp32_accumulate_small_fp16", checks=under)

_register()


def bundle_of(name):
    """The example whose compiled kernel ``name`` runs."""
    return EXAMPLES[name].bundle or name


def ops():
    """The op behind each example. Also the CPU reference, evaluated in FP32."""
    import torch
    import torch.nn.functional as F

    # eps is a tensor input: a Python scalar would become an extra kernel argument
    # that ``spyre launch`` cannot supply.
    def rms_manual(x, eps):
        return x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)

    def rms_native(x):
        return F.rms_norm(x, [x.shape[-1]], eps=1e-6)

    def ln_native(x, w, b):
        return F.layer_norm(x, [x.shape[-1]], weight=w, bias=b, eps=EPS)

    def ln_manual(x, w, b, eps):
        c = x - x.mean(dim=-1, keepdim=True)
        var = (c * c).mean(dim=-1, keepdim=True)
        return c * torch.rsqrt(var + eps) * w + b

    def ln_fp32acc(x, w, b, eps):
        xf = x.to(torch.float32)
        c = xf - xf.mean(dim=-1, keepdim=True)
        var = (c * c).mean(dim=-1, keepdim=True)
        y = c * torch.rsqrt(var + eps.to(torch.float32))
        return (y * w.to(torch.float32) + b.to(torch.float32)).to(x.dtype)

    def ln_var(x, w, b, eps):  # the existing tests' flow; Tensor.var is #1688
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps) * w + b

    def softmax_last(x):
        return torch.softmax(x, dim=-1)

    def softmax_fp32acc(x):
        return torch.softmax(x.to(torch.float32), dim=-1).to(x.dtype)

    return dict(
        rms_manual=rms_manual,
        rms_native=rms_native,
        ln_native=ln_native,
        ln_manual=ln_manual,
        ln_fp32acc=ln_fp32acc,
        ln_var=ln_var,
        softmax_last=softmax_last,
        softmax_fp32acc=softmax_fp32acc,
    )


# --------------------------------------------------------------------------
# CLI arguments, inputs and the CPU reference
# --------------------------------------------------------------------------


def tensor_spec(shape, dtype):
    """Format a shape as a ``spyre launch`` tensor argument, e.g. 512x1024@bf16."""
    return "x".join(str(d) for d in shape) + "@" + dtype


def to_torch_dtype(short):
    """The torch dtype for a ``spyre launch`` dtype name: fp16, fp32 or bf16."""
    import torch

    return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[short]


def short_dtype(dtype):
    """The ``spyre launch`` dtype name of a torch dtype."""
    import torch

    return {torch.float16: "fp16", torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]


def torch_dtype(name):
    """The torch dtype of example ``name``."""
    return to_torch_dtype(EXAMPLES[name].dtype)


# --------------------------------------------------------------------------
# launch plan
#
# ``torch.compile`` calls a kernel with tensors of its own choosing, not with the
# function's inputs in order. ``spyre launch`` and ``spyre_cli.launch`` pass tensors
# by position with no shape check, so a launch is only right if it reproduces the
# real call. The plan lists the kernel's arguments in order, each one of:
#   input   a function input, by index (Inductor orders them by first use, so
#           f(x, w, b, eps) can run its kernel on x, eps, w, b, out)
#   const   a single-element tensor the compiled wrapper creates itself, e.g. the
#           epsilon of F.rms_norm (spyre_constant_tensor); spyre-cli must be given it
#   output  the last argument, with the device layout the compiler allocated it with;
#           a kernel compiled for a non-default layout gives wrong results when it
#           writes into a default-layout tensor, which is all ``spyre launch`` creates
# --------------------------------------------------------------------------


def default_plan(name):
    """The plan assumed when none was recorded: function inputs in order, default-layout output."""
    ex = EXAMPLES[name]
    plan = [{"kind": "input", "index": i} for i in range(len(ex.inputs))]
    return plan + [{"kind": "output", "shape": list(ex.output), "dtype": ex.dtype}]


def layout_from_entry(entry):
    """The SpyreTensorLayout recorded in an output entry."""
    from torch_spyre._C import DataFormats, SpyreTensorLayout

    return SpyreTensorLayout(
        entry["device_size"], entry["stride_map"], getattr(DataFormats, entry["device_dtype"])
    )


def build_launch_plan(call_args, dev_inputs):
    """The launch plan of a kernel call, from the tensors the runner received.

    Inputs are matched to ``dev_inputs`` by object identity. Returns None if the call
    is not inputs and single-element constants followed by one output, if an input is
    missing or repeated, or if the output layout cannot be rebuilt from what is recorded.
    """
    plan = []
    for k, a in enumerate(call_args):
        j = next((j for j, d in enumerate(dev_inputs) if d is a), None)
        if j is not None:
            plan.append({"kind": "input", "index": j})
        elif k == len(call_args) - 1:
            layout = a.device_tensor_layout()
            plan.append(
                {
                    "kind": "output",
                    "shape": list(a.shape),
                    "strides": list(a.stride()),
                    "dtype": short_dtype(a.dtype),
                    "device_size": list(layout.device_size),
                    "stride_map": list(layout.stride_map),
                    "device_dtype": layout.device_dtype.name,
                }
            )
            if str(layout_from_entry(plan[-1])) != str(layout):
                return None
        elif a.numel() == 1:
            plan.append(
                {
                    "kind": "const",
                    "value": a.cpu().float().item(),
                    "shape": list(a.shape),
                    "dtype": short_dtype(a.dtype),
                }
            )
        else:
            return None
    if not plan or plan[-1]["kind"] != "output":
        return None
    if sorted(e["index"] for e in plan if e["kind"] == "input") != list(range(len(dev_inputs))):
        return None
    return plan


def read_launch_plan(path, name):
    """The plan recorded when the kernel at ``path`` was built, or the default plan."""
    f = Path(path) / PLAN_FILE
    return json.loads(f.read_text()) if f.exists() else default_plan(name)


def cli_args(name, plan=None):
    """The ``-i``/``-o`` arguments for launching example ``name`` with ``plan``.

    ``spyre launch`` fills every input with ones, so a constant such as an epsilon is
    given the right shape and dtype but not its value, and the output it prints is not
    the CPU reference. It can only allocate default-layout tensors.
    """
    ex = EXAMPLES[name]
    args = []
    for e in plan or default_plan(name):
        if e["kind"] == "input":
            args += ["-i", tensor_spec(ex.inputs[e["index"]], ex.dtype)]
        elif e["kind"] == "const":
            args += ["-i", tensor_spec(e["shape"] or [1], e["dtype"])]
        else:
            args += ["-o", tensor_spec(ex.output, ex.dtype)]
    return args


def launch_tensors(name, plan, inputs):
    """Device tensors for a launch of example ``name`` with ``plan``; returns (args, output).

    The output is zeroed and, when the plan recorded one, allocated with the layout the
    compiler chose.
    """
    import torch
    from torch_spyre import _C
    from torch_spyre._C import spyre_empty_with_layout
    from torch_spyre._inductor.op_spec import spyre_constant_tensor

    device = torch.device("spyre:0")
    args, out = [], None
    for e in plan:
        if e["kind"] == "input":
            args.append(inputs[e["index"]].to("spyre"))
        elif e["kind"] == "const":
            args.append(spyre_constant_tensor(e["value"], device, to_torch_dtype(e["dtype"])))
        elif "device_size" in e:
            out = spyre_empty_with_layout(
                tuple(e["shape"]), tuple(e["strides"]), to_torch_dtype(e["dtype"]), layout_from_entry(e), device=device
            )
            _C.fill_tensor(out, 0.0)
            args.append(out)
        else:
            out = torch.zeros(tuple(e["shape"]), dtype=to_torch_dtype(e["dtype"])).to("spyre")
            args.append(out)
    return args, out


def _draw_x(init, shape, gen, dtype):
    """The "x" input of ``shape`` drawn according to ``init``, one [rows, cols] plane at a time."""
    import torch

    kind, _, arg = init.partition(":")
    if kind == "const":
        return torch.full(shape, float(arg), dtype=dtype)
    scale = torch.logspace(-3, 3, shape[-2]).view(-1, 1) if kind == "mixed" else float(arg or 1.0)
    out = torch.empty(shape, dtype=dtype)
    # the spec-shape softmax input is 1 GiB, so never build it in FP32 whole
    for plane in out.view(-1, *shape[-2:]):
        plane.copy_(torch.randn(shape[-2:], generator=gen) * scale)
    return out


def make_inputs(name, seed=0):
    """CPU inputs for example ``name``, in its dtype."""
    import torch

    ex = EXAMPLES[name]
    dt = torch_dtype(name)
    gen = torch.Generator().manual_seed(seed)
    tensors = []
    for role, shape in zip(ex.roles, ex.inputs):
        if role == "x":
            t = _draw_x(ex.init, shape, gen, dt)
        elif role == "w":
            t = (torch.randn(shape, generator=gen).abs() + 0.5).to(dt)
        elif role == "b":
            t = (torch.randn(shape, generator=gen) * 0.1).to(dt)
        elif role == "eps":
            t = torch.full(shape, EPS, dtype=dt)
        else:
            raise ValueError(f"unknown input role {role!r}")
        tensors.append(t)
    return tensors


def planes(inputs, got):
    """Yield (inputs, output) pieces so FP32 temporaries stay small.

    The only 4-D input is the spec-shape softmax, reduced over its last axis, so
    it is split into [S, S] planes. Everything else is yielded whole.
    """
    if inputs[0].dim() == 4:
        for b in range(inputs[0].shape[0]):
            for h in range(inputs[0].shape[1]):
                yield [inputs[0][b, h]], got[b, h]
    else:
        yield list(inputs), got


def compare(name, inputs, got):
    """Compare ``got`` with the CPU reference (FP32, cast to the example dtype).

    Returns (matches within the example's atol/rtol, max |difference|).
    """
    import torch

    ex = EXAMPLES[name]
    fn = ops()[ex.ref_op or ex.op]
    dt = torch_dtype(name)
    ok, max_diff = True, 0.0
    for xs, g in planes(inputs, got):
        want = fn(*[t.float() for t in xs]).to(dt)
        max_diff = max(max_diff, (g.float() - want.float()).abs().max().item())
        ok = ok and torch.allclose(g, want, atol=ex.atol, rtol=ex.rtol)
    return ok, max_diff


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def _xfail(message):
    print(f"XFAIL: {message}")
    sys.exit(EXIT_XFAIL)


def _first_line(exc):
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else ""


def _compile_one(name, out):
    # Runs in a child process whose TORCHINDUCTOR_CACHE_DIR is already set.
    import torch
    from torch._inductor.utils import run_and_get_code
    from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

    ex = EXAMPLES[name]
    inputs = make_inputs(name)
    dev_inputs = [t.to("spyre") for t in inputs]
    # Record the tensors the compiled kernel is really launched with.
    launches = []
    real_run = SpyreSDSCKernelRunner.run

    def recording_run(self, *args):
        launches.append(args)
        return real_run(self, *args)

    SpyreSDSCKernelRunner.run = recording_run
    try:
        fn = torch.compile(ops()[ex.op], backend="inductor")
        got, sources = run_and_get_code(fn, *dev_inputs)
        got = got.cpu()
    except Exception as exc:  # noqa: BLE001 - any lowering/compile failure is a build failure
        traceback.print_exc()
        if ex.xfail_build:
            _xfail(f"{name}: {ex.xfail_build}")
        sys.exit(f"{name}: cannot build for spyre: {type(exc).__name__}: {_first_line(exc)}")
    finally:
        SpyreSDSCKernelRunner.run = real_run
    ok, max_diff = compare(name, inputs, got)
    if not ok:
        sys.exit(f"{name}: torch.compile result does not match CPU (max|diff|={max_diff})")

    cache = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    kernels = glob.glob(os.path.join(cache, "inductor-spyre", "*", "spyreCodeDir"))
    if len(kernels) != 1:
        _xfail(f"{name}: torch.compile produced {len(kernels)} spyre kernels; spyre launch runs exactly one")

    dest = Path(out) / name
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.dirname(kernels[0]), dest)
    calls = [ln.strip() for src in sources for ln in src.splitlines() if ".run(" in ln]
    print(f"{name}: kernel call: {' | '.join(calls)}")
    plan = build_launch_plan(launches[0], dev_inputs) if launches else None
    if plan is None:
        print(f"{name}: WARNING: cannot record the kernel launch plan; assuming the function's inputs in order")
    else:
        (dest / PLAN_FILE).write_text(json.dumps(plan, indent=1))
        print(f"{name}: launch plan {json.dumps(plan)} (function inputs: {list(ex.roles)})")
    print(f"{name}: wrote {dest}")


def build_bundle(name, out=DEFAULT_OUT, timeout=3600, capture=True):
    """Compile ``name`` into ``out``/<name> in a fresh process.

    Returns (exit code, combined output); exit code 77 is an expected limitation.
    With ``capture=False`` the child writes straight to this process's output and
    the returned output is empty.
    """
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as cache:
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=cache)
        cmd = [sys.executable, __file__, "--one", name, "--out", str(out)]
        try:
            r = subprocess.run(
                cmd, env=env, cwd=HERE, capture_output=capture, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return 1, f"{name}: build timed out after {timeout}s"
    return r.returncode, (r.stdout + r.stderr) if capture else ""


def generate(names, out=DEFAULT_OUT):
    """Compile the kernels behind ``names`` into ``out``; return the bundles that failed."""
    import time

    failed = []
    for bundle in sorted({bundle_of(n) for n in names}):
        rc, _ = build_bundle(bundle, out, capture=False)
        if rc not in (0, EXIT_XFAIL):
            failed.append(bundle)
        time.sleep(2)
    return failed


def main():
    parser = argparse.ArgumentParser(description="Compile Reduction & Normalization workload kernels.")
    parser.add_argument("names", nargs="*", metavar="name")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        _compile_one(args.one, args.out)
        return
    if args.names == ["list"]:
        for name, ex in EXAMPLES.items():
            share = f" (kernel of {ex.bundle})" if ex.bundle else ""
            print(f"{ex.workload} {ex.category:<16} {name}{share}: spyre launch {' '.join(cli_args(name))}")
        return
    unknown = sorted(set(args.names) - set(EXAMPLES))
    if unknown:
        parser.error(f"unknown example(s): {' '.join(unknown)}")
    failed = generate(args.names or sorted(EXAMPLES), args.out)
    if failed:
        sys.exit(f"failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
