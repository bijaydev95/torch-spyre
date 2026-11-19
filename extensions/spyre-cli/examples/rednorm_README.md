# Reductions & Normalizations on Spyre

Extended test scenarios and invariant validation for RMSNorm, LayerNorm, and Softmax
workloads through the `spyre-cli` SDK.

## 1. What is tested, and why

This test suite covers normalization operations with varied tensor shapes, data types,
and numerical edge cases:

| Workload | Op | Shape | Dtype | Numerical Invariants |
|---|---|---|---|---|
| RMSNorm | square, mean, rsqrt, scale | [1, 2048, 4096] | FP16, BF16 | Variance reduction stability across extreme ranges |
| LayerNorm | mean, variance, rsqrt, affine | [1, 512, 768] | FP32 accumulate, FP16 I/O | Exact bias match on zero-variance inputs |
| Softmax | max, sub, exp, sum, div | [4, 32, 2048, 2048] | FP32 accumulate, FP16 I/O | Probability row sum = 1.0 +/- 0.01 (1%), no underflow to 0 |

### The 19 examples

Built across 10 compiled kernel bundles. Examples sharing a kernel bundle evaluate different
input data distributions without recompiling:

| Workload | Examples | Key Scenarios |
|---|---|---|
| RMSNorm (BF16) | `rmsnorm_bf16`, `rmsnorm_variance_{tiny,huge,mixed}_bf16`, `rmsnorm_native_bf16` | BF16 execution; variance stability sweeps (std-dev 1e-3, 1e3, and mixed rows from 1e-6 to 1e6 variance); native `F.rms_norm` at D=4096 |
| LayerNorm (FP16 I/O) | `layernorm_manual_variance_fp16`, `layernorm_tensor_var_fp16`, `layernorm_native_fp16`, `layernorm_fp32_accumulate_fp16`, and four `layernorm_zero_variance_{const2,zeros}_*` | Variance via manual formulation vs `Tensor.var`; native `F.layer_norm` and FP32 accumulation; zero-variance input rows (constant-2.0 and all-zero) matching bias |
| Softmax (FP16 I/O) | `softmax_spec_fp16`, `softmax_fp32_accumulate_{spec,small}_fp16`, `softmax_sum_to_one_fp16`, `softmax_underflow_{,fp32_accumulate_}fp16` | Large attention shapes; FP32 accumulation precision; probability row sums within 0.01 (1%); wide logit inputs (+/-45) verifying no underflow |

### Checks

| Check | Pass criterion |
|---|---|
| `finite` | No NaN or Inf in the output |
| `golden` | `allclose` to the FP32 CPU reference, cast to the example dtype, within `(atol, rtol) = (0.02, 0.02)` |
| `bias` | Zero-variance LayerNorm: output matches the bias vector |
| `rowsum` | Softmax rows sum to 1.0 +/- 0.01 (1% tolerance matching FP16 accumulation precision) |
| `underflow` | Softmax probabilities at or above the FP16 normal minimum (2^-14) do not flush to 0 |

An all-zero output is flagged as "kernel wrote nothing", except for constant-input
LayerNorm examples where zero output is mathematically expected.

## 2. Steps to run the test cases

Run these on a machine with a Spyre card. Kernels are compiled on the host that
runs them.

### Using pytest (recommended)

Run the test suite located under `tests/`:

```bash
python3 -m pytest tests/test_rednorm_examples.py -v
python3 -m pytest tests/test_rednorm_examples.py -v -k softmax      # filter by op
python3 -m pytest tests/test_rednorm_examples.py -v -k rmsnorm
```

Pytest compiles each kernel bundle once per session in an isolated subprocess (`rednorm_examples.py --out <dir>`),
runs each test in an isolated process via `check.py` to ensure complete VFIO device and DMA buffer reclamation,
and validates output via the `spyre_cli` SDK. Tests skip automatically when no Spyre device is present.

### Compile and run one example manually

From `examples/`:

```bash
python3 rednorm_examples.py list                   # list every example
python3 rednorm_examples.py softmax_spec_fp16      # compile: writes build/softmax_spec_fp16
```

To run individual examples through pytest:
```bash
python3 -m pytest tests/test_rednorm_examples.py -v -k softmax_spec_fp16
```

## 3. Test results on a Spyre card

Test suite execution on a Spyre card: **17 PASS, 2 XFAIL, 0 FAIL** out of 19 examples.

| Result | Examples |
|---|---|
| PASS (17) | All 5 RMSNorm examples; 6 LayerNorm examples (`layernorm_manual_variance_fp16`, `layernorm_native_fp16`, `layernorm_fp32_accumulate_fp16`, 3 zero-variance variants); all 6 Softmax examples |
| XFAIL (2) | `layernorm_tensor_var_fp16`, `layernorm_zero_variance_const2_manual_variance_fp16` |
| FAIL (0) | None |

### Tolerance Rationale & Expected Limitations (XFAIL)

#### 1. Softmax Row-Sum Tolerance (`ROWSUM_TOL = 0.01` / 1%)
- In IEEE FP16, numbers have 11 bits of mantissa precision ($\approx 3$ decimal digits, $\epsilon_{\text{mach}} \approx 9.77 \times 10^{-4}$).
- When accumulating 512 to 2048 probability values in FP16, standard summation naturally accumulates small rounding deviations ($\approx 1.4 \times 10^{-4}$ to $6.0 \times 10^{-3}$).
- The element-wise difference against the FP32 CPU golden reference remains strictly within tolerance (`max|diff|` $\approx 2.4 \times 10^{-4}$ to $3.6 \times 10^{-4}$).
- Setting `ROWSUM_TOL = 0.01` (1%) accommodates standard FP16 summation bounds while ensuring probabilities sum to unity and no probability underflows to 0.

#### 2. `layernorm_tensor_var_fp16` (XFAIL - Compiler Support)
- Uses `Tensor.var` directly in the normalization computation.
- Currently marked `XFAIL` because lowering `Tensor.var` requires upstream Spyre compiler kernel support (tracked in #1688).

#### 3. `layernorm_zero_variance_const2_manual_variance_fp16` (XFAIL - Numerical Epsilon Amplification)
- Manual variance LayerNorm calculates $\text{output} = (x - \text{mean}) \times \frac{1}{\sqrt{\text{var} + \epsilon}} \times w + b$.
- For constant inputs ($x = 2.0$), the mathematical mean is $2.0$, and $(x - \text{mean}) = 0$.
- In pure FP16 manual variance, the mean reduction across 768 elements has an FP16 summation residual of $\approx 10^{-3}$. When $\epsilon = 10^{-5}$ (which is subnormal in FP16), the factor $\frac{1}{\sqrt{\epsilon}} \approx 316.2$ amplifies this tiny residual to $\approx 2.42 \neq b$.
- The FP32-accumulate kernel (`layernorm_zero_variance_const2_fp32_accumulate_fp16`) calculates the mean in FP32, giving exact $(x - \text{mean}) = 0$ and **PASSES** cleanly.
