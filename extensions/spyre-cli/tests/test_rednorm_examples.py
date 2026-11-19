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


"""Integration tests: Reductions & Normalizations examples on a Spyre device.

Kernels are compiled once per session by rednorm_examples.py on the current
toolchain. Each example is then checked by check.py, which launches it through the
spyre_cli SDK. Every compile and check runs in its own process, since a device fault
poisons the stream for the rest of the process it happens in. Skipped when no device is
present.

A kernel that cannot be built or launched through spyre-cli for a known reason
(Tensor.var missing, scalar kernel arguments, more than one kernel) is reported as
xfail with the reason.

    python3 -m pytest tests/test_rednorm_examples.py -v
    python3 -m pytest tests/test_rednorm_examples.py -v -k softmax      # filter by op
    python3 -m pytest tests/test_rednorm_examples.py -v -k rmsnorm
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from check import EXIT_XFAIL  # noqa: E402
from rednorm_examples import EXAMPLES  # noqa: E402


def _spyre_available():
    probe = "import torch; print(torch.accelerator.current_accelerator())"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "spyre" in result.stdout


pytestmark = pytest.mark.skipif(not _spyre_available(), reason="needs a Spyre device")


def _xfail_reason(output):
    lines = [ln for ln in output.splitlines() if ln.startswith("XFAIL:")]
    return lines[-1][len("XFAIL:") :].strip() if lines else "known spyre-cli limitation"


def _run(cmd, timeout):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=EXAMPLES_DIR,
        check=False,
    )
    return result, result.stdout + result.stderr


@pytest.fixture(scope="session")
def kernels(tmp_path_factory):
    """Compile every rednorm example once; return (output dir, compile output)."""
    out = tmp_path_factory.mktemp("rednorm_kernels")
    _, output = _run([sys.executable, "rednorm_examples.py", "--out", str(out)], 60 * 60)
    return out, output


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, id=f"{EXAMPLES[n].workload}-{EXAMPLES[n].category}-{n}") for n in sorted(EXAMPLES)],
)
def test_rednorm_example(kernels, name):
    out, compile_output = kernels
    ex = EXAMPLES[name]
    bundle = ex.bundle or name
    path = out / bundle
    if not (path / "spyreCodeDir").is_dir():
        if ex.xfail_build:
            pytest.xfail(f"{name}: {ex.xfail_build}")
        pytest.fail(f"rednorm_examples.py did not build {bundle}:\n{compile_output.strip()}")
    result, output = _run([sys.executable, "check.py", name, str(path)], 1200)
    if result.returncode == EXIT_XFAIL:
        pytest.xfail(_xfail_reason(output))
    assert result.returncode == 0, output
