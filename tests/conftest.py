"""
Test configuration and fixtures for nanoBragg PyTorch tests.

This file contains pytest configuration, fixtures, and environment setup
that is shared across all test modules.
"""

import os
import sys
from pathlib import Path
import pytest
import subprocess

# Set environment variable to prevent MKL library conflicts with PyTorch
# This must be set before importing torch in any test module
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add src to path for all tests, and to PYTHONPATH (absolute) for the tests that run
# `python -m nanobrag_torch` in a subprocess, so they work without a pip install and
# from any working directory.
_SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, _SRC)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_SRC] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != _SRC]
)


# ============================================================================
# Per-test isolation of process-global state
# ============================================================================

@pytest.fixture(autouse=True)
def _restore_process_globals():
    """Undo what a test changes in os.environ, the working directory and torch globals.

    Several tests set environment variables, chdir, or call torch.set_default_dtype
    without restoring them, which made later tests fail only in a full run (for
    example a deleted NB_C_BIN, or float64 left as the default dtype).
    """
    import torch

    env = dict(os.environ)
    cwd = os.getcwd()
    default_dtype = torch.get_default_dtype()
    default_device = torch.get_default_device()
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    os.environ.clear()
    os.environ.update(env)
    os.chdir(cwd)
    torch.set_default_dtype(default_dtype)
    torch.set_default_device(default_device)
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)


# ============================================================================
# Phase K Task K1: Session Infrastructure Gate
# ============================================================================

def _resolve_c_binary():
    """
    Resolve C binary path using documented precedence order.

    Returns:
        Path object if resolved, None otherwise
    """
    # Use absolute path resolution relative to repo root
    repo_root = Path(__file__).parent.parent

    nb_c_bin = os.environ.get('NB_C_BIN')
    if nb_c_bin:
        path = Path(nb_c_bin)
        if path.exists():
            return path
        else:
            # NB_C_BIN set but invalid - this is a user error, fail explicitly
            return None

    # Fallback 1: instrumented binary (recommended)
    fallback1 = repo_root / 'golden_suite_generator/nanoBragg'
    if fallback1.exists():
        return fallback1

    # Fallback 2: frozen reference binary
    fallback2 = repo_root / 'nanoBragg'
    if fallback2.exists():
        return fallback2

    return None


def _check_c_binary_executable(binary_path):
    """
    Verify C binary can execute -help command.

    Args:
        binary_path: Path object pointing to C binary

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    if not os.access(binary_path, os.X_OK):
        return False, f"Binary not executable (missing execute permission): {binary_path}"

    try:
        result = subprocess.run(
            [str(binary_path), '-help'],
            capture_output=True,
            timeout=10,
            text=True
        )
        # nanoBragg exits with non-zero but produces help text on stdout
        if 'usage: nanobragg' in result.stdout.lower():
            return True, None
        else:
            return False, f"Binary failed to produce help text (stdout: {result.stdout[:200]}, stderr: {result.stderr[:200]})"
    except subprocess.TimeoutExpired:
        return False, f"Binary -help command timed out (>10s): {binary_path}"
    except Exception as e:
        return False, f"Binary execution failed: {binary_path} ({type(e).__name__}: {e})"


@pytest.fixture(scope="session", autouse=True)
def session_infrastructure_gate():
    """
    Validate infrastructure prerequisites at pytest session start.

    See: reports/2026-01-test-suite-refresh/phase_j/20251015T180301Z/analysis/session_fixture_design.md
    """
    # Allow bypass for debugging (not recommended for CI)
    if os.environ.get('NB_SKIP_INFRA_GATE') == '1':
        import warnings
        warnings.warn(
            "Infrastructure gate bypassed (NB_SKIP_INFRA_GATE=1). "
            "This should only be used for debugging.",
            UserWarning
        )
        return

    errors = []

    # Check 1: C binary resolution
    c_binary_path = _resolve_c_binary()
    if c_binary_path is None:
        nb_c_bin = os.environ.get('NB_C_BIN')
        if nb_c_bin:
            errors.append(
                f"NB_C_BIN set but invalid: {nb_c_bin}\n"
                "  Verify path exists and is executable"
            )
        else:
            errors.append(
                "C binary not found. Expected one of:\n"
                "  1. NB_C_BIN environment variable pointing to binary\n"
                "  2. ./golden_suite_generator/nanoBragg (instrumented binary)\n"
                "  3. ./nanoBragg (frozen reference)\n"
                "  Rebuild with: make -C golden_suite_generator"
            )

    # Check 2: C binary executability
    if c_binary_path is not None:
        success, error_msg = _check_c_binary_executable(c_binary_path)
        if not success:
            errors.append(error_msg)

    # Fail fast if any checks failed
    if errors:
        pytest.fail(
            "\n" + "=" * 80 + "\n" +
            "Infrastructure Gate Check FAILED\n" +
            "=" * 80 + "\n" +
            "\n".join(f"  - {err}" for err in errors) + "\n\n" +
            "For details, see:\n"
            "  - reports/2026-01-test-suite-refresh/phase_h/20251015T171757Z/\n"
            "  - docs/development/testing_strategy.md S1.5\n"
            "  - plans/active/test-suite-triage-phase-h.md\n"
            "=" * 80
        )