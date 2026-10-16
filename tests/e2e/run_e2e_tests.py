#!/usr/bin/env python3
"""CLI runner for Rocora 4-Tier End-to-End Test Suite.

Usage:
    PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py
    PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 1
    PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 2
    PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 3
    PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 4
"""

import argparse
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]

TIER_FILES = {
    1: ROOT / "tests/e2e/test_tier1_features.py",
    2: ROOT / "tests/e2e/test_tier2_boundaries.py",
    3: ROOT / "tests/e2e/test_tier3_combinations.py",
    4: ROOT / "tests/e2e/test_tier4_workloads.py",
}

TIER_DESCRIPTIONS = {
    1: "Tier 1: Feature Coverage (≥60 tests across 12 features)",
    2: "Tier 2: Boundary & Corner Cases (≥60 tests across 12 features)",
    3: "Tier 3: Pairwise Combinations (≥15 tests across interactions)",
    4: "Tier 4: Realistic Workloads (≥6 bounded smoke workloads)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rocora End-to-End Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tier", "-t",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Select which test tier to run (default: all)",
    )
    parser.add_argument(
        "-k", "--expression",
        help="Pytest expression filter (-k)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose pytest output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet pytest output",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure environment variables
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    print("=" * 80)
    print("  ROCORA END-TO-END TEST SUITE RUNNER")
    print("=" * 80)
    print(f"Working Directory : {ROOT}")
    print(f"JAX Platforms     : {os.environ.get('JAX_PLATFORMS')}")
    print(f"Target Selection  : Tier {args.tier.upper()}")
    print("-" * 80)

    if args.tier == "all":
        selected_tiers = [1, 2, 3, 4]
    else:
        selected_tiers = [int(args.tier)]

    test_targets = []
    for t in selected_tiers:
        target_path = TIER_FILES[t]
        if not target_path.exists():
            print(f"ERROR: Test file {target_path} not found!", file=sys.stderr)
            return 1
        print(f"  • {TIER_DESCRIPTIONS[t]}")
        print(f"    File: {target_path.relative_to(ROOT)}")
        test_targets.append(str(target_path))

    print("-" * 80)

    pytest_args = list(test_targets)
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")
    else:
        pytest_args.append("-q")

    if args.expression:
        pytest_args.extend(["-k", args.expression])

    start_time = time.monotonic()
    result = pytest.main(pytest_args)
    elapsed = time.monotonic() - start_time

    print("-" * 80)
    status_str = "PASSED" if result == 0 else f"FAILED (exit code {result})"
    print(f"Result  : {status_str}")
    print(f"Duration: {elapsed:.2f} seconds")
    print("=" * 80)

    return int(result)


if __name__ == "__main__":
    sys.exit(main())
