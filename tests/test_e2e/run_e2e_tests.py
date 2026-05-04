# SPDX-FileCopyrightText: (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
E2E Test Runner

Provides command-line interface for running E2E tests manually.
Usage:
    python run_e2e_tests.py --test e2e_01
    python run_e2e_tests.py --test all
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Test script mapping
TEST_SCRIPTS = {
    "e2e_01": "test_e2e_01_single_frame.py",
    "e2e_02": "test_e2e_02_multi_frame.py",
    "e2e_03": "test_e2e_03_dynamic_objects.py",
    "e2e_04": "test_e2e_04_camera_specs.py",
    "e2e_05": "test_e2e_05_coordinate_transform.py",
    "e2e_06": "test_e2e_06_error_handling.py",
    "e2e_07": "test_e2e_07_performance.py",
}


def run_test(test_name: str, verbose: bool = False) -> int:
    """
    Run a single E2E test.

    Args:
        test_name: Test identifier (e.g., "e2e_01")
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if test_name not in TEST_SCRIPTS:
        print(f"ERROR: Unknown test '{test_name}'")
        print(f"Available tests: {list(TEST_SCRIPTS.keys())}")
        return 1

    test_script = TEST_SCRIPTS[test_name]
    test_path = Path(__file__).parent / test_script

    if not test_path.exists():
        print(f"ERROR: Test script not found: {test_path}")
        return 1

    # Build pytest command
    pytest_args = ["pytest", str(test_path)]
    if verbose:
        pytest_args.extend(["-v", "-s"])

    print(f"Running test: {test_name}")
    print(f"Command: {' '.join(pytest_args)}")
    print("-" * 60)

    # Run pytest
    result = subprocess.run(pytest_args, cwd=str(Path(__file__).parent))

    print("-" * 60)
    if result.returncode == 0:
        print(f"Test {test_name}: PASSED")
    else:
        print(f"Test {test_name}: FAILED")

    return result.returncode


def run_all_tests(verbose: bool = False) -> int:
    """
    Run all E2E tests sequentially.

    Args:
        verbose: Enable verbose output

    Returns:
        0 if all tests pass, non-zero otherwise
    """
    print("=" * 60)
    print("Running all E2E tests")
    print("=" * 60)

    failed_tests = []

    for test_name in TEST_SCRIPTS.keys():
        exit_code = run_test(test_name, verbose)
        if exit_code != 0:
            failed_tests.append(test_name)

    print("=" * 60)
    print("E2E Test Summary")
    print("=" * 60)

    total = len(TEST_SCRIPTS)
    passed = total - len(failed_tests)

    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
        return 1

    print("All tests passed!")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="E2E Test Runner")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        help="Test to run (e.g., 'e2e_01', 'all')"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.test == "all":
        return run_all_tests(args.verbose)
    else:
        return run_test(args.test, args.verbose)


if __name__ == "__main__":
    sys.exit(main())