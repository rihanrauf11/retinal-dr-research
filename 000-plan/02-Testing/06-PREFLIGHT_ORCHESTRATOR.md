# Phase 6: Pre-Flight Orchestrator

**Priority:** LOW (Nice to have)
**Time Estimate:** 4-5 hours
**Dependencies:** Phases 1-5 complete
**Deliverable:** `scripts/preflight_check.py`

---

## Objective

Create master test runner that orchestrates all pre-training validation tests in correct order with comprehensive reporting. One command to verify training readiness.

---

## Rationale

**Without orchestrator:**
- Must manually run 5+ test scripts
- Easy to forget a test
- No unified reporting
- Hard to share results with team

**With orchestrator:**
- Single command: `python scripts/preflight_check.py --config myconfig.yaml`
- Runs all tests in dependency order
- Generates HTML report
- Clear pass/fail status with recommendations

---

## Features

### 1. Test Orchestration
- Run tests in correct order (sanity → memory → overfitting → data loading)
- Skip optional tests if requested
- Stop on critical failures
- Continue through non-critical warnings

### 2. Comprehensive Reporting
- HTML report with all test results
- Summary dashboard
- Detailed logs per test
- Recommendations based on results

### 3. CI/CD Integration
- Exit code 0 if all critical tests pass
- Exit code 1 if any critical test fails
- JSON output for programmatic access

---

## Implementation

```python
#!/usr/bin/env python3
"""
Pre-flight check orchestrator - runs all validation tests.

Usage:
    # Run all tests
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml

    # Run only critical tests
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml --critical-only

    # Generate HTML report
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml --html report.html

    # JSON output
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml --json results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import load_config


class PreFlightChecker:
    """Orchestrate all pre-training validation tests."""

    def __init__(self, config_path: str, critical_only: bool = False):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.critical_only = critical_only
        self.results = {}
        self.start_time = time.time()

    def run_test(self, name: str, script: str, args: List[str], critical: bool = True) -> Dict:
        """Run a single test script."""
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")

        start = time.time()

        try:
            # Run test script
            cmd = [sys.executable, f"scripts/{script}"] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            elapsed = time.time() - start
            success = (result.returncode == 0)

            return {
                'name': name,
                'script': script,
                'success': success,
                'critical': critical,
                'elapsed': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return {
                'name': name,
                'script': script,
                'success': False,
                'critical': critical,
                'elapsed': elapsed,
                'stdout': '',
                'stderr': 'Test timed out after 10 minutes',
                'returncode': -1,
            }

        except Exception as e:
            elapsed = time.time() - start
            return {
                'name': name,
                'script': script,
                'success': False,
                'critical': critical,
                'elapsed': elapsed,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
            }

    def run_all_tests(self) -> Dict:
        """Run complete test suite."""
        print("=" * 60)
        print("PRE-FLIGHT CHECK")
        print("=" * 60)
        print(f"Config: {self.config_path}")
        print(f"Mode: {'Critical only' if self.critical_only else 'Full suite'}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        tests = [
            # Critical tests (must pass)
            {
                'name': 'Sanity Tests',
                'script': 'test_sanity.py',
                'args': ['--config', self.config_path],
                'critical': True,
            },
            {
                'name': 'GPU Memory Profile',
                'script': 'test_gpu_memory.py',
                'args': ['--config', self.config_path],
                'critical': True,
            },
            {
                'name': 'Overfitting Test',
                'script': 'test_overfitting.py',
                'args': ['--config', self.config_path],
                'critical': True,
            },
            # Optional tests (nice to have)
            {
                'name': 'Data Loading Performance',
                'script': 'test_data_loading.py',
                'args': ['--config', self.config_path],
                'critical': False,
            },
        ]

        # Filter tests if critical_only
        if self.critical_only:
            tests = [t for t in tests if t['critical']]

        # Run tests
        results = []
        critical_failure = False

        for test in tests:
            result = self.run_test(
                name=test['name'],
                script=test['script'],
                args=test['args'],
                critical=test['critical']
            )
            results.append(result)

            # Print result
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"\n{status}: {result['name']} ({result['elapsed']:.1f}s)")

            # Stop on critical failure
            if not result['success'] and result['critical']:
                print(f"\n⚠️  Critical test failed, stopping here")
                critical_failure = True
                break

        # Summary
        elapsed_total = time.time() - self.start_time
        passed = sum(1 for r in results if r['success'])
        failed = len(results) - passed

        summary = {
            'config': self.config_path,
            'start_time': datetime.now().isoformat(),
            'elapsed_total': elapsed_total,
            'tests_run': len(results),
            'tests_passed': passed,
            'tests_failed': failed,
            'critical_failure': critical_failure,
            'results': results,
        }

        return summary

    def print_summary(self, summary: Dict):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\nTests run: {summary['tests_run']}")
        print(f"Passed: {summary['tests_passed']}")
        print(f"Failed: {summary['tests_failed']}")
        print(f"Total time: {summary['elapsed_total']:.1f}s")

        # Detailed results
        print("\nDetailed Results:")
        for result in summary['results']:
            status = "✅" if result['success'] else "❌"
            critical = " [CRITICAL]" if result['critical'] else ""
            print(f"  {status} {result['name']}{critical} ({result['elapsed']:.1f}s)")

        # Overall status
        print("\n" + "=" * 60)
        if summary['critical_failure']:
            print("❌ PRE-FLIGHT CHECK FAILED")
            print("   Critical tests did not pass. Fix issues before training.")
        elif summary['tests_failed'] == 0:
            print("✅ PRE-FLIGHT CHECK PASSED")
            print("   All tests passed. Ready to train!")
        else:
            print("⚠️  PRE-FLIGHT CHECK PASSED WITH WARNINGS")
            print("   Critical tests passed, but some optional tests failed.")
            print("   You can proceed with training, but check warnings.")

        print("=" * 60)

    def generate_html_report(self, summary: Dict, output_path: str):
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pre-Flight Check Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .test {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .pass {{ border-left: 5px solid #4CAF50; }}
        .fail {{ border-left: 5px solid #f44336; }}
        .critical {{ background: #fff3cd; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        .status {{ font-weight: bold; }}
        .status.pass {{ color: #4CAF50; }}
        .status.fail {{ color: #f44336; }}
    </style>
</head>
<body>
    <h1>Pre-Flight Check Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Config:</strong> {summary['config']}</p>
        <p><strong>Started:</strong> {summary['start_time']}</p>
        <p><strong>Duration:</strong> {summary['elapsed_total']:.1f}s</p>
        <p><strong>Tests Run:</strong> {summary['tests_run']}</p>
        <p><strong>Passed:</strong> {summary['tests_passed']}</p>
        <p><strong>Failed:</strong> {summary['tests_failed']}</p>
        <p class="status {'pass' if not summary['critical_failure'] and summary['tests_failed'] == 0 else 'fail'}">
            <strong>Overall Status:</strong> {'PASSED' if not summary['critical_failure'] and summary['tests_failed'] == 0 else 'FAILED'}
        </p>
    </div>

    <h2>Test Results</h2>
"""

        for result in summary['results']:
            status_class = 'pass' if result['success'] else 'fail'
            critical_class = 'critical' if result['critical'] else ''

            html += f"""
    <div class="test {status_class} {critical_class}">
        <h3>{result['name']}</h3>
        <p><span class="status {status_class}">{'✅ PASSED' if result['success'] else '❌ FAILED'}</span>
           {' [CRITICAL]' if result['critical'] else ''}
           <span style="float: right;">Duration: {result['elapsed']:.1f}s</span></p>

        <details>
            <summary>Output</summary>
            <pre>{result['stdout']}</pre>
        </details>

        {f'<details><summary>Errors</summary><pre>{result["stderr"]}</pre></details>' if result['stderr'] else ''}
    </div>
"""

        html += """
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"\n📄 HTML report saved to: {output_path}")

    def save_json(self, summary: Dict, output_path: str):
        """Save results as JSON."""
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 JSON results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run pre-flight checks before training")
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--critical-only', action='store_true', help='Run only critical tests')
    parser.add_argument('--html', type=str, help='Generate HTML report')
    parser.add_argument('--json', type=str, help='Save JSON results')

    args = parser.parse_args()

    # Run tests
    checker = PreFlightChecker(args.config, critical_only=args.critical_only)
    summary = checker.run_all_tests()

    # Print summary
    checker.print_summary(summary)

    # Generate reports
    if args.html:
        checker.generate_html_report(summary, args.html)

    if args.json:
        checker.save_json(summary, args.json)

    # Exit with appropriate code
    if summary['critical_failure']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
```

---

## Usage

```bash
# Run all tests
python scripts/preflight_check.py --config configs/retfound_lora_config.yaml

# Critical tests only (faster)
python scripts/preflight_check.py --config configs/retfound_lora_config.yaml --critical-only

# Generate HTML report
python scripts/preflight_check.py \
    --config configs/retfound_lora_config.yaml \
    --html preflight_report.html

# Save JSON for CI/CD
python scripts/preflight_check.py \
    --config configs/retfound_lora_config.yaml \
    --json preflight_results.json
```

---

## Expected Output

```
============================================================
PRE-FLIGHT CHECK
============================================================
Config: configs/retfound_lora_config.yaml
Mode: Full suite
Started: 2025-11-04 14:30:00

============================================================
Running: Sanity Tests
============================================================
[... sanity test output ...]

✅ PASS: Sanity Tests (45.2s)

============================================================
Running: GPU Memory Profile
============================================================
[... memory test output ...]

✅ PASS: GPU Memory Profile (32.1s)

============================================================
Running: Overfitting Test
============================================================
[... overfitting test output ...]

✅ PASS: Overfitting Test (127.5s)

============================================================
Running: Data Loading Performance
============================================================
[... data loading test output ...]

✅ PASS: Data Loading Performance (58.3s)

============================================================
SUMMARY
============================================================

Tests run: 4
Passed: 4
Failed: 0
Total time: 263.1s

Detailed Results:
  ✅ Sanity Tests [CRITICAL] (45.2s)
  ✅ GPU Memory Profile [CRITICAL] (32.1s)
  ✅ Overfitting Test [CRITICAL] (127.5s)
  ✅ Data Loading Performance (58.3s)

============================================================
✅ PRE-FLIGHT CHECK PASSED
   All tests passed. Ready to train!
============================================================
```

---

## Integration with Workflow

### Before Every Training Run
```bash
# 1. Run pre-flight check
python scripts/preflight_check.py --config configs/my_experiment.yaml

# 2. If pass, start training
python scripts/train_retfound_lora.py --config configs/my_experiment.yaml
```

### CI/CD Integration
```yaml
# .github/workflows/test.yml
- name: Pre-flight check
  run: |
    python scripts/preflight_check.py \
      --config configs/retfound_lora_config.yaml \
      --critical-only \
      --json preflight_results.json

- name: Upload report
  uses: actions/upload-artifact@v2
  with:
    name: preflight-report
    path: preflight_results.json
```

---

## Next Steps

After Phase 6:
1. **Complete implementation plan done!**
2. **Ready to start training with confidence**
3. **Document your results and iterate**

---

## Success Criteria

- ✅ Script orchestrates all test phases
- ✅ Generates comprehensive HTML report
- ✅ Proper exit codes for CI/CD
- ✅ Clear pass/fail status
- ✅ Runs in < 5 minutes (critical tests) or < 10 minutes (full suite)

---

**Implementation complete!** You now have a comprehensive pre-training validation system.
