#!/usr/bin/env python3
"""
Pre-flight validation orchestrator - runs all tests before training.

This script orchestrates all validation tests in dependency order:
1. Sanity tests (critical)
2. GPU memory tests (critical)
3. Overfitting tests (critical)
4. Data loading tests (optional)

Usage:
    # Run all tests
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml

    # Run only critical tests (faster)
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml --critical-only

    # Integration with training
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml && \
        python scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config import Config, validate_config


class PreFlightChecker:
    """Orchestrate all pre-training validation tests."""

    def __init__(self, config_path: str, critical_only: bool = False, verbose: bool = True):
        """
        Initialize pre-flight checker.

        Args:
            config_path: Path to YAML config file
            critical_only: Run only critical tests (faster)
            verbose: Print detailed information
        """
        # Validate config file exists
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            print(f"❌ Error: Config file not found: {config_path}")
            sys.exit(1)

        self.config_path = str(config_path_obj)
        self.critical_only = critical_only
        self.verbose = verbose

        # Load and validate config
        self.config = Config.from_yaml(self.config_path)
        validate_config(self.config)

        # Test registry with dependency order
        self.tests = self._define_tests()

        # Results storage
        self.results = {}

    def _define_tests(self) -> List[Dict]:
        """
        Define all tests with metadata.

        Returns:
            List of test configurations with name, script, criticality, timeout, args
        """
        all_tests = [
            {
                'name': 'Sanity Tests',
                'script': 'test_sanity.py',
                'critical': True,
                'timeout': 120,
                'args': ['--config', self.config_path, '--verbose']
            },
            {
                'name': 'GPU Memory Tests',
                'script': 'test_gpu_memory.py',
                'critical': True,
                'timeout': 300,
                'args': ['--config', self.config_path, '--verbose']
            },
            {
                'name': 'Overfitting Tests',
                'script': 'test_overfitting.py',
                'critical': True,
                'timeout': 600,
                'args': ['--config', self.config_path, '--num-samples', '10', '--verbose']
            },
            {
                'name': 'Data Loading Tests',
                'script': 'test_data_loading.py',
                'critical': False,
                'timeout': 300,
                'args': ['--config', self.config_path, '--num-workers', '0,2,4']
            },
        ]

        # Filter based on critical_only flag
        if self.critical_only:
            return [t for t in all_tests if t['critical']]
        return all_tests

    def run_test(self, test_config: Dict) -> Dict:
        """
        Run a single test script.

        Args:
            test_config: Test configuration dictionary

        Returns:
            Dictionary with test results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running: {test_config['name']}")
            print(f"{'='*60}")

        # Build command
        script_path = Path(__file__).parent / test_config['script']
        cmd = [sys.executable, str(script_path)] + test_config['args']

        # Run test
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=test_config['timeout']
            )

            duration = (datetime.now() - start_time).total_seconds()

            # Determine status
            success = (result.returncode == 0)
            status = 'PASS' if success else 'FAIL'

            if self.verbose:
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
                print(f"\n{status}: {test_config['name']} ({duration:.1f}s)")

            return {
                'name': test_config['name'],
                'status': status,
                'critical': test_config['critical'],
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }

        except subprocess.TimeoutExpired:
            duration = test_config['timeout']
            if self.verbose:
                print(f"\n❌ TIMEOUT: {test_config['name']} exceeded {duration}s")

            return {
                'name': test_config['name'],
                'status': 'TIMEOUT',
                'critical': test_config['critical'],
                'duration': duration,
                'returncode': -1,
                'stdout': '',
                'stderr': f'Test exceeded timeout of {duration}s',
            }

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            if self.verbose:
                print(f"\n❌ ERROR: {test_config['name']}: {e}")

            return {
                'name': test_config['name'],
                'status': 'ERROR',
                'critical': test_config['critical'],
                'duration': duration,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
            }

    def run_all_tests(self) -> bool:
        """
        Run all tests in sequence.

        Stops on first critical test failure (fail-fast behavior).

        Returns:
            True if all critical tests passed
        """
        if self.verbose:
            mode = "CRITICAL ONLY" if self.critical_only else "ALL TESTS"
            print(f"\n{'='*60}")
            print(f"PRE-FLIGHT VALIDATION: {mode}")
            print(f"{'='*60}")
            print(f"Config: {self.config_path}")
            print(f"Model: {self.config.model.model_variant}")
            print(f"Tests to run: {len(self.tests)}")

        # Run tests in order
        for test_config in self.tests:
            result = self.run_test(test_config)
            self.results[test_config['name']] = result

            # Stop on critical failure (fail-fast)
            if result['critical'] and result['status'] != 'PASS':
                if self.verbose:
                    print(f"\n⚠️  Critical test failed, stopping execution")
                break

        # Check if all critical tests passed
        critical_passed = all(
            r['status'] == 'PASS'
            for r in self.results.values()
            if r['critical']
        )

        return critical_passed

    def print_summary(self):
        """Print test results summary to console."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}\n")

        total_duration = sum(r['duration'] for r in self.results.values())

        for name, result in self.results.items():
            icon = '✅' if result['status'] == 'PASS' else '❌'
            critical_tag = '[CRITICAL]' if result['critical'] else '[OPTIONAL]'
            print(f"{icon} {critical_tag} {name}: {result['status']} ({result['duration']:.1f}s)")

        print(f"\nTotal duration: {total_duration:.1f}s")

        # Overall status
        critical_passed = all(
            r['status'] == 'PASS'
            for r in self.results.values()
            if r['critical']
        )

        print(f"\n{'='*60}")
        if critical_passed:
            print("✅ PRE-FLIGHT CHECK PASSED")
            print("   → All critical tests passed")
            print("   → Ready to start training")
        else:
            print("❌ PRE-FLIGHT CHECK FAILED")
            print("   → Critical tests failed")
            print("   → Fix issues before training")
        print(f"{'='*60}\n")

    def generate_html_report(self):
        """Generate HTML report with test results and outputs."""
        output_dir = Path(self.config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "preflight_report.html"

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Pre-Flight Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; border-left: 5px solid #2196F3; }}
        .test {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: white; }}
        .pass {{ border-left: 5px solid #4CAF50; }}
        .fail {{ border-left: 5px solid #f44336; }}
        .timeout {{ border-left: 5px solid #ff9800; }}
        .error {{ border-left: 5px solid #f44336; }}
        .output {{ background: #f5f5f5; padding: 10px; margin: 10px 0; overflow-x: auto; border-radius: 3px; }}
        pre {{ margin: 0; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 12px; }}
        .critical {{ color: #f44336; font-weight: bold; }}
        .optional {{ color: #666; font-weight: normal; }}
        summary {{ cursor: pointer; user-select: none; }}
        summary:hover {{ text-decoration: underline; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; font-size: 18px; }}
        .metadata {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Pre-Flight Validation Report</h1>
        <p class="metadata"><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p class="metadata"><strong>Config:</strong> {self.config_path}</p>
        <p class="metadata"><strong>Model Variant:</strong> {self.config.model.model_variant}</p>
    </div>
"""

        for name, result in self.results.items():
            status_class = result['status'].lower()
            critical_label = '<span class="critical">[CRITICAL]</span>' if result['critical'] else '<span class="optional">[OPTIONAL]</span>'

            html += f"""
    <div class="test {status_class}">
        <h2>{name} {critical_label}</h2>
        <p><strong>Status:</strong> {result['status']}</p>
        <p><strong>Duration:</strong> {result['duration']:.1f}s</p>
        <p><strong>Return Code:</strong> {result['returncode']}</p>

        <details>
            <summary>Output</summary>
            <div class="output">
                <pre>{result['stdout']}</pre>
            </div>
        </details>

        {f'<details><summary>Errors</summary><div class="output"><pre>{result["stderr"]}</pre></div></details>' if result['stderr'] else ''}
    </div>
"""

        html += """
</body>
</html>
"""

        report_path.write_text(html)

        if self.verbose:
            print(f"📊 HTML report saved to: {report_path}")

    def save_json(self):
        """Save results as JSON for CI/CD integration."""
        output_dir = Path(self.config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "preflight_results.json"

        # Prepare JSON output
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config_path,
            'model_variant': self.config.model.model_variant,
            'critical_only': self.critical_only,
            'tests': self.results,
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results.values() if r['status'] == 'PASS'),
                'failed': sum(1 for r in self.results.values() if r['status'] in ['FAIL', 'TIMEOUT', 'ERROR']),
                'critical_passed': all(r['status'] == 'PASS' for r in self.results.values() if r['critical']),
                'total_duration': sum(r['duration'] for r in self.results.values()),
            }
        }

        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        if self.verbose:
            print(f"💾 JSON results saved to: {json_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pre-flight validation tests before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests (recommended before first training)
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml

    # Run only critical tests (faster, 5-10 minutes)
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml --critical-only

    # Integration with training
    python scripts/preflight_check.py --config configs/retfound_lora_config.yaml && \\
        python scripts/train_retfound_lora.py --config configs/retfound_lora_config.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--critical-only',
        action='store_true',
        help='Run only critical tests (faster validation)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information'
    )

    args = parser.parse_args()

    # Create checker
    checker = PreFlightChecker(
        config_path=args.config,
        critical_only=args.critical_only,
        verbose=args.verbose
    )

    # Run all tests
    try:
        success = checker.run_all_tests()

        # Print summary
        checker.print_summary()

        # Generate reports
        checker.generate_html_report()
        checker.save_json()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pre-flight check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during pre-flight check: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
