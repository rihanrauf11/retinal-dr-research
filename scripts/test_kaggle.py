#!/usr/bin/env python3
"""
Kaggle API Configuration Test Script

This script verifies that the Kaggle API is properly configured for downloading
datasets. It performs comprehensive checks and provides clear feedback on the
configuration status.

Usage:
    python scripts/test_kaggle.py

    # With verbose output
    python scripts/test_kaggle.py --verbose

    # Check specific competition access
    python scripts/test_kaggle.py --check-competition aptos2019-blindness-detection

Author: Generated with Claude Code
"""

import os
import sys
import json
import stat
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLOR CODES FOR TERMINAL OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{text}{Colors.END}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{text}{Colors.END}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{text}{Colors.END}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.BLUE}{text}{Colors.END}"

    @staticmethod
    def bold(text: str) -> str:
        return f"{Colors.BOLD}{text}{Colors.END}"


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_kaggle_package() -> Tuple[bool, str]:
    """
    Check if the kaggle package is installed.

    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    try:
        import kaggle
        version = kaggle.__version__
        return True, f"Kaggle package version {version} is installed"
    except ImportError:
        return False, "Kaggle package is not installed"


def check_kaggle_json() -> Tuple[bool, str, Optional[Path]]:
    """
    Check if kaggle.json exists in the expected location.

    Returns
    -------
    Tuple[bool, str, Optional[Path]]
        (success, message, path_to_file)
    """
    # Check for custom config directory
    config_dir = os.environ.get('KAGGLE_CONFIG_DIR')

    if config_dir:
        kaggle_json_path = Path(config_dir) / 'kaggle.json'
    else:
        kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'

    if kaggle_json_path.exists():
        return True, f"Found kaggle.json at {kaggle_json_path}", kaggle_json_path
    else:
        return False, f"kaggle.json not found at {kaggle_json_path}", kaggle_json_path


def check_file_permissions(file_path: Path) -> Tuple[bool, str]:
    """
    Check if kaggle.json has correct permissions (600 on Unix systems).

    Parameters
    ----------
    file_path : Path
        Path to kaggle.json

    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    if sys.platform == 'win32':
        # Windows handles permissions differently
        return True, "Permission check skipped on Windows"

    try:
        file_stat = file_path.stat()
        file_mode = stat.filemode(file_stat.st_mode)

        # Check if permissions are 600 (rw-------)
        permissions = file_stat.st_mode & 0o777

        if permissions == 0o600:
            return True, f"File permissions are correct: {oct(permissions)}"
        else:
            return False, (
                f"File permissions are {oct(permissions)} but should be 0o600. "
                f"Run: chmod 600 {file_path}"
            )
    except Exception as e:
        return False, f"Error checking permissions: {e}"


def check_kaggle_json_content(file_path: Path) -> Tuple[bool, str]:
    """
    Check if kaggle.json has valid content.

    Parameters
    ----------
    file_path : Path
        Path to kaggle.json

    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)

        # Check for required fields
        if 'username' not in content:
            return False, "kaggle.json missing 'username' field"

        if 'key' not in content:
            return False, "kaggle.json missing 'key' field"

        username = content['username']
        key_length = len(content['key'])

        return True, f"Valid credentials found for user: {username} (key length: {key_length})"

    except json.JSONDecodeError:
        return False, "kaggle.json is not valid JSON"
    except Exception as e:
        return False, f"Error reading kaggle.json: {e}"


def check_kaggle_authentication() -> Tuple[bool, str]:
    """
    Check if Kaggle API authentication works.

    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    try:
        import kaggle

        # This will raise an exception if authentication fails
        kaggle.api.authenticate()

        return True, "Kaggle API authentication successful"

    except OSError as e:
        return False, f"Authentication failed: {e}"
    except Exception as e:
        return False, f"Unexpected error during authentication: {e}"


def check_api_connectivity(verbose: bool = False) -> Tuple[bool, str, Optional[list]]:
    """
    Check if we can connect to Kaggle API and list competitions.

    Parameters
    ----------
    verbose : bool
        If True, return list of competitions

    Returns
    -------
    Tuple[bool, str, Optional[list]]
        (success, message, competition_list)
    """
    try:
        import kaggle

        # Try to list competitions (limit to 5 for speed)
        competitions = kaggle.api.competitions_list()[:5]

        competition_names = [comp.ref for comp in competitions]

        message = f"Successfully connected to Kaggle API. Found {len(competitions)} competitions."

        return True, message, competition_names if verbose else None

    except Exception as e:
        return False, f"Failed to connect to Kaggle API: {e}", None


def check_competition_access(competition_name: str) -> Tuple[bool, str]:
    """
    Check if we can access a specific competition.

    Parameters
    ----------
    competition_name : str
        Name of the competition (e.g., 'aptos2019-blindness-detection')

    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    try:
        import kaggle

        # Try to get competition details
        competition = kaggle.api.competition_view(competition_name)

        return True, (
            f"Successfully accessed competition: {competition.title}\n"
            f"           Status: {competition.enabledDate} - {competition.deadline}"
        )

    except Exception as e:
        error_msg = str(e).lower()

        if '403' in error_msg or 'forbidden' in error_msg:
            return False, (
                f"Access denied (403). You may need to:\n"
                f"           1. Visit: https://www.kaggle.com/c/{competition_name}\n"
                f"           2. Click 'Join Competition' or 'I Understand and Accept'\n"
                f"           3. Accept the competition rules"
            )
        elif '404' in error_msg or 'not found' in error_msg:
            return False, f"Competition '{competition_name}' not found (404)"
        else:
            return False, f"Error accessing competition: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_checks(
    verbose: bool = False,
    check_competition: Optional[str] = None
) -> Dict[str, bool]:
    """
    Run all Kaggle API checks and print results.

    Parameters
    ----------
    verbose : bool
        Print additional details
    check_competition : Optional[str]
        Name of specific competition to check access

    Returns
    -------
    Dict[str, bool]
        Dictionary of check results
    """
    print(Colors.bold("\n" + "=" * 80))
    print(Colors.bold("KAGGLE API CONFIGURATION TEST"))
    print(Colors.bold("=" * 80 + "\n"))

    results = {}

    # Check 1: Package installation
    print(Colors.bold("1. Checking Kaggle package installation..."))
    success, message = check_kaggle_package()
    results['package'] = success

    if success:
        print(f"   {Colors.success('✓')} {message}")
    else:
        print(f"   {Colors.error('✗')} {message}")
        print(f"   {Colors.warning('→')} Install with: pip install kaggle")
        print_final_summary(results)
        return results

    # Check 2: kaggle.json existence
    print(Colors.bold("\n2. Checking for kaggle.json file..."))
    success, message, kaggle_json_path = check_kaggle_json()
    results['json_exists'] = success

    if success:
        print(f"   {Colors.success('✓')} {message}")
    else:
        print(f"   {Colors.error('✗')} {message}")
        print(f"   {Colors.warning('→')} See KAGGLE_SETUP.md for instructions")
        print(f"   {Colors.warning('→')} Download from: https://www.kaggle.com/settings/account")
        print_final_summary(results)
        return results

    # Check 3: File permissions (Unix only)
    if sys.platform != 'win32':
        print(Colors.bold("\n3. Checking file permissions..."))
        success, message = check_file_permissions(kaggle_json_path)
        results['permissions'] = success

        if success:
            print(f"   {Colors.success('✓')} {message}")
        else:
            print(f"   {Colors.error('✗')} {message}")
            print(f"   {Colors.warning('→')} Fix with: chmod 600 {kaggle_json_path}")
            print_final_summary(results)
            return results
    else:
        results['permissions'] = True  # Skip on Windows

    # Check 4: JSON content validity
    print(Colors.bold("\n4. Validating kaggle.json content..."))
    success, message = check_kaggle_json_content(kaggle_json_path)
    results['json_valid'] = success

    if success:
        print(f"   {Colors.success('✓')} {message}")
    else:
        print(f"   {Colors.error('✗')} {message}")
        print(f"   {Colors.warning('→')} Regenerate token at: https://www.kaggle.com/settings/account")
        print_final_summary(results)
        return results

    # Check 5: API authentication
    print(Colors.bold("\n5. Testing Kaggle API authentication..."))
    success, message = check_kaggle_authentication()
    results['auth'] = success

    if success:
        print(f"   {Colors.success('✓')} {message}")
    else:
        print(f"   {Colors.error('✗')} {message}")
        print(f"   {Colors.warning('→')} Check your credentials and try regenerating token")
        print_final_summary(results)
        return results

    # Check 6: API connectivity
    print(Colors.bold("\n6. Testing Kaggle API connectivity..."))
    success, message, competitions = check_api_connectivity(verbose=verbose)
    results['connectivity'] = success

    if success:
        print(f"   {Colors.success('✓')} {message}")
        if verbose and competitions:
            print(f"   {Colors.info('→')} Sample competitions:")
            for comp in competitions:
                print(f"      - {comp}")
    else:
        print(f"   {Colors.error('✗')} {message}")
        print_final_summary(results)
        return results

    # Check 7: Specific competition access (if requested)
    if check_competition:
        print(Colors.bold(f"\n7. Checking access to competition: {check_competition}..."))
        success, message = check_competition_access(check_competition)
        results['competition'] = success

        if success:
            print(f"   {Colors.success('✓')} {message}")
        else:
            print(f"   {Colors.error('✗')} {message}")

    # Final summary
    print_final_summary(results)

    return results


def print_final_summary(results: Dict[str, bool]) -> None:
    """
    Print final summary of test results.

    Parameters
    ----------
    results : Dict[str, bool]
        Dictionary of check results
    """
    print(Colors.bold("\n" + "=" * 80))
    print(Colors.bold("SUMMARY"))
    print(Colors.bold("=" * 80))

    all_passed = all(results.values())

    if all_passed:
        print(Colors.success("\n✓ ALL CHECKS PASSED!"))
        print(Colors.success("\nYour Kaggle API is properly configured and ready to use."))
        print("\nYou can now download datasets with:")
        print(Colors.info("  python scripts/prepare_data.py --aptos-only"))
        print(Colors.info("  kaggle competitions download -c aptos2019-blindness-detection"))
    else:
        failed_checks = [k for k, v in results.items() if not v]
        print(Colors.error(f"\n✗ CONFIGURATION INCOMPLETE ({len(failed_checks)} issue(s) found)"))
        print(Colors.warning("\nPlease fix the issues above and run this test again."))
        print("\nFor detailed setup instructions, see:")
        print(Colors.info("  KAGGLE_SETUP.md"))
        print(Colors.info("  https://github.com/Kaggle/kaggle-api"))

    print(Colors.bold("\n" + "=" * 80 + "\n"))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Test Kaggle API configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test
  python scripts/test_kaggle.py

  # Verbose output
  python scripts/test_kaggle.py --verbose

  # Test specific competition access
  python scripts/test_kaggle.py --check-competition aptos2019-blindness-detection

  # Test APTOS competition with verbose output
  python scripts/test_kaggle.py -v --check-competition aptos2019-blindness-detection
        """
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output with additional details'
    )

    parser.add_argument(
        '-c', '--check-competition',
        type=str,
        metavar='COMPETITION_NAME',
        help='Check access to a specific competition (e.g., aptos2019-blindness-detection)'
    )

    parser.add_argument(
        '--show-setup-instructions',
        action='store_true',
        help='Display setup instructions and exit'
    )

    return parser.parse_args()


def show_setup_instructions() -> None:
    """Display quick setup instructions."""
    print(Colors.bold("\n" + "=" * 80))
    print(Colors.bold("KAGGLE API SETUP INSTRUCTIONS"))
    print(Colors.bold("=" * 80 + "\n"))

    print(Colors.bold("1. Install the Kaggle package:"))
    print(Colors.info("   pip install kaggle\n"))

    print(Colors.bold("2. Get your API credentials:"))
    print("   - Go to: https://www.kaggle.com/settings/account")
    print("   - Scroll to 'API' section")
    print("   - Click 'Create New API Token'")
    print(Colors.info("   - This downloads kaggle.json\n"))

    print(Colors.bold("3. Place the credentials file:"))
    if sys.platform == 'win32':
        print(Colors.info("   Windows:"))
        print("   mkdir %USERPROFILE%\\.kaggle")
        print("   move %USERPROFILE%\\Downloads\\kaggle.json %USERPROFILE%\\.kaggle\\")
    else:
        print(Colors.info("   macOS/Linux:"))
        print("   mkdir -p ~/.kaggle")
        print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")

    print(Colors.bold("\n4. Test the setup:"))
    print(Colors.info("   python scripts/test_kaggle.py\n"))

    print(Colors.bold("For detailed instructions, see: KAGGLE_SETUP.md\n"))
    print(Colors.bold("=" * 80 + "\n"))


def main() -> int:
    """
    Main function.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    if args.show_setup_instructions:
        show_setup_instructions()
        return 0

    # Run all checks
    results = run_all_checks(
        verbose=args.verbose,
        check_competition=args.check_competition
    )

    # Return exit code based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(Colors.warning("\n\nTest interrupted by user."))
        sys.exit(1)
    except Exception as e:
        print(Colors.error(f"\n\nUnexpected error: {e}"))
        sys.exit(1)
