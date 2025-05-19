#!/usr/bin/env python3
"""
Test Runner Script for Cairo Traffic Project
Runs all test files individually and displays results for each
"""

import os
import sys
import unittest
import importlib
import time
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_separator(title=None):
    """Print a separator line with optional title"""
    width = 80
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + Colors.BOLD + "=" * padding + f" {title} " + "=" * padding + Colors.ENDC)
    else:
        print("\n" + Colors.BOLD + "=" * width + Colors.ENDC)

def run_test_file(test_file):
    """Run a single test file and return results"""
    module_name = test_file.replace('.py', '')
    
    # Capture stdout and stderr
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        try:
            # Import the test module
            test_module = importlib.import_module(module_name)
            
            # Create a test suite from the module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run the tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            # Return test results and output
            return {
                'file': test_file,
                'total': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'success': result.wasSuccessful(),
                'output': output.getvalue()
            }
        except Exception as e:
            return {
                'file': test_file,
                'total': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'output': f"Error running tests: {str(e)}\n{output.getvalue()}"
            }

def main():
    """Main function to run all tests"""
    # Get all test files
    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
    test_files.sort()  # Sort alphabetically
    
    print_separator("CAIRO TRAFFIC PROJECT TEST RUNNER")
    print(f"{Colors.BLUE}Running {len(test_files)} test files...{Colors.ENDC}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Summary statistics
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    failed_files = []
    passed_files = []
    
    # Run each test file
    for test_file in test_files:
        print_separator(test_file)
        result = run_test_file(test_file)
        
        # Print test output
        print(result['output'])
        
        # Update statistics
        total_tests += result['total']
        total_failures += result['failures']
        total_errors += result['errors']
        total_skipped += result['skipped']
        
        if not result['success']:
            failed_files.append(test_file)
        else:
            passed_files.append(test_file)
    
    # Print summary
    print_separator("SUMMARY")
    print(f"{Colors.BOLD}Total test files: {len(test_files)}{Colors.ENDC}")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    
    if passed_files:
        print(f"\n{Colors.GREEN}Passed test files ({len(passed_files)}):{Colors.ENDC}")
        for file in passed_files:
            print(f"  ✓ {file}")
    
    if failed_files:
        print(f"\n{Colors.RED}Failed test files ({len(failed_files)}):{Colors.ENDC}")
        for file in failed_files:
            print(f"  ✗ {file}")
    else:
        print(f"\n{Colors.GREEN}All test files passed successfully!{Colors.ENDC}")
    
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Return exit code based on test results
    return 1 if (total_failures > 0 or total_errors > 0) else 0

if __name__ == "__main__":
    try:
        import streamlit
    except ImportError:
        import unittest.mock
        sys.modules['streamlit'] = unittest.mock.MagicMock()
    
    sys.exit(main())
