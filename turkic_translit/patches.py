"""
Patches for third-party libraries to fix encoding issues on Windows.
This module is imported automatically at startup.
"""
import os
import sys
import functools

def apply_patches():
    """Apply all necessary patches for third-party libraries."""
    # Fix panphon encoding issues on Windows
    if sys.platform == 'win32':
        try:
            import panphon.featuretable
            import io
            import csv
            
            # Save the original open function
            original_open = open
            
            # Monkey patch the built-in open function when used by panphon
            def patched_open_for_panphon(file, mode='r', *args, **kwargs):
                # Add explicit UTF-8 encoding for CSV files opened by panphon
                if 'panphon' in sys.modules and mode == 'r' and isinstance(file, str) and file.endswith('.csv'):
                    if 'encoding' not in kwargs:
                        kwargs['encoding'] = 'utf-8'
                        print(f"Applied UTF-8 encoding patch for {file}")
                return original_open(file, mode, *args, **kwargs)
            
            # Set the environment variable for good measure
            os.environ['PYTHONUTF8'] = '1'
            
            # Apply the patch
            import builtins
            builtins.open = patched_open_for_panphon
            print("Applied panphon UTF-8 patch for Windows")

            # We've already applied the patch above
        except ImportError:
            print("Warning: Could not patch panphon (not installed)")

# Apply patches when module is imported
apply_patches()
