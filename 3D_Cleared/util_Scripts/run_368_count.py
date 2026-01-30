#!/usr/bin/env python3
"""Quick script to run counting on brain 368 and write results."""
import sys
sys.path.insert(0, '.')

# Redirect output to file
with open('368_count_log.txt', 'w') as log:
    sys.stdout = log
    sys.stderr = log

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '6_count_regions.py', '--brain', '368_CNT_03_08_1p625x_z4'],
            capture_output=True,
            text=True,
            timeout=600
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("RETURN CODE:", result.returncode)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("Done")
