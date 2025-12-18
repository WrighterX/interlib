import os
import subprocess
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the name of this script to exclude it
this_script = os.path.basename(__file__)

# Find all .py files in the directory, excluding this script and __init__.py
python_files = [
    f for f in os.listdir(script_dir)
    if f.endswith('.py')
    and os.path.isfile(os.path.join(script_dir, f))
    and f != this_script
]

# Launch each script sequentially
for file in python_files:
    script_path = os.path.join(script_dir, file)
    print(f"\nLaunching {file}...")
    try:
        # Use the same Python interpreter that's running this script
        result = subprocess.run([sys.executable, script_path], check=False)
        if result.returncode == 0:
            print(f"{file} completed successfully.")
        else:
            print(f"{file} exited with error code {result.returncode}.")
    except Exception as e:
        print(f"Failed to launch {file}: {e}")

print("\nAll tests have been launched.")