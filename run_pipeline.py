import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    """Runs a python script and waits for it to finish."""
    print(f"\n{'='*40}")
    print(f"Starting: {script_path}")
    print(f"{'='*40}")
    

    path = Path(script_path)
    
    if not path.exists():
        print(f"Error: Could not find {script_path}")
        return False

    try:
        subprocess.run([sys.executable, str(path)], check=True)
        print(f"\nFinished successfully: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError: {script_path} failed with exit code {e.returncode}")
        return False

if __name__ == "__main__":

    step1_success = run_script("CSV_train/sum.py")
    
    # Run the feature engineering script ONLY if step 1 succeeded
    if step1_success:
        run_script("dataset/adding_features.py")
    else:
        print("\nPipeline stopped because the first step failed.")

    print("\n🎉 All tasks processed.")