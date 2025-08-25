import os
import subprocess
import sys
import time
import pytest


def test_run_test5_script():
    """Run test5.py as an end-to-end smoke test. Requires GEMINI_API_KEY_1 in env to truly run; otherwise, it will still attempt startup and likely fail quickly, which is acceptable for detecting import/runtime errors."""
    env = os.environ.copy()
    # Ensure project root on PYTHONPATH
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    # Use the venv python if available
    python_exe = os.path.join(project_root, '..', 'venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    script_path = os.path.join(project_root, 'test5.py')
    assert os.path.exists(script_path), f"Missing script: {script_path}"

    # Run with a timeout to avoid hanging
    proc = subprocess.Popen([python_exe, script_path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        out, err = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(f"test5.py timed out. stdout=\n{out}\n\nstderr=\n{err}")

    # Accept exit code 0 as pass; otherwise show logs for debugging
    if proc.returncode != 0:
        print("===== test5.py stdout =====")
        print(out)
        print("===== test5.py stderr =====")
        print(err)
        pytest.fail(f"test5.py exited with code {proc.returncode}")
