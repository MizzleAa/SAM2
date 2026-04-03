"""테스트 결과를 파일에 저장하는 러너"""
import subprocess
import sys
import os

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

result = subprocess.run(
    [sys.executable, "-m", "pytest", 
     "hvs/tests/test_02b_overfit_multi.py", 
     "-v", "--tb=short", "-s", "--no-header"],
    capture_output=True, text=True, 
    cwd=r"c:\workspace\SAM2",
    encoding="utf-8",
    env=env,
)

with open(r"c:\workspace\SAM2\hvs\tests\test_output.log", "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout or "(empty)")
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr or "(empty)")
    f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")

print(f"Exit code: {result.returncode}")
print("Results written to hvs/tests/test_output.log")
