"""MULT-ANYLEN control plane (Mac side; all compute on Modal CPU).

  python3 -B cloud.py verify WEIGHTS.npz --plan main|drift|big|top200|time [--tag T] [--seed S]   (verify_any.py)
  python3 -B cloud.py stop      stop app manylen-verify
"""
import os
import subprocess
import sys

LANE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANE)
os.chdir(LANE)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    c, rest = sys.argv[1], sys.argv[2:]
    if c == "verify":
        import verify_any
        verify_any.main(rest)
    elif c == "stop":
        p = subprocess.run(["modal", "app", "stop", "-y", "manylen-verify"], capture_output=True, text=True)
        print(p.returncode, (p.stdout + p.stderr).strip()[-300:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
