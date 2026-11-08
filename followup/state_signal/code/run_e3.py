"""run_e3.py --ck DIR --program P --norm N --out OUT --procs K --stop-seconds S: e3.py over every checkpoint in DIR plus the teacher, K processes on one GPU.
A checkpoint with all its JSONs is skipped; no new checkpoint starts after the stop."""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--stop-seconds", type=float, default=14400)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sets", default="3,20261107")
    ap.add_argument("--sizes", default="16,64,128")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = [sys.executable, "-B", os.path.join(HERE, "e3.py"), "--program", a.program, "--norm", a.norm, "--out", a.out, "--sets", a.sets, "--sizes", a.sizes]
    jobs = [("teacher", base + ["--teacher", "--name", "teacher"])]
    for d in sorted(glob.glob(os.path.join(a.ck, "s-*"))):
        name = os.path.basename(d)
        jobs.append((name, base + ["--ckpt", os.path.join(d, "best.npz"), "--name", name, "--arm", name.split("-")[1]]))
    if a.only:
        jobs = [j for j in jobs if j[0] in a.only]
    n_files = len(a.sizes.split(",")) * len(a.sets.split(","))
    jobs = [j for j in jobs if len(glob.glob(os.path.join(a.out, j[0] + "-n*.json"))) < n_files]
    print(len(jobs), "checkpoints", flush=True)
    stop = time.time() + a.stop_seconds
    live, fails = [], []
    while (jobs and time.time() < stop) or live:
        while jobs and len(live) < a.procs and time.time() < stop:
            name, c = jobs.pop(0)
            live.append((subprocess.Popen(c, stdout=open(os.path.join(a.out, name + ".log"), "a"), stderr=subprocess.STDOUT), name, time.time()))
        for row in list(live):
            if row[0].poll() is not None:
                live.remove(row)
                print("end", row[1], "rc", row[0].returncode, round(time.time() - row[2]), flush=True)
                if row[0].returncode:
                    fails.append(row[1])
        time.sleep(3)
    print("failed:", fails, "not started:", [j[0] for j in jobs], flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
