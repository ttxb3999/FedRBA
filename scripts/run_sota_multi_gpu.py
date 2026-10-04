#!/usr/bin/env python3
# Cross-platform multi-GPU launcher for matched FAL baselines.
# One independent experiment per RTX 3090; no DDP.

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def detect_3090_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}")

    all_gpus = []
    gpus_3090 = []
    for line in out.splitlines():
        if not line.strip():
            continue
        idx_text, name = line.split(",", 1)
        idx = int(idx_text.strip())
        name = name.strip()
        all_gpus.append((idx, name))
        if "3090" in name.upper():
            gpus_3090.append((idx, name))

    if not all_gpus:
        raise RuntimeError("No NVIDIA GPU detected by nvidia-smi.")
    return all_gpus, gpus_3090


def build_job(method, dataset, seed, smoke):
    ds_short = "c10" if dataset == "cifar10_lt" else "c100"
    name = f"sota_{method}_{ds_short}_s{seed}"
    if smoke:
        name = f"smoke_{method}_{ds_short}_s{seed}"
    cmd = [
        sys.executable, "-u", "main.py",
        "--seed", str(seed),
        "--custom_name", name,
        "--al_method", method,
        "--noise_mode", "hetero",
        "--model", "cnn4conv",
        "--dataset", dataset,
        "--imb_ratio", "0.05",
        "--partition", "dir_balance",
        "--dd_beta", "0.1",
        "--num_users", "10",
        "--frac", "1.0",
        "--rounds", "1" if smoke else "300",
        "--local_ep", "1" if smoke else "5",
        "--reset", "random",
        "--query_ratio", "0.05",
        "--end_ratio", "0.10" if smoke else "0.45",
        "--budget_mode", "fixed",
    ]
    return name, cmd


def worker(gpu_id, gpu_name, jobs, log_dir, status_rows, lock):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return

        name, base_cmd = job
        cmd = base_cmd[:]
        cmd[3:3] = ["--gpu", str(gpu_id)]  # after main.py
        log_path = log_dir / f"{name}.log"

        start = time.time()
        print(f"[GPU {gpu_id} | {gpu_name}] START {name}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        with log_path.open("w", encoding="utf-8", errors="replace") as logf:
            logf.write("COMMAND: " + " ".join(cmd) + "\n\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            )
            rc = proc.wait()

        elapsed = time.time() - start
        row = {
            "job": name,
            "gpu": gpu_id,
            "gpu_name": gpu_name,
            "return_code": rc,
            "elapsed_seconds": round(elapsed, 1),
            "log": str(log_path),
        }
        with lock:
            status_rows.append(row)
            state = "DONE" if rc == 0 else "FAILED"
            print(f"[GPU {gpu_id}] {state} {name} ({elapsed/3600:.2f} h)", flush=True)
        jobs.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["kafal"],
                    help="e.g. --methods kafal ; after IFAL is integrated: --methods ifal")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=["cifar10_lt", "cifar100_lt"])
    ap.add_argument("--gpus", nargs="+", type=int, default=None,
                    help="optional explicit GPU IDs; otherwise all detected RTX 3090s")
    ap.add_argument("--smoke", action="store_true",
                    help="2-stage quick check: rounds=1, local_ep=1, end_ratio=0.10")
    args = ap.parse_args()

    if not Path("main.py").exists():
        raise SystemExit("ERROR: run this script from the project root containing main.py")

    all_gpus, gpus_3090 = detect_3090_gpus()
    by_id = {i: n for i, n in all_gpus}

    if args.gpus is not None:
        selected = []
        for i in args.gpus:
            if i not in by_id:
                raise SystemExit(f"ERROR: GPU {i} not found. Detected: {all_gpus}")
            selected.append((i, by_id[i]))
    else:
        selected = gpus_3090
        if not selected:
            print("WARNING: no GPU name containing '3090' was found; using all detected NVIDIA GPUs.")
            selected = all_gpus

    print("Detected GPUs:")
    for i, n in all_gpus:
        mark = "  <-- selected" if any(i == x[0] for x in selected) else ""
        print(f"  GPU {i}: {n}{mark}")

    # For smoke mode, one job is enough to verify the path and KAFAL query stage.
    if args.smoke:
        method = args.methods[0]
        dataset = args.datasets[0]
        seed = args.seeds[0]
        job_defs = [build_job(method, dataset, seed, True)]
    else:
        job_defs = [
            build_job(method, dataset, seed, False)
            for method in args.methods
            for dataset in args.datasets
            for seed in args.seeds
        ]

    log_dir = Path("sota_logs")
    log_dir.mkdir(exist_ok=True)
    jobs = queue.Queue()
    for job in job_defs:
        jobs.put(job)

    print(f"Jobs: {len(job_defs)}; selected GPUs: {len(selected)}")
    for name, _ in job_defs:
        print("  " + name)

    status_rows = []
    lock = threading.Lock()
    threads = []
    for gpu_id, gpu_name in selected:
        t = threading.Thread(
            target=worker,
            args=(gpu_id, gpu_name, jobs, log_dir, status_rows, lock),
            daemon=False,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    status_path = log_dir / "status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["job", "gpu", "gpu_name", "return_code", "elapsed_seconds", "log"],
        )
        writer.writeheader()
        writer.writerows(sorted(status_rows, key=lambda r: r["job"]))

    failed = [r for r in status_rows if r["return_code"] != 0]
    print(f"\nAll queued jobs finished. Status: {status_path}")
    if failed:
        print("FAILED JOBS:")
        for r in failed:
            print(f"  {r['job']} -> {r['log']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
