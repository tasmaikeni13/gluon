#!/usr/bin/env python3
"""Hold a pod-worker lock and bind a process group to its SSH connection."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


class Cancelled(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("command")
    args = parser.parse_args()
    path = Path(args.lock)
    path.parent.mkdir(parents=True, exist_ok=True)
    child = None

    def cancelled(signum, frame):
        raise Cancelled()

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, cancelled)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Worker busy: another Rocora job holds the pod lock", flush=True)
            return 75
        lock.seek(0)
        lock.truncate()
        lock.write(f"{args.job_id} {os.getpid()}\n")
        lock.flush()
        try:
            print(f"ROCORA_WORKER_READY {args.job_id}", flush=True)
            # No TPU process starts until the launcher has locked every worker.
            if sys.stdin.buffer.readline() != b"START\n":
                return 130
            child = subprocess.Popen(["bash", "-c", args.command], start_new_session=True)

            def disconnected():
                # A daemon blocked on Python's buffered stdin holds its lock
                # during interpreter shutdown. Raw reads avoid that deadlock.
                while os.read(sys.stdin.fileno(), 1):
                    pass
                os.kill(os.getpid(), signal.SIGTERM)

            threading.Thread(target=disconnected, daemon=True).start()
            return child.wait()
        except Cancelled:
            return 130
        finally:
            # Ignore repeated disconnect signals while reaping the entire group.
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(signum, signal.SIG_IGN)
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    sys.exit(main())
