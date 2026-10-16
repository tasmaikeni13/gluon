#!/usr/bin/env python3
"""Sync and run one coordinated JAX job across the existing four TPU workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rocora.config import OPTIMIZERS


def run_workers(pod, commands, guarded=True):
    """Claim every worker before launch, and reap remote groups on any failure."""
    if len(commands) != len(pod["workers"]):
        raise ValueError("Exactly one command per worker is required")
    jobs = []
    ready = [threading.Event() for _ in commands]
    job_id = uuid.uuid4().hex
    wrapper = (ROOT / "scripts/worker.py").read_text()
    with ThreadPoolExecutor(max_workers=len(commands)) as readers:
        try:
            for worker, command in zip(pod["workers"], commands):
                if guarded:
                    command = shlex.join([pod["python"], "-u", "-c", wrapper,
                                          "--lock", pod["lock_file"], "--job-id", job_id, command])
                process = subprocess.Popen(["ssh", "-T", "-o", "ConnectTimeout=15", worker, command],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                jobs.append(process)

            def read_output(proc, name, event):
                for line in proc.stdout:
                    if line.rstrip() == f"ROCORA_WORKER_READY {job_id}":
                        event.set()
                    else:
                        print(f"[{name}] {line.rstrip()}", flush=True)

            for worker, process, event in zip(pod["workers"], jobs, ready):
                readers.submit(read_output, process, worker, event)
            if guarded:
                deadline = time.monotonic() + 45
                while not all(event.is_set() for event in ready):
                    if any(p.poll() is not None for p in jobs):
                        raise RuntimeError("Could not lock every worker; no TPU job was started")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Timed out acquiring pod locks; no TPU job was started")
                    time.sleep(.05)
                for process in jobs:
                    process.stdin.write("START\n")
                    process.stdin.flush()
            while True:
                codes = [p.poll() for p in jobs]
                if any(code not in (None, 0) for code in codes):
                    raise RuntimeError(f"Worker exit codes: {dict(zip(pod['workers'], codes))}")
                if all(code is not None for code in codes):
                    break
                time.sleep(.05)
        finally:
            # EOF reaches the remote wrapper even when the training process
            # ignores SIGTERM. The wrapper kills/reaps its process group.
            for process in jobs:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            for process in jobs:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if any(codes):
        raise RuntimeError(f"Worker exit codes: {dict(zip(pod['workers'], codes))}")


def sync(pod):
    # Transfer only source, configuration and docs. Never copy token streams or outputs.
    included = ["src", "configs", "scripts", "tests", "docs", "lean", "README.md", "AGENTS.md",
                "theory.md", "pyproject.toml", "LICENSE", ".gitignore"]
    remote = shlex.quote(pod["remote_dir"])
    with tempfile.TemporaryDirectory(prefix="rocora-sync-") as temp:
        archive = Path(temp) / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as out:
            for name in included:
                path = ROOT / name
                if path.exists():
                    out.add(path, arcname=name, filter=lambda info: None if
                            ("__pycache__" in info.name or ".pytest_cache" in info.name) else info)
        remote_archive = "/tmp/rocora-source-" + uuid.uuid4().hex + ".tar.gz"
        commands = []
        for worker in pod["workers"]:
            subprocess.run(["scp", "-q", str(archive), f"{worker}:{remote_archive}"], check=True)
            command = (f"mkdir -p {remote} && "
                       f"if test -e {remote}/.rocora-managed || test -z \"$(ls -A {remote})\"; then "
                       f"touch {remote}/.rocora-managed; else echo 'Refusing to replace an unmanaged directory'; exit 1; fi && "
                       f"tar -xzf {shlex.quote(remote_archive)} -C {remote} && "
                       # Retire the custom optimizer implementation on workers
                       # too; preserve all outputs/data and unrelated files.
                       f"rm -f {remote}/src/rocora/kernels.py && rm {shlex.quote(remote_archive)}")
            commands.append(command)
        # The same locks prohibit source mutation while training is active.
        run_workers(pod, commands)
        for worker in pod["workers"]:
            print(f"Synced {worker}:{pod['remote_dir']}", flush=True)


def train_commands(pod, config, optimizer, seed, extra):
    commands = []
    for i, worker in enumerate(pod["workers"]):
        args = [pod["python"], "-u", "-m", "rocora.train", "--config", config,
                "--optimizer", optimizer, "--seed", str(seed),
                "--coordinator", pod["coordinator"], "--processes", str(len(pod["workers"])),
                "--process-id", str(i), *extra]
        commands.append("cd " + shlex.quote(pod["remote_dir"]) +
                        " && env PYTHONPATH=src JAX_PLATFORMS=tpu " + shlex.join(args))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", default=str(ROOT / "configs/pod.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync")
    sub.add_parser("status")
    p = sub.add_parser("sweep")
    p.add_argument("--spec", required=True)
    p.add_argument("--plan-only", action="store_true", help="Materialize configs without SSH or training")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-steps", type=int)
    for name in ("run", "stage"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--max-steps", type=int)
        p.add_argument("--run-name")
        p.add_argument("--skip-checkpoint", action="store_true")
        if name == "run":
            p.add_argument("--optimizer", choices=OPTIMIZERS, required=True)
            p.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if hasattr(args,"max_steps") and args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if getattr(args, "skip_checkpoint", False) and (not args.max_steps or args.resume):
        parser.error("--skip-checkpoint requires --max-steps and cannot be used with --resume")
    pod = json.loads(Path(args.pod).read_text())
    if args.command == "sync":
        sync(pod)
    elif args.command == "status":
        status_code = """import fcntl,json,pathlib,socket,sys
info={'host':socket.gethostname(),'training':[],'lock_held':False}
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=path.read_bytes().split(b'\\0')
  if b'-m' in args and b'rocora.train' in args:
   info['training'].append({'pid':int(path.parent.name),'args':[a.decode() for a in args if a]})
 except (OSError,UnicodeError): pass
lock=pathlib.Path(sys.argv[1])
if lock.exists():
 with lock.open() as stream:
  try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError: info['lock_held']=True
print(json.dumps(info,sort_keys=True))
"""
        command = shlex.join([pod["python"],"-c",status_code,pod["lock_file"]])
        run_workers(pod, [command] * len(pod["workers"]), guarded=False)
    elif args.command == "sweep":
        from rocora.sweep import materialize
        manifest = materialize(args.spec)
        print(json.dumps({k: v for k, v in manifest.items() if k != "trials"}, indent=2), flush=True)
        if args.plan_only:
            return
        # Generated configs must reach all replicas under the normal sync lock.
        sync(pod)
        for trial in manifest["trials"]:
            run_name = ("validation/" if args.max_steps else "") + trial["run_name"]
            extra = ["--run-name", run_name]
            if args.resume:
                check = ("import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                         "lines=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines() if l] if (p/'metrics.jsonl').exists() else []; "
                         "complete=any(l.get('type')=='complete' and l.get('budget_complete') for l in lines); "
                         "print(json.dumps({'exists':p.exists(),"
                         "'complete':complete,"
                         "'checkpoint':(p/'checkpoints/latest.npz').exists()}))")
                command = "cd " + shlex.quote(pod["remote_dir"]) + " && " + shlex.join(
                    [pod["python"], "-c", check, str(Path(trial["config"]["output_dir"]) / run_name)])
                status = json.loads(subprocess.check_output(["ssh", pod["workers"][0], command], text=True))
                if status.get("complete"):
                    print(f"Skipping completed trial {trial['id']}", flush=True)
                    continue
                if status["checkpoint"]:
                    extra.append("--resume")
                elif status["exists"] and not args.dry_run:
                    raise RuntimeError(f"{run_name} has no checkpoint; preserve it and choose a new sweep name")
            if args.dry_run:
                extra.append("--dry-run")
            if args.max_steps:
                extra.extend(["--max-steps", str(args.max_steps)])
            run_workers(pod, train_commands(pod, trial["config_path"], trial["optimizer"],
                                            trial["config"]["seed"], extra))
    else:
        extra = []
        if args.resume:
            extra.append("--resume")
        if args.dry_run:
            extra.append("--dry-run")
        if args.max_steps:
            extra += ["--max-steps", str(args.max_steps)]
        if args.skip_checkpoint:
            extra.append("--skip-checkpoint")
        if args.run_name:
            if args.command == "stage":
                parser.error("--run-name is only valid with run; stage creates distinct arm/seed names")
            extra += ["--run-name", args.run_name]
        if args.command == "run":
            if args.max_steps and not args.run_name:
                stage=json.loads((ROOT/args.config).read_text())["stage"]
                extra += ["--run-name",f"validation/{stage}/{args.optimizer}-s{args.seed}"]
            run_workers(pod, train_commands(pod, args.config, args.optimizer, args.seed, extra))
        else:
            config = json.loads((ROOT / args.config).read_text())
            stage = config["stage"]
            for seed in config.get("seeds", pod["seeds"].get(stage, [1337])):
                for optimizer in config.get("arms", ("adamw", "sophia", "rocora")):
                    run_name = f"validation/{stage}/{optimizer}-s{seed}" if args.max_steps else f"{stage}/{optimizer}-s{seed}"
                    arm_extra = [x for x in extra if x != "--resume"]
                    p = Path(config.get("output_dir", "outputs")) / run_name
                    m = p / "metrics.jsonl"
                    lines = [json.loads(l) for l in m.read_text().splitlines() if l.strip()] if m.exists() else []
                    complete = any(l.get("type") == "complete" and l.get("budget_complete") for l in lines)
                    if args.resume:
                        if complete:
                            print(f"Skipping completed run {run_name}", flush=True)
                            continue
                        if (p / "checkpoints" / "latest.npz").exists():
                            arm_extra.append("--resume")
                    elif complete:
                        print(f"Run {run_name} is already complete; skipping (use fresh run or inspect metrics)", flush=True)
                        continue
                    print(f"Starting {stage}: {optimizer}, seed {seed}", flush=True)
                    if args.max_steps:
                        arm_extra += ["--run-name", f"validation/{stage}/{optimizer}-s{seed}"]
                    run_workers(pod, train_commands(pod, args.config, optimizer, seed, arm_extra))
                    # Checkpoint management: offload/clean to protect coordinator storage
                    lines = [json.loads(l) for l in m.read_text().splitlines() if l.strip()] if m.exists() else []
                    complete = any(l.get("type") == "complete" and l.get("budget_complete") for l in lines)
                    ckpt = p / "checkpoints" / "latest.npz"
                    if complete and ckpt.exists():
                        try:
                            backup_dir = f"/home/tasma/checkpoints_backup/{stage}"
                            subprocess.run(["ssh", "w1", f"mkdir -p {backup_dir}"], check=False)
                            subprocess.run(["scp", "-q", str(ckpt), f"w1:{backup_dir}/{optimizer}-s{seed}.npz"], check=False)
                        except Exception:
                            pass
                        ckpt.unlink(missing_ok=True)
                        print(f"Archived/cleaned {ckpt} to preserve coordinator disk space.", flush=True)


if __name__ == "__main__":
    main()
