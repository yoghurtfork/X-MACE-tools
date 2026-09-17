#!/usr/bin/env python3
"""
Run SHARC MACE and SHARC Molcas ensemble jobs in command argument order
"""

import subprocess
import sys
from pathlib import Path


def main():
    runners_dir = Path(__file__).resolve().parent
    runners = {
        "mace": runners_dir / "sharc_mace_scripts" / "run_sharc_mace.py",
        "molcas": runners_dir / "sharc_molcas_scripts" / "run_sharc_molcas.py",
    }

    mode = None
    jobs = []
    for argument in sys.argv[1:]:
        if argument in runners:
            mode = argument
        elif mode is None:
            print("usage: run_sharc_queue.py {mace|molcas} job.json [...]", file=sys.stderr)
            return 2
        else:
            jobs.append((mode, Path(argument)))
    if not jobs:
        print("usage: run_sharc_queue.py {mace|molcas} job.json [...]", file=sys.stderr)
        return 2

    failed = False
    try:
        for mode, input_path in jobs:
            print(f"\nStarting {mode.upper()} job: {input_path}", flush=True)
            try:
                result = subprocess.run([
                    sys.executable,
                    str(runners[mode]),
                    str(input_path),
                ])
            except OSError:
                failed = True
                continue
            if result.returncode:
                failed = True
    except KeyboardInterrupt:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
