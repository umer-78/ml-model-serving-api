"""mlserve: train a model version and run the service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .registry import ModelRegistry
from .train import MODELS, train


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps the real work so that piping into `head` — which closes
    the pipe early — ends quietly instead of printing a BrokenPipeError."""
    try:
        return _run(argv)
    except BrokenPipeError:
        # The reader went away. Point stdout at the void so the interpreter's
        # own flush on exit does not raise the same error again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


def _run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlserve", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and save a new model version")
    t.add_argument("--version", help="defaults to a UTC timestamp")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--models", type=Path, default=MODELS)

    sub.add_parser("versions", help="list model versions").add_argument(
        "--models", type=Path, default=MODELS)

    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--models", type=Path, default=MODELS)
    s.add_argument("--reload", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "train":
        print(json.dumps(train(args.version, args.seed, args.models), indent=2))
        return 0

    if args.cmd == "versions":
        registry = ModelRegistry(args.models)
        versions = registry.versions()
        if not versions:
            print(f"no model versions in {args.models} — run `mlserve train`")
            return 1
        for v in versions:
            bundle = registry.load(v)
            m = bundle.metadata["metrics"]
            marker = "*" if v == registry.latest() else " "
            print(f"{marker} {v}  accuracy {m['accuracy']:.3f}  f1 {m['f1_macro']:.3f}  "
                  f"{bundle.metadata['created_at'][:19]}")
        return 0

    import uvicorn
    from .app import create_app

    uvicorn.run(create_app(args.models), host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
