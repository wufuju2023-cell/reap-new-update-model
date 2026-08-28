#!/usr/bin/env python3
"""Stable entrypoint for the segmented Lean/Reap TTT supervisor."""

from .run_toy_ttt import main


if __name__ == "__main__":
    raise SystemExit(main())
