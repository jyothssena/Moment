"""
run_api.py
==========
Launch the Moment API server from the Moment root directory.

Usage (from inside the Moment/ folder):
  python run_api.py                        # stub mode, port 8080
  python run_api.py --port 8081            # different port
  python run_api.py --mode real            # real Gemini agents
  python run_api.py --reload               # auto-reload on file changes (dev)

Why this file exists instead of `uvicorn api.main:app`:
  uvicorn's --reload flag spawns a subprocess on Windows that doesn't
  inherit the working directory's sys.path, so `api` is not findable.
  This script adds the Moment root to sys.path before uvicorn starts,
  which fixes the issue on Windows, Mac, and Linux consistently.
"""

import os
import sys
import argparse

# ── Ensure Moment root is on sys.path before anything else ───────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))   # Moment/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Start the Moment API server")
parser.add_argument("--port",   type=int, default=8080,   help="Port (default 8080)")
parser.add_argument("--host",            default="0.0.0.0", help="Host (default 0.0.0.0)")
parser.add_argument("--mode",            default="stub",   choices=["stub", "real"],
                    help="Model mode: stub (no Gemini key) or real (requires GEMINI_API_KEY_MOMENT)")
parser.add_argument("--reload", action="store_true",       help="Auto-reload on file changes (dev only)")
parser.add_argument("--workers", type=int, default=1,      help="Number of worker processes")
args = parser.parse_args()

# ── Set env before importing uvicorn or the app ───────────────────────────────
os.environ["MODEL_MODE"] = args.mode

print(f"\nMoment API")
print(f"  Mode:    {args.mode}")
print(f"  Address: http://{args.host}:{args.port}")
print(f"  Docs:    http://localhost:{args.port}/docs")
print(f"  Reload:  {args.reload}\n")

# ── Start uvicorn programmatically — no subprocess spawning issues ────────────
import uvicorn

uvicorn.run(
    "api.main:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
    workers=1 if args.reload else args.workers,   # reload is incompatible with workers > 1
)
