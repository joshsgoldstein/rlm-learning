from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from tui.app import RLMApp


load_dotenv()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLM TUI")
    parser.add_argument("--query", type=str, default="", help="Run this query automatically on startup.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run RLM_TEST_QUERY from .env automatically on startup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    startup_query = args.query.strip()
    if args.test and not startup_query:
        startup_query = os.getenv("RLM_TEST_QUERY", "").strip()
    app = RLMApp(startup_query=startup_query)
    app.run()
