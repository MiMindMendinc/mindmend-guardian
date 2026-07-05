"""Backward-compatible entrypoint for the Luna safety backend."""

from __future__ import annotations

import logging
import sys

from guardian.luna.app import create_app
from guardian.luna.config import load_config
from guardian.luna.safety import scan_message, toxicity_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_tests() -> None:
    print("Running Luna Safety Core Tests...\n")

    flagged = scan_message("please meet me alone at the hotel")
    assert flagged["is_flagged"] is True, "Expected grooming phrase to be flagged"

    safe = scan_message("hello friend")
    assert safe["is_flagged"] is False, "Expected benign message to pass"

    toxic = toxicity_score("you are stupid and I hate you")
    assert toxic["toxic"] is True, "Expected keyword fallback toxicity match"

    print("\nAll tests complete! Luna's ready.")


def main() -> None:
    config = load_config()
    app = create_app(config)
    app.run(debug=config.flask_debug, host=config.flask_host, port=config.flask_port)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        main()
