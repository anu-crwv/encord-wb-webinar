# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "encord",
#     "typer",
# ]
# ///
"""Compatibility wrapper for updating the Encord client metadata schema.

Prefer:
    uv run --script scripts/encord/data-registration/update_metadata_schema.py \
      registration.json --dry-run
"""

from __future__ import annotations

import typer

from update_metadata_schema import main


if __name__ == "__main__":
    typer.run(main)
