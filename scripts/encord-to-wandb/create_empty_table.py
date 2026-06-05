#!/usr/bin/env python3
"""Create an empty W&B Table as a simple connectivity smoke test.

This is useful when a tool asks for a W&B table name and you want to confirm
that your local W&B login, entity, and project are working.
"""

from __future__ import annotations

import argparse

import wandb


DEFAULT_ENTITY = "encord-wb-physical-ai"
DEFAULT_PROJECT = "wam-finetune-webinar"
DEFAULT_TABLE_NAME = "encord_labels"
DEFAULT_RUN_NAME = "create-empty-encord-table"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an empty W&B Table.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="W&B entity or org.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project.")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME, help="Name/key for the W&B Table.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Name for the W&B run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    table = wandb.Table(
        columns=[
            "data_hash",
            "data_title",
            "label_status",
            "notes",
        ]
    )

    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        job_type="create-empty-table",
    ) as run:
        run.log({args.table_name: table})
        print(f"Logged empty table '{args.table_name}' to {run.url}")


if __name__ == "__main__":
    main()
