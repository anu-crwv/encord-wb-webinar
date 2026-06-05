# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pillow",
#     "typer",
#     "wandb>=0.18.0",
# ]
# ///

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
import wandb
from PIL import Image


DEFAULT_ENTITY = "encord-wb-physical-ai"
DEFAULT_PROJECT = "wam-finetune-webinar"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{path} is not valid JSON: {exc}") from exc


def _iter_data_units(payload: Any) -> list[dict[str, Any]]:
    """Return a normalized list of image/video data units from common Encord export shapes."""

    if isinstance(payload, dict) and isinstance(payload.get("data_units"), dict):
        return [
            {"data_hash": data_hash, **data_unit}
            for data_hash, data_unit in payload["data_units"].items()
            if isinstance(data_unit, dict)
        ]

    if isinstance(payload, dict) and isinstance(payload.get("label_rows"), list):
        rows = payload["label_rows"]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = [payload] if isinstance(payload, dict) else []

    data_units: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        nested_units = row.get("data_units")
        if isinstance(nested_units, dict):
            for data_hash, data_unit in nested_units.items():
                if isinstance(data_unit, dict):
                    merged = {
                        "data_hash": data_hash,
                        "label_hash": row.get("label_hash"),
                        "data_title": data_unit.get("data_title") or row.get("data_title"),
                        "data_type": data_unit.get("data_type") or row.get("data_type"),
                        **data_unit,
                    }
                    data_units.append(merged)
            continue

        data_units.append(
            {
                "data_hash": row.get("data_hash"),
                "label_hash": row.get("label_hash"),
                "data_title": row.get("data_title"),
                "data_type": row.get("data_type"),
                "width": row.get("width"),
                "height": row.get("height"),
                "labels": row.get("labels") or row.get("label"),
                **row,
            }
        )

    return data_units


def _labels_for_data_unit(data_unit: dict[str, Any]) -> list[dict[str, Any]]:
    labels = data_unit.get("labels")
    if isinstance(labels, dict):
        return [label for label in labels.values() if isinstance(label, dict)]
    if isinstance(labels, list):
        return [label for label in labels if isinstance(label, dict)]

    label = data_unit.get("label")
    if isinstance(label, dict):
        return [label]

    return []


def _objects_for_data_unit(data_unit: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for label in _labels_for_data_unit(data_unit):
        label_objects = label.get("objects")
        if isinstance(label_objects, list):
            objects.extend(obj for obj in label_objects if isinstance(obj, dict))
    return objects


def _class_name(obj: dict[str, Any]) -> str:
    for key in ("name", "class_name", "object_name", "value"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _shape_name(obj: dict[str, Any]) -> str:
    for key in ("shape", "annotation_type", "type"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    if "bounding_box" in obj:
        return "bounding_box"
    if "polygon" in obj:
        return "polygon"
    if "bitmask" in obj:
        return "bitmask"
    return "unknown"


def _bbox_payload(obj: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("bounding_box", "bbox", "box"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value

    if all(key in obj for key in ("x", "y", "w", "h")):
        return obj

    return None


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_to_wandb_box(
    obj: dict[str, Any],
    width: int,
    height: int,
    class_ids: dict[str, int],
) -> dict[str, Any] | None:
    bbox = _bbox_payload(obj)
    if not bbox:
        return None

    x = _numeric(bbox.get("x"))
    y = _numeric(bbox.get("y"))
    w = _numeric(bbox.get("w"))
    h = _numeric(bbox.get("h"))
    if None in (x, y, w, h):
        return None

    # Encord boxes are normally normalized. If values look pixel-scaled already, preserve them.
    assert x is not None and y is not None and w is not None and h is not None
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        min_x = x * width
        min_y = y * height
        max_x = (x + w) * width
        max_y = (y + h) * height
    else:
        min_x = x
        min_y = y
        max_x = x + w
        max_y = y + h

    name = _class_name(obj)
    return {
        "position": {
            "minX": max(0.0, min(float(width), min_x)),
            "minY": max(0.0, min(float(height), min_y)),
            "maxX": max(0.0, min(float(width), max_x)),
            "maxY": max(0.0, min(float(height), max_y)),
        },
        "domain": "pixel",
        "class_id": class_ids[name],
        "box_caption": name,
    }


def _resolve_image_path(image_root: Path | None, data_title: str | None) -> Path | None:
    if image_root is None or not data_title:
        return None

    candidates = [
        image_root / data_title,
        image_root / Path(data_title).name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def _coerce_dimension(data_unit: dict[str, Any], key: str) -> int | None:
    value = _numeric(data_unit.get(key))
    if value is not None:
        return int(value)

    nested = data_unit.get("data_unit")
    if isinstance(nested, dict):
        value = _numeric(nested.get(key))
        if value is not None:
            return int(value)

    return None


def _write_summary(
    output_dir: Path,
    data_units: list[dict[str, Any]],
    project_hash: str | None,
    class_counts: Counter[str],
    shape_counts: Counter[str],
) -> Path:
    summary_path = output_dir / "encord_label_summary.json"
    summary = {
        "project_hash": project_hash,
        "data_unit_count": len(data_units),
        "object_count": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path


def main(
    labels_json: Annotated[
        Path,
        typer.Option(help="Path to the exported Encord label JSON."),
    ] = Path("encord_labels.json"),
    entity: Annotated[str, typer.Option(help="W&B entity/org to log into.")] = DEFAULT_ENTITY,
    project: Annotated[str, typer.Option(help="W&B project to log into.")] = DEFAULT_PROJECT,
    artifact_name: Annotated[
        str,
        typer.Option(help="Name for the W&B dataset artifact."),
    ] = "encord-labels",
    image_root: Annotated[
        Path | None,
        typer.Option(help="Optional root folder containing images referenced by data_title."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(help="Local folder for generated summaries before W&B upload."),
    ] = Path("wandb_export"),
    max_table_rows: Annotated[
        int,
        typer.Option(help="Maximum rows to include in the W&B Table preview."),
    ] = 500,
) -> None:
    """Log an Encord label export to W&B as an artifact plus an inspection table."""

    if max_table_rows < 1:
        raise typer.BadParameter("--max-table-rows must be at least 1")
    if not labels_json.exists():
        raise typer.BadParameter(f"Labels JSON does not exist: {labels_json}")

    payload = _load_json(labels_json)
    data_units = _iter_data_units(payload)
    if not data_units:
        typer.echo("No data units found. Check that this is an Encord label export JSON.")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    for data_unit in data_units:
        for obj in _objects_for_data_unit(data_unit):
            class_counts[_class_name(obj)] += 1
            shape_counts[_shape_name(obj)] += 1

    class_names = sorted(class_counts) or ["unknown"]
    class_ids = {name: idx for idx, name in enumerate(class_names)}
    class_labels = {idx: name for name, idx in class_ids.items()}
    project_hash = None
    if isinstance(payload, dict):
        export_info = payload.get("export_info")
        if isinstance(export_info, dict):
            project_hash = export_info.get("project_hash")

    summary_path = _write_summary(output_dir, data_units, project_hash, class_counts, shape_counts)

    table = wandb.Table(
        columns=[
            "data_hash",
            "data_title",
            "data_type",
            "width",
            "height",
            "object_count",
            "class_counts",
            "shape_counts",
            "image",
        ]
    )

    image_root = image_root.expanduser() if image_root else None
    table_rows = 0
    for data_unit in data_units:
        if table_rows >= max_table_rows:
            break

        objects = _objects_for_data_unit(data_unit)
        row_class_counts = Counter(_class_name(obj) for obj in objects)
        row_shape_counts = Counter(_shape_name(obj) for obj in objects)
        width = _coerce_dimension(data_unit, "width")
        height = _coerce_dimension(data_unit, "height")
        data_title = data_unit.get("data_title")
        data_type = data_unit.get("data_type")

        image_value = None
        image_path = _resolve_image_path(image_root, data_title)
        if image_path:
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            width = width or image_width
            height = height or image_height
            boxes = []
            if width and height:
                for obj in objects:
                    box = _bbox_to_wandb_box(obj, width, height, class_ids)
                    if box:
                        boxes.append(box)

            image_value = wandb.Image(
                str(image_path),
                boxes={
                    "ground_truth": {
                        "box_data": boxes,
                        "class_labels": class_labels,
                    }
                }
                if boxes
                else None,
            )

        table.add_data(
            data_unit.get("data_hash"),
            data_title,
            data_type,
            width,
            height,
            len(objects),
            json.dumps(dict(sorted(row_class_counts.items()))),
            json.dumps(dict(sorted(row_shape_counts.items()))),
            image_value,
        )
        table_rows += 1

    metadata = {
        "source": "encord",
        "project_hash": project_hash,
        "data_unit_count": len(data_units),
        "object_count": sum(class_counts.values()),
        "classes": class_names,
        "shapes": sorted(shape_counts),
        "table_rows": table_rows,
    }

    with wandb.init(entity=entity, project=project, job_type="encord-label-export") as run:
        run.log({"encord_label_preview": table})

        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata=metadata,
            description="Raw Encord labels and summary exported for W&B lineage tracking.",
        )
        artifact.add_file(str(labels_json), name="encord_labels.json")
        artifact.add_file(str(summary_path), name="encord_label_summary.json")
        logged = run.log_artifact(artifact, aliases=["latest"])
        logged.wait()

        typer.echo(
            f"Logged W&B artifact {entity}/{project}/{artifact_name}:{logged.version} "
            f"with {len(data_units)} data units and {sum(class_counts.values())} objects."
        )


if __name__ == "__main__":
    typer.run(main)
