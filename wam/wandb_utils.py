"""W&B helpers shared by the training entrypoint.

The whole point of this module is to make training consume **W&B Registry artifacts**
(so dataset + base models show up on the run's lineage) while never re-downloading the
70GB of base weights — they already live on the checkpoints PVC as reference artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path


def use_and_resolve(run, artifact_path: str, download_root: str | None = None) -> str:
    """`use_artifact` (records an input-lineage edge on `run`) and return a local directory.

    - Reference artifacts (their files are `file://` URIs on the shared PVC) resolve to the
      PVC directory directly, with **no download/copy**.
    - Managed artifacts (e.g. the dataset) are downloaded to `download_root`.
    """
    art = run.use_artifact(artifact_path)

    refs: list[str] = []
    for entry in art.manifest.entries.values():
        ref = getattr(entry, "ref", None)
        if ref and ref.startswith("file://"):
            refs.append(ref[len("file://"):])

    if refs:
        base = os.path.dirname(refs[0]) if len(refs) == 1 else os.path.commonpath(refs)
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"Reference artifact {artifact_path} points at {base}, which is not present on "
                f"this node's PVC mount. Is the checkpoints PVC mounted at the expected path?"
            )
        print(f"[wandb] {artifact_path} -> {base} (reference, no download)")
        return base

    dest = art.download(root=download_root)
    print(f"[wandb] {artifact_path} -> {dest} (downloaded)")
    return dest


def log_checkpoint_artifact(run, name: str, ckpt_dir: str, metadata: dict | None = None,
                            registry: str | None = None, aliases: list[str] | None = None):
    """Log a trained checkpoint directory as a `model` artifact and (optionally) link it to the Registry.

    This is the **output-lineage edge**: dataset + base models -> this run -> checkpoint.
    """
    import wandb

    art = wandb.Artifact(name=name, type="model", metadata=metadata or {},
                         description="Fine-tuned DreamZero DROID pick-place checkpoint.")
    art.add_dir(str(ckpt_dir))
    logged = run.log_artifact(art, aliases=aliases or [])
    logged.wait()
    print(f"[wandb] logged checkpoint artifact {name}:{logged.version} from {ckpt_dir}")
    if registry:
        target = f"{registry}/{name}"
        run.link_artifact(logged, target_path=target)
        print(f"[wandb] linked checkpoint to Registry: {target}")
    return logged
