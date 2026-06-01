"""Reliable launch shim for the upstream Hydra trainer.

torchrun runs THIS module (`torchrun ... -m wam._ds_launch <hydra overrides>`) instead of
experiment.py directly, so we can apply training-time, non-architectural patches to the pip-installed
DeepSpeed *before* it is used — without editing `groot/` and without depending on site/sitecustomize
auto-import (which doesn't fire in this image). Then we hand off to the unmodified upstream entrypoint
via runpy, preserving Hydra's `@hydra.main` argv + config-path resolution.
"""

import runpy
import sys

import wam._ds_zero3_leaf  # noqa: F401  (import installs the ZeRO-3 leaf-module patch)

_ENTRY = "groot/vla/experiment/experiment.py"

if __name__ == "__main__":
    # `-m` gives us sys.argv = [<this module path>, <hydra overrides>...]; Hydra reads argv[1:].
    sys.argv = [_ENTRY] + sys.argv[1:]
    try:
        runpy.run_path(_ENTRY, run_name="__main__")
    finally:
        # HF's WandbCallback does not finish the run on exit, which leaves it "in use" and blocks the
        # launcher (rank0) from reopening it to log the checkpoint artifact. Finish it here (only the
        # global-rank0 trainer has an active run; other ranks are no-ops).
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.finish()
        except Exception:  # noqa: BLE001
            pass
