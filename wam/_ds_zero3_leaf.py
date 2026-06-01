"""Make DeepSpeed ZeRO-3 compatible with the Wan VAE / text / image encoders.

ZeRO-3 installs per-submodule forward hooks to gather/release partitioned params. That breaks the
Wan VAE's `isinstance(layer, CausalConv3d)` checks which drive its causal-conv `feat_cache` indexing
(`feat_idx` desyncs -> wrong cache slot -> `cat 384 vs 96` channel mismatch). The documented fix is
to mark such custom-forward modules as ZeRO-3 **leaf modules**: their whole subtree's params are
gathered together for the module's forward and the internals are NOT hooked, so `isinstance` stays
intact. The frozen encoders are small relative to the 14B DiT, which still shards normally.

We apply this by wrapping `deepspeed.initialize` (called by HF Trainer) so it tags the encoder
classes on the raw model right before the engine is built. Importing this module installs the patch;
it is imported by `wam/_ds_launch.py`, which the trainer is launched through.
"""

from __future__ import annotations

import os

# Wan submodules with custom (isinstance-driven) forwards that ZeRO-3 must not hook into.
LEAF_CLASS_NAMES = {"WanVideoVAE", "WanTextEncoder", "WanImageEncoder", "Encoder3d", "Decoder3d"}


def _install_checkpoint_patch() -> None:
    """Route torch activation checkpointing through DeepSpeed's ZeRO-3-aware checkpoint.

    The DiT calls `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` internally. Under
    ZeRO-3 the sharded params are released after forward, so torch's recompute in backward sees
    shape-[0] params -> CheckpointError. `deepspeed.checkpointing.checkpoint` re-gathers the
    partitioned params during recompute, fixing this. Only active when DeepSpeed is configured.
    """
    if not os.environ.get("DEEPSPEED"):
        return
    try:
        import torch.utils.checkpoint as tuc
        from functools import partial
        from deepspeed.runtime.activation_checkpointing import checkpointing as dsc
    except Exception as e:  # noqa: BLE001
        print(f"[wam/zero3-ckpt] deepspeed activation checkpointing unavailable ({e}); skipping")
        return
    if getattr(tuc, "_wam_ckpt_patched", False):
        return

    import torch

    def checkpoint(function, *args, use_reentrant=None, determinism_check=None,  # noqa: ARG001
                   context_fn=None, **kwargs):
        base = partial(function, **kwargs) if kwargs else function   # deepspeed ckpt takes only *args

        def fn(*a):
            # DeepSpeed's checkpoint recompute does NOT restore the outer autocast context (torch's
            # did), so fp32 LoRA params would meet bf16 activations -> dtype mismatch. Re-enter
            # autocast so forward and recompute use consistent dtypes.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return base(*a)

        return dsc.checkpoint(fn, *args)

    tuc.checkpoint = checkpoint
    tuc._wam_ckpt_patched = True
    print("[wam/zero3-ckpt] routed torch.utils.checkpoint -> deepspeed.checkpointing (ZeRO-3 aware)")


def _install_patch() -> None:
    try:
        import deepspeed
    except Exception:  # noqa: BLE001
        return
    if getattr(deepspeed, "_wam_leaf_patched", False):
        return
    try:
        from deepspeed.utils import set_z3_leaf_modules
    except Exception as e:  # noqa: BLE001
        print(f"[wam/zero3-leaf] set_z3_leaf_modules unavailable ({e}); skipping")
        return

    _orig_initialize = deepspeed.initialize

    def initialize(*args, **kwargs):
        model = kwargs.get("model")
        if model is None and len(args) >= 2:
            model = args[1]              # deepspeed.initialize(args, model, ...)
        try:
            if model is not None and hasattr(model, "modules"):
                classes = list({type(m) for m in model.modules() if type(m).__name__ in LEAF_CLASS_NAMES})
                if classes:
                    set_z3_leaf_modules(model, classes)
                    print(f"[wam/zero3-leaf] marked leaf modules: {sorted(c.__name__ for c in classes)}")
        except Exception as e:  # noqa: BLE001
            print(f"[wam/zero3-leaf] could not set leaf modules (continuing): {e}")
        return _orig_initialize(*args, **kwargs)

    deepspeed.initialize = initialize
    deepspeed._wam_leaf_patched = True
    print("[wam/zero3-leaf] patched deepspeed.initialize")


_install_patch()
_install_checkpoint_patch()
