# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab-Arena policy adapter for the DreamZero (groot) world-action model.

This package plugs a fine-tuned DreamZero checkpoint served over the openpi
websocket protocol into NVIDIA Isaac Lab-Arena's policy-evaluation runner. It is
the DreamZero analogue of Arena's bundled ``isaaclab_arena_openpi`` (pi0) and
``isaaclab_arena_gr00t`` adapters.

It lives in the dreamzero-wam repo (not the Arena tree) so it is versioned with
the model/training code; ``eval/runner.sh`` puts it on ``PYTHONPATH`` at eval
time. Arena's ``policy_runner`` references it by dotted path, e.g.
``--policy_type isaaclab_arena_dreamzero.policy.dreamzero_remote_policy.DreamZeroRemotePolicy``.
"""
