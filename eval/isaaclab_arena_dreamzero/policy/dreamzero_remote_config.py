# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

# The DreamZero inference server defaults (see deploy/cks/scripts/runner.sh and
# the DROID eval's DZ_HOST/DZ_PORT). Port 8001 is the project convention.
DEFAULT_REMOTE_PORT = 8001

# Refetch a fresh action chunk after replaying this many steps open-loop. The
# Trossen action modality is trained with a 24-step horizon (delta_indices
# 0..23 in modality_config_trossen) but max_chunk_size=5, so the server returns
# at least 5 rows; keep the default replay <= the smallest guaranteed chunk.
DEFAULT_OPEN_LOOP_HORIZON = 5

MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class DreamZeroRemotePolicyArgs:
    """Connection + runtime config for ``DreamZeroRemotePolicy``.

    Mirrors ``isaaclab_arena_openpi.policy.pi0_remote_config.Pi0RemotePolicyArgs``
    but adds the embodiment-adapter selector and the open-loop horizon (the
    DreamZero server, unlike pi0, has no fixed per-variant horizon table).
    """

    # Which embodiment adapter translates obs/action wire format. Only
    # "trossen" ships today; "droid" can be added to mirror the upstream eval.
    embodiment_adapter: str = "trossen"
    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = DEFAULT_REMOTE_PORT
    open_loop_horizon: int = DEFAULT_OPEN_LOOP_HORIZON
