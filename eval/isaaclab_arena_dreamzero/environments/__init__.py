# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Importing this package registers the Trossen pick-and-place environment (and its
raised work-table asset) with Arena's registries. The eval entrypoints import it
after the sim app launches so `environment: "trossen_pick_and_place"` resolves."""

from isaaclab_arena_dreamzero.environments.trossen_pick_and_place import (  # noqa: F401
    TrossenPickAndPlaceEnvironment,
    TrossenWorkTable,
)
