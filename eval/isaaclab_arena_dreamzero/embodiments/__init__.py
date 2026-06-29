# Copyright (c) 2026, dreamzero-wam.
# SPDX-License-Identifier: Apache-2.0

"""Importing this package registers the Trossen embodiment with Arena's
AssetRegistry (via the @register_asset decorator on TrossenMobileAIEmbodiment).
The eval entrypoint imports it before building the env so `embodiment:
"trossen_mobile_ai"` in the jobs config resolves."""

from isaaclab_arena_dreamzero.embodiments.trossen import TrossenMobileAIEmbodiment  # noqa: F401
