# Evaluation (placeholder)

Deferred — working eval code will be added here by the W&B/Encord team.

**Contract:** an eval run loads a fine-tuned checkpoint via
`run.use_artifact("encord-wb-webinar/wandb-registry-model/dreamzero-droid-pickplace-lora:<version>")`,
runs it under a fixed test scenario, and logs the results linked to that artifact — so every training
variant (random vs captioned vs curated dataset) is comparable side-by-side, with lineage from
dataset → training run → checkpoint → eval.
