# W&B API key secret

Every pod that talks to W&B reads the key from a `wandb-api-key` secret via `envFrom`, so the key is
never written into a manifest. It usually already exists in the `dreamzero` namespace:

```bash
kubectl -n dreamzero get secret wandb-api-key
```

If it's missing, create it **out-of-band** (never commit the key). Use a personal W&B key that has access
to the **`encord-wb-physical-ai`** entity and the **`encord-wb-webinar`** org Registry:

```bash
export WANDB_API_KEY=...
kubectl -n dreamzero create secret generic wandb-api-key \
  --from-literal=WANDB_API_KEY="$WANDB_API_KEY"
```

Verify the key can reach the target entity (any pod that runs `wandb` will print
`Currently logged in as: <user> (encord-wb-physical-ai)` on init).
