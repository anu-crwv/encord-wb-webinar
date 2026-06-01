# PVCs and namespace

These are already provisioned on the cluster; listed here for reference / re-creation.

| Resource | Spec |
|---|---|
| namespace | `dreamzero` |
| PVC `dreamzero-data` | RWX, `shared-vast`, 750Gi → mounted at `/data` (datasets, artifact cache, staged repo, per-arch pip homes) |
| PVC `dreamzero-checkpoints` | RWX, `shared-vast`, 500Gi → mounted at `/checkpoints` (base model weights, run outputs, rendezvous flags) |

On-PVC layout used by this harness:

```
/checkpoints/wam/models/<model>           # base model weights (reference artifacts point here)
/checkpoints/wam/runs/<run>/              # training output dirs + checkpoints
/checkpoints/wam/rendezvous/<run>.ready   # multi-node ready flag (rank0 -> rank1)
/data/wam/datasets/droid_pickplace_v0     # built dataset subset
/data/wam/artifact_cache/droid-pickplace  # dataset artifact download (training input)
/data/src/dreamzero-wam                   # this repo, staged for the training pods
/data/.home-amd64, /data/.home-gh200-<idx># per-arch pip --user homes (kept separate by arch)
```

Recreate a PVC if needed:
```bash
kubectl -n dreamzero apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: dreamzero-data, namespace: dreamzero}
spec:
  accessModes: ["ReadWriteMany"]
  storageClassName: shared-vast
  resources: {requests: {storage: 750Gi}}
EOF
```
