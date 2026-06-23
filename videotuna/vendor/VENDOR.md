# Vendor: SimpleTuner (reference submodule)

| Field | Value |
|-------|-------|
| **Path** | `videotuna/vendor/simpletuner/` (git submodule) |
| **Upstream** | https://github.com/bghira/SimpleTuner |
| **License** | Apache-2.0 |
| **Pinned commit** | `34b1fd729fd0fa86e6b085ba0f3dbc44ca8757dc` (2025-01-29) |
| **Import date** | 2025-06 (reference submodule; runtime trainer replaced) |
| **VideoTuna entrypoints** | *(none — reference only)* |
| **Runtime replacement** | `videotuna/training/flux_lora/` via `poetry run train-flux-lora` |

## Purpose

Reference-only submodule for upstream provenance. VideoTuna does **not** import this tree at runtime.
The deleted in-tree snapshot (`videotuna/third_party/flux/`) was namespace-rewritten and had two
functional patches — see [`docs/vendor/simpletuner-archive.md`](../../docs/vendor/simpletuner-archive.md).

## Update procedure

```bash
cd videotuna/vendor/simpletuner
git fetch origin
git checkout <new-sha>
cd ../../..
git add videotuna/vendor/simpletuner
# Update this file and docs/vendor/simpletuner-archive.md with the new SHA
```

Init on clone (optional):

```bash
git submodule update --init videotuna/vendor/simpletuner
```
