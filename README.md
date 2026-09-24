# ComfyUI-MiniMaxH3RefLoRA

One `.safetensors` holding a MiniMax H3 LoRA **and** its RefMod reference(s).

A **RefLoRA** is an ordinary LoRA file with reference latents packed alongside the
weights. Standard loaders read it as a plain LoRA and ignore the extra tensors, so it
ships as a single download with nothing lost:

| Loader | What it sees |
| --- | --- |
| `Load LoRA`, `LoraLoaderModelOnly`, rgthree Power Lora Loader | the LoRA, exactly as before |
| `ComfyUI-MiniMaxH3Mod`'s RefMod loader (unmodified) | the references |
| `Load H3 RefLoRA` (this pack) | both, in one node |

The two halves can never drift apart, and there is no "which refmod went with which
epoch" problem.

## Why a LoRA *and* a reference

They are not redundant. A LoRA holds everything a hundred training images showed it — the
angles, the odd details, the nooks and crannies. A reference cannot, because references
cost tokens at generation time and the budget stretches to a handful of images, so
everything it was not shown it has to infer. The reference contributes a VAE-invariant
identity lock the weights alone do not give you. Shipping both in one file is the point.

## Install

Drop this folder into `ComfyUI/custom_nodes/` and restart. No dependencies beyond what
ComfyUI already has — the packer needs nothing at all, not even `safetensors`.

## Nodes

**Load H3 RefLoRA (LoRA + RefMod)** — `MODEL` (+ optional `CLIP`) in, patched `MODEL`,
`CLIP`, `H3_REF_MODS` and an info string out.

- `members` picks which references to load, numbered as `Inspect` shows them: `all`,
  `none`, `1`, `1,3-4`. Unselected members are never read off disk.
- `refmod_retention` is the reference strength: 1.0 fully preserved, 0.7 partially,
  0.4 attribute transfer, 0.15 weak, 0 loads none.
- `attach_to_model` puts the references on the **model line** — nothing needs wiring into
  the conditioning at all. Leave it off and use the `mods` output with an Apply node
  instead. Doing both injects every reference twice.
- `max_total_tokens` is a budget guard. Use it: references are not free.

**Apply H3 RefLoRA References** — injects `H3_REF_MODS` into a MiniMax H3 conditioning.
Present so a workflow needs nothing but this pack. If you have ComfyUI-MiniMaxH3Mod
installed, its richer *Apply H3 RefMod* (curves, scrambling, graph presets) accepts this
pack's output too — the types are deliberately interchangeable.

**Pack H3 RefLoRA** — build a container from a LoRA plus 1–8 RefMods, into
`models/loras/`. A RefMod bundle passed in is flattened into its own members.

**Inspect H3 RefLoRA** — header-only report: LoRA tensor count, dim/alpha, every member
with its kind, shape and token cost, and whether the weight half is intact.

## Console packer

`pack_reflora.py` is standard library only — no torch, no `safetensors`, no venv. It
streams in 1 MiB chunks, so packing a 150 MB LoRA costs about a megabyte of RAM and the
weights come out byte-identical to the source (verify it yourself with `verify --lora`).

```
python pack_reflora.py pack --lora minimaxh3_aneta_v1-000060 \
                            --refmod minimaxh3_aneta_v1_refmod \
                            --out models/loras/MinimaxH3/reflora/minimaxh3_aneta_v1_reflora.safetensors

python pack_reflora.py inspect <reflora>
python pack_reflora.py verify  <reflora> --lora <source lora>     # byte-compares the weights
python pack_reflora.py unpack  <reflora> --out-dir <dir>          # back to separate files
```

Bare names are resolved against `models/loras` and `models/refmods`, so you rarely need
full paths. Set `COMFYUI_MODELS_DIR` if the pack is not inside a ComfyUI install.

## Example workflows

In `examples/`, all four derived from a working MiniMax H3 turbo 8-step graph:

| File | What it shows |
| --- | --- |
| `workflow_api_minimaxh3_reflora_turbo.json` | the straight swap: one RefLoRA node replaces the RefMod loader |
| `..._feliciaday.json` | the same graph, a different subject |
| `..._model_attached.json` | `attach_to_model` — references on the model line, nothing on the conditioning |
| `..._multi.json` | a two-reference container, loading only member 1 |

Each expects the matching container in `models/loras/MinimaxH3/reflora/`.

## The format

`HYBRID_FORMAT.md` has the full spec. The short version:

```
tensors   lora_unet_*.lora_down/up/alpha     the LoRA, byte-identical to its source
          ref_0 .. ref_n                     [1,24,T,H,W] visual | [1,32,2,T] audio

header    ss_*, modelspec.*                  carried over from the source LoRA
          refmod_meta                        v5 bundle JSON — MiniMaxH3Mod's own format
          h3_hybrid                          additive marker: version, key count, sources
```

The reference half is deliberately **not** new. It is the version-5 bundle format from
ComfyUI-MiniMaxH3Mod, which addresses tensors by exact name and never enumerates the
file — so the LoRA keys are invisible to it and that pack reads our containers with no
changes at all.

## One thing to know

`bundle.save_bundle()` in ComfyUI-MiniMaxH3Mod writes **only** the `ref_i` tensors. If you
load a RefLoRA there and use its **Bundle Save** node, the weight half is dropped with no
error. (Its *Fix Config* node is safe — that path copies every tensor through.)

This pack's writer always passes non-`ref` tensors through, the loader warns when a
container's declared LoRA tensor count no longer matches what is present, and
`verify --lora` catches it outright. Re-save through **Pack H3 RefLoRA**, not through the
other pack's bundle saver.

## Credits

See `NOTICE`. The container format and the reference maths are Luisa (luisacaotica)'s
ComfyUI-MiniMaxH3Mod (MIT); the compact curve implementation follows Fizgig
(Peter Neill, Apache-2.0). Both were offered for reuse, and this pack would be a much
longer job without them.
