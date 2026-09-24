# H3 RefLoRA container format, version 1

A RefLoRA is one `.safetensors` holding a MiniMax H3 LoRA and one or more RefMod
references. It is not a new file format: it is an ordinary safetensors file whose tensor
dictionary happens to contain both kinds of payload, and whose header carries both kinds
of metadata.

## Layout

```
┌──────────────────┬───────────────────────────────┬────────────────────────────────┐
│ 8 bytes (u64 LE) │ N bytes UTF-8 JSON header     │ contiguous raw tensor blob     │
│ header length N  │ {"__metadata__": {...}, ...}  │ [tensor][tensor][tensor]...    │
└──────────────────┴───────────────────────────────┴────────────────────────────────┘
```

### Tensors

| Key | Shape | Purpose |
| --- | --- | --- |
| `lora_unet_<block>.lora_down.weight` | `[rank, dim]` | LoRA down projection (kohya naming) |
| `lora_unet_<block>.lora_up.weight` | `[dim, rank]` | LoRA up projection |
| `lora_unet_<block>.alpha` | scalar | LoRA alpha |
| `ref_0` … `ref_n` | `[1,24,T,H,W]` | visual reference latent (`image` T=1, or `video`) |
| `ref_i` | `[1,32,2,T]` | audio reference latent |

Reference indices are contiguous from `ref_0` and match the order of `members` in the
header. Visual `H` and `W` must be positive and even (the DiT patchifies 2×2). Up to 256
references.

### Header metadata

`__metadata__` is a flat string→string map, so LoRA and reference metadata coexist
without a namespace scheme.

| Key | Contents |
| --- | --- |
| `ss_*`, `modelspec.*` | carried over verbatim from the source LoRA |
| `refmod_meta` | JSON: a version-5 RefMod **bundle** object (below) |
| `h3_hybrid` | JSON: this format's own marker (below) |

`refmod_meta` is exactly the version-5 bundle documented in ComfyUI-MiniMaxH3Mod's
`BUNDLE_FORMAT.md` — not a variant of it:

```json
{
  "_format_version": 5,
  "kind": "bundle",
  "name": "minimaxh3_aneta_v1_reflora",
  "members": [
    {"_format_version": 4, "name": "minimaxh3_aneta_v1_refmod", "kind": "video",
     "latent_t": 23, "latent_h": 32, "latent_w": 32, "mode": "encode",
     "concept_type": "identity", "description": "...", "tags": ["23 img, 0 vid"]}
  ]
}
```

`h3_hybrid` is additive and nothing depends on it existing. It exists so a reader can
identify a RefLoRA, and detect a damaged one, from the header alone:

```json
{
  "version": 1,
  "lora": {"format": "kohya", "keys": 600, "network_dim": "16", "network_alpha": "16.0"},
  "refmod_count": 1,
  "packed_by": "ComfyUI-MiniMaxH3RefLoRA",
  "sources": {"lora": "minimaxh3_aneta_v1-000060.safetensors",
              "refmods": ["minimaxh3_aneta_v1_refmod.safetensors"]}
}
```

A reader that finds `lora.keys` disagreeing with the number of non-`ref_*` tensors is
looking at a file something stripped the weights out of.

## Why this works

**Standard LoRA loaders ignore the references.** ComfyUI's `load_lora()` walks the
*model's* key map and looks up matching entries in the LoRA dict; anything left over is
reported with `logging.warning("lora key not loaded: ...")` and discarded. Two extra
tensors produce two warning lines and nothing else. Measured on a real container: 600
LoRA tensors, 200 patches applied, weights byte-identical to the standalone file.

**The RefMod loader ignores the LoRA.** `bundle.load_bundle()` reads the header's
`refmod_meta`, then fetches `ref_0`, `ref_1`, … by exact name through `safe_open`. It
never enumerates the file, so the 600 LoRA keys are invisible to it and never leave disk.
An unmodified install reads a RefLoRA today.

This is the whole trick, and it is why the reference half deliberately reuses the
existing bundle format rather than inventing `refmod.latent`-style keys: a new key scheme
would have bought nothing and cost compatibility with every RefMod tool already shipped.

## Writing one

Packing is a header rewrite plus a byte copy — no tensor library required. Collect each
source tensor's `(dtype, shape, source file, offset, length)`, lay them out in a new data
block, emit a header pointing into it, then stream the bytes across. The LoRA weights are
therefore identical at the byte level, with no dtype round-trip to trust.

Two rules for any writer:

1. **Pass unknown tensors through.** A tool that rewrites a container must copy every
   tensor it did not come to change. Writing only `ref_i` silently destroys the weights.
2. **Renumber on removal.** Member indices are positional; dropping member 1 of 3 means
   rewriting `ref_2` as `ref_1` and the `members` array to match.

## Compatibility

Version 1 readers accept standalone RefMod files of format version 2 and 4 (a bare
`latent` tensor) as pack inputs, and flatten a version-5 bundle into its members. Older
readers that expect only a `latent` tensor need bundle support; there is no promise of
forward compatibility beyond that.
