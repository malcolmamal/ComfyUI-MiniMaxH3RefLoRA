"""ComfyUI-MiniMaxH3RefLoRA — one .safetensors holding a LoRA and its RefMod(s).

A "RefLoRA" is an ordinary MiniMax H3 LoRA with reference latents packed into the same
file.  Standard LoRA loaders (Load LoRA, LoraLoaderModelOnly, rgthree's Power Lora
Loader) read it as a plain LoRA and ignore the extra tensors, so it can be shared as one
download with no loss of compatibility.  This pack's loader reads both halves at once.

Nodes
─────
  Load H3 RefLoRA      — one file -> MODEL + CLIP patched, references out as H3_REF_MODS
  Apply H3 RefLoRA     — inject those references into a MiniMax H3 conditioning
  Pack H3 RefLoRA      — build a container from a LoRA + 1-8 RefMods
  Inspect H3 RefLoRA   — header-only report: LoRA tensors, members, token cost

Console packer: ``python pack_reflora.py pack --lora … --refmod …`` (see --help).

The reference half is the version-5 bundle format from Luisa (luisacaotica)'s
ComfyUI-MiniMaxH3Mod, not a new format, so an unmodified install of that pack reads the
references straight out of a RefLoRA.  See NOTICE for credits.
"""

__author__ = "Malcolm Reynolds KS"
__version__ = "0.1.0"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
