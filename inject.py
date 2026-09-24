"""The two ways a reference reaches the model.

Route A, conditioning: append the ref blocks to the conditioning's reference list.  Works
with the built-in ComfyUI ``CONDITIONING`` from ``MiniMaxH3ImageToVideo`` /
``MiniMaxH3ReferenceToVideo`` (field ``minimax_refs``) and with the ComfyUI-MiniMaxH3
pack's ``MINIMAX_H3_COND`` dataclass (field ``refs``).

Route B, model: clone the model and register an ``OUTER_SAMPLE`` wrapper that injects the
refs into the guider's conditioning right before sampling, then puts the originals back.
This is the "nothing on the conditioning noodle" path -- the model line carries both the
LoRA weights and the references, so a RefLoRA is one node with one wire.

Credit: route B is the technique in Luisa (luisacaotica)'s ComfyUI-MiniMaxH3Mod
``continuum_bridge.py``, MIT.  It uses only public ComfyUI patcher API and holds
no global state, so it survives Continuum chunking and cancellation alike.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from comfy.patcher_extension import WrappersMP

BRIDGE_KEY = "minimax_h3_reflora_bridge"


def ref_blocks(mods: Sequence[Tuple[Any, float]], retention: float = 1.0,
               curve: Optional[Tuple[str, str, float]] = None,
               max_total_tokens: int = 0) -> List[Dict[str, Any]]:
    """Ref blocks for a list of ``(mod, strength)`` rows, scaled by strength x retention."""
    factor = RETENTION_ALIASES.get(retention, retention) if isinstance(retention, str) \
        else float(retention)
    blocks: List[Dict[str, Any]] = []
    used = 0
    for mod, strength in mods or []:
        effective = min(1.0, max(0.0, float(strength) * float(factor)))
        block = mod.ref_block(effective, curve=curve)
        if block is None:
            continue
        if max_total_tokens and used + mod.token_count > max_total_tokens:
            raise ValueError(
                f"Reference token budget exceeded: '{mod.name}' would take the total to "
                f"{used + mod.token_count}, over the {max_total_tokens} limit. Raise "
                "max_total_tokens, drop a member, or set its strength to 0.")
        used += mod.token_count
        block["refmod"] = True
        blocks.append(block)
    return blocks


RETENTION_ALIASES = {"fully_preserved": 1.0, "partially_preserved": 0.7,
                     "attribute_transfer": 0.4, "weak_reference": 0.15}


def into_conditioning(conditioning: Any, blocks: Sequence[Dict[str, Any]]) -> Any:
    """Route A. Returns the same conditioning type it was given."""
    if not blocks:
        return conditioning
    if isinstance(conditioning, list):
        out = []
        for entry in conditioning:
            values = dict(entry[1])
            values["minimax_refs"] = list(values.get("minimax_refs") or []) + list(blocks)
            out.append([entry[0], values])
        return out
    return replace(conditioning, refs=list(conditioning.refs) + list(blocks))


def into_model(model: Any, blocks: Sequence[Dict[str, Any]], enabled: bool = True) -> Any:
    """Route B. Returns a clone carrying the refs; never mutates the model passed in."""
    model = model.clone()
    model.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, BRIDGE_KEY)
    if not enabled or not blocks:
        return model
    refs = tuple(dict(block) for block in blocks)

    def wrapper(executor, *args, **kwargs):
        guider = executor.class_obj
        original = guider.conds
        guider.conds = {
            key: [dict(entry,
                       minimax_refs=list(entry.get("minimax_refs") or [])
                       + [dict(ref) for ref in refs])
                  for entry in entries]
            for key, entries in original.items()
        }
        try:
            return executor(*args, **kwargs)
        finally:
            guider.conds = original

    model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, BRIDGE_KEY, wrapper)
    return model
