"""The RefLoRA nodes."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import comfy.sd
import comfy.utils
import folder_paths

from . import container, inject
from .container import ContainerError
from .refmod import load_refmods, parse_members

MAX_PACK_SLOTS = 8
NONE = "(none)"

try:
    folder_paths.add_model_folder_path("refmods", os.path.join(folder_paths.models_dir, "refmods"))
except Exception:                                       # pragma: no cover - older ComfyUI
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lora_names() -> List[str]:
    return folder_paths.get_filename_list("loras")


def _refmod_names() -> List[str]:
    """Every RefMod-looking file under the refmods folder(s), newest search dir first."""
    names: List[str] = []
    seen = set()
    roots = []
    try:
        roots.extend(folder_paths.get_folder_paths("refmods"))
    except Exception:                                   # pragma: no cover
        pass
    fallback = os.path.join(folder_paths.models_dir, "refmods")
    if fallback not in roots:
        roots.append(fallback)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = sorted(s for s in subdirs if s not in (".git", "__pycache__",
                                                               "graph_presets"))
            for filename in sorted(files):
                if not filename.endswith(".safetensors"):
                    continue
                name = os.path.relpath(os.path.join(directory, filename), root).replace("\\", "/")
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def _resolve_refmod(name: str) -> str:
    roots = []
    try:
        roots.extend(folder_paths.get_folder_paths("refmods"))
    except Exception:                                   # pragma: no cover
        pass
    fallback = os.path.join(folder_paths.models_dir, "refmods")
    if fallback not in roots:
        roots.append(fallback)
    for root in roots:
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            root_real = os.path.realpath(root)
            if os.path.commonpath((root_real, os.path.realpath(candidate))) == root_real:
                return candidate
    raise ContainerError(f"RefMod '{name}' not found under models/refmods/.")


def _report(info: Dict[str, Any]) -> str:
    lines = [f"{os.path.basename(info['path'])}  ({info['size'] / 1e6:.1f} MB)",
             f"  LoRA tensors : {info['lora_keys']}"
             + (f"  dim {info['network_dim']} / alpha {info['network_alpha']}"
                if info.get("network_dim") else "")]
    if info["members"]:
        lines.append(f"  references   : {len(info['members'])}"
                     f"  ({info['total_tokens']} tokens total)")
        for member in info["members"]:
            shape = "x".join(str(n) for n in member["shape"])
            lines.append(f"    {member['index'] + 1}. {member['name']}  [{member['kind']}]"
                         f"  {shape}  {member['tokens']} tokens"
                         + (f"  -- {member['concept_type']}"
                            if member["concept_type"] != "generic" else ""))
    else:
        lines.append("  references   : none")
    for problem in info.get("problems", []):
        lines.append(f"  !! {problem}")
    return "\n".join(lines)


def _prompt_hint(mods) -> str:
    return "; ".join(f"{m.concept_type}: {m.description}" for m in mods if m.description)


# ---------------------------------------------------------------------------
# Node: load a RefLoRA
# ---------------------------------------------------------------------------

class MiniMaxH3RefLoRALoader:
    """Load one .safetensors as both a LoRA and its reference(s).

    The LoRA half is patched into MODEL/CLIP exactly as an ordinary LoRA loader would.
    The reference half comes out as H3_REF_MODS for an Apply node, and/or is attached
    straight to the model line when ``attach_to_model`` is on.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model the LoRA weights patch."}),
                "reflora_name": (_lora_names(), {
                    "tooltip": "A RefLoRA container in models/loras/. An ordinary LoRA "
                               "also works -- it simply has no references to load."}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                    "step": 0.01, "tooltip": "LoRA weight strength on the model. 0 skips "
                                             "patching entirely."}),
                "refmod_retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Reference strength. 1.0 = fully preserved, 0.7 = partially "
                               "preserved, 0.4 = attribute transfer, 0.15 = weak. 0 loads "
                               "no references at all."}),
                "members": ("STRING", {"default": "all",
                    "tooltip": "Which references to load, numbered as the Inspect node "
                               "shows them: 'all', 'none', or a list like '1' or '1,3-4'. "
                               "Unselected members are never read off disk."}),
                "attach_to_model": ("BOOLEAN", {"default": False,
                    "tooltip": "ON: the references ride the MODEL output, so nothing has "
                               "to be wired into the conditioning -- one node, one wire. "
                               "OFF (default): use the H3_REF_MODS output with an Apply "
                               "node. Do not do both, or every reference is injected twice."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional. MiniMax H3 LoRAs are lora_unet_* "
                                             "only, so this is usually left unconnected."}),
                "lora_strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                    "step": 0.01}),
                "components": (["All", "Visual", "Audio"], {"default": "All",
                    "tooltip": "Filter references by modality before loading them."}),
                "max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 1048576,
                    "tooltip": "Reference token budget; 0 disables it. References are not "
                               "free -- a full-resolution identity reference can cost "
                               "several thousand tokens in every DiT block."}),
                "show_info": ("BOOLEAN", {"default": True,
                    "tooltip": "Print the container's contents to the console on load."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "H3_REF_MODS", "STRING")
    RETURN_NAMES = ("model", "clip", "mods", "info")
    FUNCTION = "load"
    CATEGORY = "MiniMax-H3/reflora"
    DESCRIPTION = ("Load a RefLoRA container: LoRA weights onto MODEL/CLIP and the "
                   "embedded reference(s) out as H3_REF_MODS.")

    def load(self, model, reflora_name, lora_strength=1.0, refmod_retention=1.0,
             members="all", attach_to_model=False, clip=None, lora_strength_clip=1.0,
             components="All", max_total_tokens=0, show_info=True):
        path = folder_paths.get_full_path_or_raise("loras", reflora_name)
        info = container.describe(path)
        if show_info:
            print("[RefLoRA] " + _report(info).replace("\n", "\n[RefLoRA] "))

        declared = (info.get("hybrid") or {}).get("lora", {}).get("keys")
        if declared is not None and declared != info["lora_keys"]:
            print(f"[RefLoRA] WARNING: this container declares {declared} LoRA tensors but "
                  f"only {info['lora_keys']} are present. It was probably re-saved by a "
                  "tool that keeps only reference tensors; the weight half is gone.")

        out_model, out_clip = model, clip
        if lora_strength != 0.0 or (clip is not None and lora_strength_clip != 0.0):
            state = comfy.utils.load_torch_file(path, safe_load=True)
            for key in [k for k in state if k.startswith("ref_")]:
                state.pop(key)                          # keeps the loader's log quiet
            if state:
                out_model, out_clip = comfy.sd.load_lora_for_models(
                    model, clip, state, lora_strength, lora_strength_clip)
                if out_clip is None:
                    out_clip = clip
            else:
                print(f"[RefLoRA] '{reflora_name}' carries no LoRA weights; "
                      "passing the model through unchanged.")

        mods: List[Any] = []
        if refmod_retention > 0 and info["members"]:
            selected = parse_members(members, len(info["members"]))
            mods = load_refmods(path, selection=components, members=selected)

        rows: List[Tuple[Any, float]] = [(mod, 1.0) for mod in mods]
        total = sum(mod.token_count for mod in mods)
        if mods:
            print(f"[RefLoRA] {len(mods)} reference(s) loaded @ retention "
                  f"{refmod_retention:.2f}: "
                  + ", ".join(f"{m.name}({m.token_count}tok)" for m in mods)
                  + f"  = {total} tokens")
            if max_total_tokens and total > max_total_tokens:
                raise ValueError(
                    f"Reference token budget exceeded: {total} > {max_total_tokens}. "
                    "Select fewer members, or raise max_total_tokens.")
        elif info["members"]:
            print("[RefLoRA] references present but none selected "
                  "(retention 0, members filter, or components filter).")

        if attach_to_model:
            blocks = inject.ref_blocks(rows, refmod_retention,
                                       max_total_tokens=max_total_tokens)
            out_model = inject.into_model(out_model, blocks)
            print(f"[RefLoRA] {len(blocks)} reference block(s) attached to the model line; "
                  "do not also wire 'mods' into an Apply node.")

        summary = _report(info)
        hint = _prompt_hint(mods)
        if hint:
            summary += f"\nprompt hint: {hint}"
        # retention is folded in here so an Apply node can use a flat 1.0 and still honour
        # the dial on this node.
        rows = [(mod, float(refmod_retention)) for mod, _ in rows]
        return (out_model, out_clip, rows, summary)


# ---------------------------------------------------------------------------
# Node: apply references to a conditioning
# ---------------------------------------------------------------------------

class MiniMaxH3RefLoRAApply:
    """Inject references into a MiniMax H3 conditioning.

    Present so a RefLoRA workflow needs nothing but this pack. If ComfyUI-MiniMaxH3Mod is
    installed, its richer 'Apply H3 RefMod' (curves, scrambling, graph presets) accepts
    this pack's H3_REF_MODS output too.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "From MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo."}),
                "mods": ("H3_REF_MODS", {"tooltip": "From Load H3 RefLoRA."}),
                "retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Master reference strength, multiplied with each row's own "
                               "strength. 1.0 = fully preserved, 0.4 = attribute transfer."}),
            },
            "optional": {
                "max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 1048576}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax-H3/reflora"

    def apply(self, conditioning, mods, retention=1.0, max_total_tokens=0):
        blocks = inject.ref_blocks(mods, retention, max_total_tokens=max_total_tokens)
        out = inject.into_conditioning(conditioning, blocks)
        print(f"[RefLoRA Apply] retention={retention:.2f}, "
              f"{len(blocks)} reference block(s) injected.")
        return (out,)


# ---------------------------------------------------------------------------
# Node: build a RefLoRA
# ---------------------------------------------------------------------------

class MiniMaxH3RefLoRAPack:
    """Pack a LoRA and 1-8 RefMods into one RefLoRA container in models/loras/."""

    @classmethod
    def INPUT_TYPES(cls):
        refmods = [NONE] + _refmod_names()
        required = {
            "lora_name": (_lora_names(), {"tooltip": "The LoRA to pack."}),
            "output_name": ("STRING", {"default": "",
                "tooltip": "Output stem. Empty = the LoRA's name with '_reflora' appended."}),
            "subfolder": ("STRING", {"default": "MinimaxH3/reflora",
                "tooltip": "Destination inside models/loras/."}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }
        for slot in range(1, MAX_PACK_SLOTS + 1):
            required[f"refmod_{slot}"] = (refmods, {
                "tooltip": f"Reference {slot}, or {NONE}. A RefMod bundle is flattened "
                           "into its own members."})
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "report")
    FUNCTION = "pack"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax-H3/reflora"

    def pack(self, lora_name, output_name="", subfolder="MinimaxH3/reflora",
             overwrite=False, **kwargs):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        refmods = []
        for slot in range(1, MAX_PACK_SLOTS + 1):
            name = kwargs.get(f"refmod_{slot}")
            if name and name != NONE:
                refmods.append(_resolve_refmod(name))
        if not refmods:
            raise ContainerError("Select at least one RefMod to pack.")

        stem = (output_name or "").strip()
        if not stem:
            stem = os.path.splitext(os.path.basename(lora_name))[0] + "_reflora"
        if stem.endswith(".safetensors"):
            stem = stem[: -len(".safetensors")]
        root = folder_paths.get_folder_paths("loras")[0]
        out_path = os.path.join(root, subfolder.strip().strip("/\\"), stem + ".safetensors")

        result = container.pack(lora_path, refmods, out_path, name=stem, force=overwrite)
        report = _report(container.describe(out_path))
        print(f"[RefLoRA Pack] wrote {out_path} ({result['size'] / 1e6:.1f} MB)")
        return {"ui": {"text": [out_path]}, "result": (out_path, report)}


# ---------------------------------------------------------------------------
# Node: look inside a container
# ---------------------------------------------------------------------------

class MiniMaxH3RefLoRAInspect:
    """Report what a .safetensors in models/loras/ actually contains, header only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reflora_name": (_lora_names(),),
                             "verify_bytes": ("BOOLEAN", {"default": False,
                                 "tooltip": "Also check that the declared LoRA tensor "
                                            "count still matches what is in the file."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "inspect"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax-H3/reflora"

    def inspect(self, reflora_name, verify_bytes=False):
        path = folder_paths.get_full_path_or_raise("loras", reflora_name)
        info = container.verify(path) if verify_bytes else container.describe(path)
        report = _report(info)
        if not info.get("is_reflora", False):
            report += "\n  (not a RefLoRA -- " + (
                "no references embedded)" if info["lora_keys"] else "no LoRA weights)")
        print("[RefLoRA Inspect]\n" + report)
        return {"ui": {"text": [report]}, "result": (report,)}


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3RefLoRALoader": MiniMaxH3RefLoRALoader,
    "MiniMaxH3RefLoRAApply": MiniMaxH3RefLoRAApply,
    "MiniMaxH3RefLoRAPack": MiniMaxH3RefLoRAPack,
    "MiniMaxH3RefLoRAInspect": MiniMaxH3RefLoRAInspect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3RefLoRALoader": "Load H3 RefLoRA (LoRA + RefMod)",
    "MiniMaxH3RefLoRAApply": "Apply H3 RefLoRA References",
    "MiniMaxH3RefLoRAPack": "Pack H3 RefLoRA",
    "MiniMaxH3RefLoRAInspect": "Inspect H3 RefLoRA",
}
