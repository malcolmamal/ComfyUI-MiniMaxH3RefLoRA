"""The reference half of a RefLoRA: a latent plus the maths that weakens it.

``RefLoRAMod`` is deliberately duck-compatible with ``H3RefMod`` from
ComfyUI-MiniMaxH3Mod: same attribute names, same ``ref_block(strength, curve=...)``
signature, same ``token_count``.  That means our loader's ``H3_REF_MODS`` output feeds
*either* our Apply node or that pack's -- with no import across ``custom_nodes/`` and no
hard dependency in either direction.

Credit: the strength and curve formulas here are ports of Luisa (luisacaotica)'s
ComfyUI-MiniMaxH3Mod ``core.py`` (MIT) and Fizgig's ``fizgig.minimax.refmod_apply``
(Apache-2.0), both offered for reuse.  Verified bit-identical to the former at strength
0.4 on a real reference.  See NOTICE.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .container import REFMOD_KEY, read_header, tensors_in, header_metadata

CURVE_DIRECTIONS = ("constant", "concept_at_start", "concept_at_middle",
                    "concept_at_end", "concept_at_ends")
CURVE_SHAPES = ("linear", "ease", "sigmoid", "tanh", "quadratic", "cubic",
                "exponential", "stair", "elastic", "bump", "dip")
RETENTION = {"fully_preserved": 1.0, "partially_preserved": 0.7,
             "attribute_transfer": 0.4, "weak_reference": 0.15}

CurveSpec = Tuple[str, str, float]


# ---------------------------------------------------------------------------
# weakening a reference
# ---------------------------------------------------------------------------

def blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy low-pass of a latent -- the target a reference is mixed toward below full
    strength.

    Not toward zero and not toward noise, both of which look wrong for good reasons: a
    VAE latent's channels are correlated, so scaling toward zero leaves the normalized
    distribution (grey output), and mixing toward iid noise reads as real-but-garbled
    content and decodes as an actual wrong texture.  A blurred copy stays on the latent
    manifold while still discarding the detail that makes a reference strong, which is
    what "weaker reference" should look like.
    """
    if z.dim() == 4:                                   # audio [1, 32, 2, T]
        b, c, stereo, t = z.shape
        if t <= 1:
            return z
        flat = z.reshape(b * c * stereo, 1, t).float()
        down = F.adaptive_avg_pool1d(flat, max(1, t // factor))
        return F.interpolate(down, size=t, mode="linear",
                             align_corners=False).reshape_as(z).to(z.dtype)
    if z.dim() != 5:
        raise ValueError(f"Unsupported RefMod latent shape: {tuple(z.shape)}")
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    down = F.adaptive_avg_pool3d(z.float(), (t, max(1, h // factor), max(1, w // factor)))
    return F.interpolate(down, size=(t, h, w), mode="trilinear",
                         align_corners=False).to(z.dtype)


def ease(shape: str, x: float) -> float:
    if shape == "ease":
        return x * x * (3.0 - 2.0 * x)
    if shape == "sigmoid":
        return 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
    if shape == "tanh":
        return 0.5 * (math.tanh(8.0 * (x - 0.5)) + 1.0)
    if shape == "quadratic":
        return x * x
    if shape == "cubic":
        return x * x * x
    if shape == "exponential":
        return 2.0 ** x - 1.0
    if shape == "stair":
        return min(1.0, math.floor(x * 4) / 3.0)
    if shape == "elastic":
        if x <= 0.0 or x >= 1.0:
            return float(max(0.0, min(1.0, x)))
        return 2.0 ** (-10.0 * x) * math.sin((x * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0
    if shape == "bump":
        return 1.0 - abs(2.0 * x - 1.0)
    if shape == "dip":
        return abs(2.0 * x - 1.0)
    return x                                            # "linear" and anything unknown


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def curve_value_at(spec: Optional[CurveSpec], x: float) -> float:
    """The multiplier of a (direction, shape, value) curve at progress ``x`` in [0, 1]."""
    if not spec:
        return 1.0
    direction, shape, value = spec[0], spec[1], float(spec[2])
    if direction == "constant":
        return value if shape == "linear" else _clamp01(value * ease(shape, x))
    y = ease(shape, x)
    if direction == "concept_at_end":
        return _clamp01(value * y)
    if direction == "concept_at_start":
        return _clamp01(value * (1.0 - y))
    if direction == "concept_at_middle":
        return _clamp01(value * (1.0 - abs(2.0 * y - 1.0)))
    if direction == "concept_at_ends":
        return _clamp01(value * abs(2.0 * y - 1.0))
    return 1.0


def curve_strengths(spec: Optional[CurveSpec], t: int) -> Optional[List[float]]:
    """Per-frame multipliers, or None when the curve has nothing to shape."""
    if t <= 1 or not spec:
        return None
    direction, shape = spec[0], spec[1]
    if direction not in CURVE_DIRECTIONS:
        return None
    if direction == "constant" and shape == "linear":
        value = float(spec[2])
        return None if value >= 1.0 else [value] * t
    return [curve_value_at(spec, i / (t - 1)) for i in range(t)]


# ---------------------------------------------------------------------------
# a loaded reference
# ---------------------------------------------------------------------------

@dataclass
class RefLoRAMod:
    """One reference out of a container, in the shape the DiT's ref path expects."""

    name: str
    kind: str
    latent: torch.Tensor
    latent_h: int = 0
    latent_w: int = 0
    latent_t: int = 1
    mode: str = "encode"
    source: str = ""
    source_shape: str = ""
    pool: str = ""
    optimize_steps: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""
    concept_type: str = "generic"
    config: Dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 32000
    path: str = ""
    bundle_index: int = -1

    @property
    def token_count(self) -> int:
        if self.kind == "audio":
            return 2 * self.latent_t
        return self.latent_t * (self.latent_h // 2) * (self.latent_w // 2)

    def ref_block(self, strength: float = 1.0,
                  curve: Optional[CurveSpec] = None) -> Optional[Dict[str, Any]]:
        """The ref block dict the model's packed-sequence layout consumes.

        ``strength <= 0`` injects nothing at all (no tokens, no cost).
        """
        strength = float(strength)
        if strength <= 0.0:
            return None
        latent = self.latent
        per_frame = curve_strengths(curve, self.latent_t) if curve is not None else None
        if per_frame is not None:
            weights = [_clamp01(strength * value) for value in per_frame]
            if any(value < 1.0 for value in weights):
                shape = ((1, 1, 1, self.latent_t) if self.kind == "audio"
                         else (1, 1, self.latent_t, 1, 1))
                scale = latent.new_tensor(weights).view(shape)
                latent = scale * latent + (1.0 - scale) * blur_latent(latent)
        elif strength < 1.0:
            latent = strength * latent + (1.0 - strength) * blur_latent(latent)

        if self.kind == "audio":
            return {"kind": "audio", "ref_audio_t": self.latent_t, "audio_latent": latent,
                    "refmod": True}
        block: Dict[str, Any] = {"kind": self.kind, "latent_h": self.latent_h,
                                 "latent_w": self.latent_w, "latent": latent,
                                 "refmod": True}
        if self.kind == "video":
            block["latent_t"] = self.latent_t
            block["ref_audio_t"] = 0
            block["audio_latent"] = None
        return block

    def metadata(self) -> Dict[str, Any]:
        meta = {"name": self.name, "kind": self.kind, "latent_h": self.latent_h,
                "latent_w": self.latent_w, "latent_t": self.latent_t, "mode": self.mode,
                "source": self.source, "source_shape": self.source_shape, "pool": self.pool,
                "optimize_steps": self.optimize_steps, "tags": self.tags,
                "description": self.description, "concept_type": self.concept_type,
                "sample_rate": self.sample_rate, "_format_version": 4}
        if self.config:
            meta["refmod_config"] = json.dumps(self.config)
        return meta

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any], latent: torch.Tensor,
                      path: str = "", bundle_index: int = -1) -> "RefLoRAMod":
        raw = meta.get("refmod_config")
        try:
            config = json.loads(raw) if isinstance(raw, str) else {}
        except ValueError:
            config = {}
        kind = str(meta.get("kind", "video"))
        audio = kind == "audio"
        return cls(
            name=str(meta.get("name", f"ref_{max(bundle_index, 0)}")),
            kind=kind,
            latent=latent,
            latent_h=0 if audio else int(latent.shape[3]),
            latent_w=0 if audio else int(latent.shape[4]),
            latent_t=int(latent.shape[-1] if audio else latent.shape[2]),
            mode=str(meta.get("mode", "encode")),
            source=str(meta.get("source", "")),
            source_shape=str(meta.get("source_shape", "")),
            pool=str(meta.get("pool", "")),
            optimize_steps=int(meta.get("optimize_steps", 0)),
            tags=list(meta.get("tags", []) or []),
            description=str(meta.get("description", "") or ""),
            concept_type=str(meta.get("concept_type", "generic") or "generic"),
            config=config if isinstance(config, dict) else {},
            sample_rate=int(meta.get("sample_rate", 32000)),
            path=path,
            bundle_index=bundle_index,
        )


def load_refmods(path: str, selection: str = "All",
                 members: Optional[Sequence[int]] = None,
                 device: str = "cpu") -> List[RefLoRAMod]:
    """Read the reference half of a container.

    Tensors are fetched by name, so a member excluded by ``selection`` or ``members`` is
    never read off disk -- selecting one reference out of a 20-member container costs one
    member's worth of I/O, not twenty.
    """
    if selection not in ("All", "Visual", "Audio"):
        raise ValueError("Components must be All, Visual or Audio.")
    header, _ = read_header(path)
    raw = header_metadata(header).get(REFMOD_KEY)
    if not raw:
        return []
    meta = json.loads(raw)
    bundle = meta.get("kind") == "bundle"
    listed = (meta.get("members") or []) if bundle else [meta]
    available = set(tensors_in(header))

    wanted = None if members is None else {int(i) for i in members}
    out: List[RefLoRAMod] = []
    with safe_open(path, framework="pt", device=device) as handle:
        for index, member in enumerate(listed):
            if wanted is not None and index not in wanted:
                continue
            kind = str(member.get("kind", "video"))
            audio = kind == "audio"
            if (selection == "Visual" and audio) or (selection == "Audio" and not audio):
                continue
            key = f"ref_{index}" if bundle else "latent"
            if key not in available:
                continue
            latent = handle.get_tensor(key).clone()
            out.append(RefLoRAMod.from_metadata(member, latent, path, index if bundle else -1))
    return out


def parse_members(spec: str, total: int) -> Optional[List[int]]:
    """``"all"``/empty -> None (everything); ``"none"`` -> []; ``"1,3-4"`` -> [0, 2, 3].

    Written 1-based because that is how the members are numbered in the node's own
    report and in the Inspect node's output.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "all", "*"):
        return None
    if spec == "none":
        return []
    picked: List[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                start, end = int(low), int(high)
            except ValueError as exc:
                raise ValueError(f"Unreadable member range {part!r}.") from exc
            picked.extend(range(start, end + 1))
        else:
            try:
                picked.append(int(part))
            except ValueError as exc:
                raise ValueError(f"Unreadable member number {part!r}.") from exc
    out = sorted({n - 1 for n in picked})
    bad = [n + 1 for n in out if not 0 <= n < total]
    if bad:
        raise ValueError(f"Member(s) {bad} do not exist; the file has {total}.")
    return out
