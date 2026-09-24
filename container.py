"""H3 RefLoRA container: read/write a .safetensors holding a LoRA *and* its RefMods.

A safetensors file is ``[u64 header length][JSON header][raw tensor blob]``, so packing
a LoRA together with reference latents is a header rewrite plus a byte copy.  This module
does exactly that with the standard library only -- no torch, no safetensors.

Two reasons it is written this way rather than ``save_file({**lora, **refs})``:

  * the LoRA weights come out byte-identical to the source.  Deserializing 150 MB of
    BF16 into tensors and re-serializing them is a round-trip this never has to trust.
  * it streams in 1 MiB chunks, so packing a 150 MB LoRA costs ~1 MiB of RAM, and the
    same code runs anywhere Python does (the sd-backend script environment has neither
    torch nor safetensors installed).

Format -- "H3 RefLoRA v1", one file in ComfyUI/models/loras/:

    tensors   lora_unet_* / lora_te_* ...      the LoRA, untouched
              ref_0 .. ref_n                   [1,24,T,H,W] visual | [1,32,2,T] audio

    header    ss_*, modelspec.*                carried over from the source LoRA
              refmod_meta                      v5 bundle JSON (the RefMod members)
              h3_hybrid                        our additive marker (see HYBRID_KEY)

The RefMod half is deliberately *not* a new format: it is the version-5 bundle layout
documented in ComfyUI-MiniMaxH3Mod's BUNDLE_FORMAT.md.  Because that pack's loader
addresses tensors by exact name (``ref_0``, ``ref_1``, ...) and never enumerates the
file, the LoRA keys are invisible to it -- an unmodified install reads our containers.

Credit: the v5 bundle format and the RefMod metadata schema are Luisa (luisacaotica)'s
ComfyUI-MiniMaxH3Mod, MIT.  See NOTICE.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

REFMOD_KEY = "refmod_meta"
HYBRID_KEY = "h3_hybrid"
BUNDLE_VERSION = 5
MEMBER_VERSION = 4
HYBRID_VERSION = 1
CHUNK = 1 << 20
MAX_MEMBERS = 256

# A member's metadata is the pack's; these are the fields worth carrying and nothing here
# depends on the exact set -- unknown keys ride along untouched.
VISUAL_KINDS = ("image", "video")


class ContainerError(ValueError):
    """Raised for anything malformed enough that writing would produce a bad file."""


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def read_header(path: str) -> Tuple[Dict[str, Any], int]:
    """The parsed JSON header and the byte offset where the tensor blob starts."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ContainerError(f"{path}: not a safetensors file (too short).")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length < (1 << 28):
            raise ContainerError(f"{path}: implausible safetensors header length {length}.")
        blob = handle.read(length)
    if len(blob) != length:
        raise ContainerError(f"{path}: truncated safetensors header.")
    try:
        header = json.loads(blob)
    except ValueError as exc:
        raise ContainerError(f"{path}: unreadable safetensors header ({exc}).") from exc
    if not isinstance(header, dict):
        raise ContainerError(f"{path}: safetensors header is not an object.")
    return header, 8 + length


def tensors_in(header: Dict[str, Any]) -> Dict[str, Any]:
    """Tensor entries only -- ``__metadata__`` is not a tensor."""
    return {k: v for k, v in header.items() if k != "__metadata__"}


def header_metadata(header: Dict[str, Any]) -> Dict[str, str]:
    meta = header.get("__metadata__") or {}
    return dict(meta) if isinstance(meta, dict) else {}


def token_count(kind: str, shape: Sequence[int]) -> int:
    """Reference tokens this latent injects into the packed sequence.

    Visual latents are patchified 2x2 per latent frame; audio rides two rows per step.
    """
    dims = list(shape)
    if kind == "audio":
        return 2 * int(dims[-1])
    return int(dims[2]) * (int(dims[3]) // 2) * (int(dims[4]) // 2)


def validate_latent(kind: str, shape: Sequence[int], where: str) -> None:
    """The same layout checks the node pack's load_bundle applies, done on the header."""
    dims = [int(n) for n in shape]
    if kind == "audio":
        if len(dims) != 4 or dims[:3] != [1, 32, 2] or dims[3] < 1:
            raise ContainerError(f"{where}: audio RefMod must be [1,32,2,T], got {dims}.")
        return
    if kind not in VISUAL_KINDS:
        raise ContainerError(f"{where}: unknown RefMod kind {kind!r}.")
    if len(dims) != 5 or dims[:2] != [1, 24]:
        raise ContainerError(f"{where}: visual RefMod must be [1,24,T,H,W], got {dims}.")
    if any(n <= 0 for n in dims[2:]):
        raise ContainerError(f"{where}: visual RefMod has a zero dimension {dims}.")
    if dims[3] % 2 or dims[4] % 2:
        raise ContainerError(f"{where}: visual RefMod H and W must be even, got {dims}.")
    if kind == "image" and dims[2] != 1:
        raise ContainerError(f"{where}: an 'image' RefMod must have exactly one latent frame.")


def _members_of(path: str) -> List[Tuple[Dict[str, Any], str, List[int]]]:
    """Every reference in a RefMod file as (metadata, tensor key, shape).

    Accepts a standalone mod (a bare ``latent`` tensor, format version 2 or 4) and a v5
    bundle alike, so a bundle passed in is flattened into its members.
    """
    header, _ = read_header(path)
    meta_raw = header_metadata(header).get(REFMOD_KEY)
    if not meta_raw:
        raise ContainerError(f"{path}: no '{REFMOD_KEY}' in the header -- not a RefMod.")
    try:
        meta = json.loads(meta_raw)
    except ValueError as exc:
        raise ContainerError(f"{path}: unreadable RefMod metadata ({exc}).") from exc
    if not isinstance(meta, dict):
        raise ContainerError(f"{path}: RefMod metadata is not an object.")
    tensors = tensors_in(header)

    if meta.get("kind") == "bundle":
        if int(meta.get("_format_version", 0)) != BUNDLE_VERSION:
            raise ContainerError(f"{path}: unsupported bundle format version.")
        members = meta.get("members")
        if not isinstance(members, list) or not members:
            raise ContainerError(f"{path}: bundle carries no members.")
        out = []
        for index, member in enumerate(members):
            key = f"ref_{index}"
            if not isinstance(member, dict) or key not in tensors:
                raise ContainerError(f"{path}: bundle member {index} has no {key} tensor.")
            out.append((dict(member), key, list(tensors[key]["shape"])))
        return out

    if "latent" not in tensors:
        raise ContainerError(f"{path}: standalone RefMod has no 'latent' tensor.")
    return [(dict(meta), "latent", list(tensors["latent"]["shape"]))]


def describe(path: str) -> Dict[str, Any]:
    """Everything worth reporting about a container, read from the header alone."""
    header, _ = read_header(path)
    meta = header_metadata(header)
    tensors = tensors_in(header)
    ref_keys = sorted((k for k in tensors if k.startswith("ref_")),
                      key=lambda k: int(k.split("_")[1]))
    # "latent" is a standalone RefMod's own tensor, not a weight -- counting it as one
    # would report a 1.1 MB reference file as carrying a LoRA.
    lora_keys = [k for k in tensors if not k.startswith("ref_") and k != "latent"]
    members: List[Dict[str, Any]] = []
    raw = meta.get(REFMOD_KEY)
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            listed = parsed.get("members") if parsed.get("kind") == "bundle" else [parsed]
            for index, member in enumerate(listed or []):
                key = f"ref_{index}" if parsed.get("kind") == "bundle" else "latent"
                if key not in tensors:
                    continue
                kind = str(member.get("kind", "video"))
                shape = list(tensors[key]["shape"])
                members.append({"index": index, "name": member.get("name", key),
                                "kind": kind, "shape": shape,
                                "concept_type": member.get("concept_type", "generic"),
                                "description": member.get("description", ""),
                                "tokens": token_count(kind, shape)})
    hybrid = None
    if meta.get(HYBRID_KEY):
        try:
            hybrid = json.loads(meta[HYBRID_KEY])
        except ValueError:
            hybrid = None
    return {
        "path": path,
        "size": os.path.getsize(path),
        "lora_keys": len(lora_keys),
        "ref_keys": len(ref_keys),
        "members": members,
        "total_tokens": sum(m["tokens"] for m in members),
        "hybrid": hybrid,
        "network_dim": meta.get("ss_network_dim"),
        "network_alpha": meta.get("ss_network_alpha"),
        "title": meta.get("modelspec.title"),
        "is_reflora": bool(lora_keys) and bool(ref_keys),
    }


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _write(entries: Sequence[Tuple[str, str, List[int], str, int, int]],
           metadata: Dict[str, str], out_path: str) -> str:
    """Write a safetensors file by copying each tensor's bytes from its source.

    ``entries`` is (key, dtype, shape, source path, source offset, byte length).  The
    file lands through a temp file in the destination directory and an atomic replace,
    so an interrupted pack never leaves a half-written model behind.
    """
    header: Dict[str, Any] = {"__metadata__": metadata}
    offset = 0
    for key, dtype, shape, _src, _off, size in entries:
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)          # safetensors wants an 8-byte-aligned header

    directory = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".reflora-", suffix=".tmp", dir=directory)
    os.close(handle)
    try:
        with open(temporary, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for _key, _dtype, _shape, source, source_offset, size in entries:
                with open(source, "rb") as src:
                    src.seek(source_offset)
                    remaining = size
                    while remaining:
                        chunk = src.read(min(CHUNK, remaining))
                        if not chunk:
                            raise ContainerError(f"{source}: unexpected end of tensor data.")
                        out.write(chunk)
                        remaining -= len(chunk)
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return out_path


def pack(lora_path: str, refmod_paths: Sequence[str], out_path: str,
         name: Optional[str] = None, force: bool = False,
         packed_by: str = "ComfyUI-MiniMaxH3RefLoRA") -> Dict[str, Any]:
    """Combine a LoRA and one or more RefMods into a single RefLoRA container."""
    if not refmod_paths:
        raise ContainerError("Pass at least one RefMod to pack.")
    if os.path.exists(out_path) and not force:
        raise ContainerError(f"{out_path} already exists (pass force to overwrite).")

    lora_header, lora_base = read_header(lora_path)
    lora_tensors = tensors_in(lora_header)
    if not any(k.endswith(".lora_down.weight") or k.endswith(".lora_A.weight")
               for k in lora_tensors):
        raise ContainerError(f"{lora_path}: no LoRA weights found (no *.lora_down.weight).")
    colliding = sorted(k for k in lora_tensors if k.startswith("ref_"))
    if colliding:
        raise ContainerError(f"{lora_path}: already uses reserved key(s) {colliding}.")

    metadata = header_metadata(lora_header)
    metadata.pop(REFMOD_KEY, None)
    metadata.pop(HYBRID_KEY, None)

    entries: List[Tuple[str, str, List[int], str, int, int]] = []
    for key, spec in lora_tensors.items():
        start, end = spec["data_offsets"]
        entries.append((key, spec["dtype"], list(spec["shape"]),
                        lora_path, lora_base + start, end - start))

    members: List[Dict[str, Any]] = []
    member_shapes: List[List[int]] = []
    sources: List[str] = []
    for source in refmod_paths:
        _header, base = read_header(source)
        tensors = tensors_in(_header)
        for member, key, shape in _members_of(source):
            index = len(members)
            if index >= MAX_MEMBERS:
                raise ContainerError(f"A bundle holds at most {MAX_MEMBERS} references.")
            kind = str(member.get("kind", "video"))
            validate_latent(kind, shape, f"{os.path.basename(source)}:{key}")
            member["_format_version"] = MEMBER_VERSION
            member.pop("path", None)
            member.pop("bundle_index", None)
            members.append(member)
            member_shapes.append([int(n) for n in shape])
            spec = tensors[key]
            start, end = spec["data_offsets"]
            entries.append((f"ref_{index}", spec["dtype"], shape,
                            source, base + start, end - start))
        sources.append(os.path.basename(source))

    display = name or os.path.splitext(os.path.basename(out_path))[0]
    metadata[REFMOD_KEY] = json.dumps({"_format_version": BUNDLE_VERSION, "kind": "bundle",
                                       "name": display, "members": members})
    metadata[HYBRID_KEY] = json.dumps({
        "version": HYBRID_VERSION,
        "lora": {"format": "kohya", "keys": len(lora_tensors),
                 "network_dim": metadata.get("ss_network_dim"),
                 "network_alpha": metadata.get("ss_network_alpha")},
        "refmod_count": len(members),
        "packed_by": packed_by,
        "sources": {"lora": os.path.basename(lora_path), "refmods": sources},
    })

    _write(entries, metadata, out_path)
    return {
        "path": out_path,
        "size": os.path.getsize(out_path),
        "lora_keys": len(lora_tensors),
        "members": [{"name": m.get("name", f"ref_{i}"), "kind": m.get("kind", "video"),
                     "tokens": token_count(str(m.get("kind", "video")), member_shapes[i])}
                    for i, m in enumerate(members)],
    }


def unpack(hybrid_path: str, out_dir: str, force: bool = False) -> List[str]:
    """Split a container back into a plain LoRA and standalone RefMod files."""
    header, base = read_header(hybrid_path)
    tensors = tensors_in(header)
    metadata = header_metadata(header)
    ref_keys = {k for k in tensors if k.startswith("ref_")}
    if not ref_keys:
        raise ContainerError(f"{hybrid_path}: no ref_* tensors -- nothing to unpack.")

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(hybrid_path))[0]
    marker = stem.find("_reflora")
    if marker > 0:                                      # "..._reflora_multi" -> "..."
        stem = stem[:marker]
    written: List[str] = []

    lora_meta = {k: v for k, v in metadata.items() if k not in (REFMOD_KEY, HYBRID_KEY)}
    lora_entries = []
    for key, spec in tensors.items():
        if key in ref_keys:
            continue
        start, end = spec["data_offsets"]
        lora_entries.append((key, spec["dtype"], list(spec["shape"]),
                             hybrid_path, base + start, end - start))
    if lora_entries:
        target = os.path.join(out_dir, f"{stem}.safetensors")
        if os.path.exists(target) and not force:
            raise ContainerError(f"{target} already exists (pass force to overwrite).")
        written.append(_write(lora_entries, lora_meta, target))

    bundle = json.loads(metadata[REFMOD_KEY])
    members = bundle.get("members") if bundle.get("kind") == "bundle" else [bundle]
    for index, member in enumerate(members or []):
        key = f"ref_{index}"
        if key not in tensors:
            continue
        spec = tensors[key]
        start, end = spec["data_offsets"]
        member = dict(member)
        member.pop("bundle_index", None)
        target = os.path.join(out_dir, f"{member.get('name', key)}.safetensors")
        if os.path.exists(target) and not force:
            raise ContainerError(f"{target} already exists (pass force to overwrite).")
        written.append(_write(
            [("latent", spec["dtype"], list(spec["shape"]),
              hybrid_path, base + start, end - start)],
            {REFMOD_KEY: json.dumps(member)}, target))
    return written


def verify(hybrid_path: str, lora_path: Optional[str] = None) -> Dict[str, Any]:
    """Structural check, plus a byte comparison against the source LoRA when given one."""
    info = describe(hybrid_path)
    problems: List[str] = []
    claims_reflora = bool(info["ref_keys"]) or info.get("hybrid") is not None
    if claims_reflora:
        # Only a file that presents itself as a container is held to both halves; a plain
        # LoRA or a standalone RefMod passed in here is reported, not failed.
        if not info["lora_keys"]:
            problems.append("no LoRA tensors -- the weight half is missing")
        if not info["members"]:
            problems.append("no RefMod members")
    for member in info["members"]:
        try:
            validate_latent(member["kind"], member["shape"], f"ref_{member['index']}")
        except ContainerError as exc:
            problems.append(str(exc))
    declared = (info.get("hybrid") or {}).get("lora", {}).get("keys")
    if declared is not None and declared != info["lora_keys"]:
        problems.append(
            f"header declares {declared} LoRA tensors but {info['lora_keys']} are present "
            "-- the file was rewritten by a tool that dropped them")

    if lora_path:
        source_header, source_base = read_header(lora_path)
        target_header, target_base = read_header(hybrid_path)
        source_tensors = tensors_in(source_header)
        target_tensors = tensors_in(target_header)
        missing = sorted(set(source_tensors) - set(target_tensors))
        if missing:
            problems.append(f"{len(missing)} source LoRA tensor(s) missing, e.g. {missing[:3]}")
        with open(lora_path, "rb") as a, open(hybrid_path, "rb") as b:
            for key, spec in source_tensors.items():
                other = target_tensors.get(key)
                if other is None:
                    continue
                if other["dtype"] != spec["dtype"] or list(other["shape"]) != list(spec["shape"]):
                    problems.append(f"{key}: dtype/shape differs from source")
                    continue
                start, end = spec["data_offsets"]
                a.seek(source_base + start)
                b.seek(target_base + other["data_offsets"][0])
                remaining = end - start
                while remaining:
                    size = min(CHUNK, remaining)
                    if a.read(size) != b.read(size):
                        problems.append(f"{key}: bytes differ from source")
                        break
                    remaining -= size
        info["byte_compared"] = True
    info["problems"] = problems
    info["ok"] = not problems
    return info
