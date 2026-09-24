#!/usr/bin/env python3
"""Console packer for H3 RefLoRA containers.

Standard library only -- no torch, no safetensors, no venv.  Run it with any Python 3.9+:

    python pack_reflora.py pack    --lora <lora.safetensors> --refmod <mod.safetensors> \
                                   [--refmod <mod2.safetensors> ...] [--out <path>]
    python pack_reflora.py inspect <reflora.safetensors>
    python pack_reflora.py verify  <reflora.safetensors> [--lora <source lora>]
    python pack_reflora.py unpack  <reflora.safetensors> --out-dir <dir>

Names are resolved against ComfyUI's models/loras and models/refmods when they are not
paths, so ``--lora MinimaxH3/test/minimaxh3_aneta_v1-000060`` works from anywhere.
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__:
    from .container import ContainerError, describe, pack, token_count, unpack, verify
else:                                                   # run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from container import ContainerError, describe, pack, token_count, unpack, verify

DEFAULT_COMFY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", ".."))


def _models_root() -> str:
    return os.environ.get("COMFYUI_MODELS_DIR") or os.path.join(DEFAULT_COMFY, "models")


def resolve(name: str, kind: str) -> str:
    """A path as given, or a name looked up under models/<kind>/ with or without suffix."""
    if os.path.isfile(name):
        return os.path.abspath(name)
    root = os.path.join(_models_root(), kind)
    for candidate in (name, name + ".safetensors"):
        path = os.path.join(root, candidate)
        if os.path.isfile(path):
            return os.path.abspath(path)
    if os.path.isdir(root):
        for directory, _subdirs, files in os.walk(root):
            for filename in files:
                stem = os.path.splitext(filename)[0]
                if filename == name or stem == name:
                    return os.path.abspath(os.path.join(directory, filename))
    raise ContainerError(f"Could not find {kind[:-1]} '{name}' (looked in {root}).")


def _print_info(info) -> None:
    print(f"{info['path']}")
    print(f"  size          {info['size'] / 1e6:.1f} MB")
    print(f"  LoRA tensors  {info['lora_keys']}"
          + (f"   dim {info['network_dim']} / alpha {info['network_alpha']}"
             if info.get("network_dim") else ""))
    if info["members"]:
        print(f"  references    {len(info['members'])}   "
              f"{info['total_tokens']} tokens total")
        for member in info["members"]:
            shape = "x".join(str(n) for n in member["shape"])
            print(f"    {member['index'] + 1}. {member['name']:<42} [{member['kind']}] "
                  f"{shape:>18}  {member['tokens']:>6} tokens")
    else:
        print("  references    none")
    if info.get("hybrid"):
        print(f"  packed by     {info['hybrid'].get('packed_by', '?')}")
    for problem in info.get("problems", []):
        print(f"  !! {problem}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pack a MiniMax H3 LoRA and its RefMod(s) into one .safetensors.")
    sub = parser.add_subparsers(dest="command", required=True)

    packer = sub.add_parser("pack", help="build a RefLoRA container")
    packer.add_argument("--lora", required=True, help="source LoRA (path or name)")
    packer.add_argument("--refmod", action="append", required=True, metavar="PATH",
                        help="a RefMod to embed; repeat for more than one")
    packer.add_argument("--out", default=None,
                        help="output path (default: models/loras/MinimaxH3/reflora/"
                             "<lora>_reflora.safetensors)")
    packer.add_argument("--name", default=None, help="display name stored in the container")
    packer.add_argument("--force", action="store_true", help="overwrite an existing file")

    for command, helptext in (("inspect", "report a container's contents"),
                              ("verify", "structural check, optionally byte-compared")):
        node = sub.add_parser(command, help=helptext)
        node.add_argument("path")
        if command == "verify":
            node.add_argument("--lora", default=None,
                              help="source LoRA to byte-compare the weights against")

    unpacker = sub.add_parser("unpack", help="split a container back into its parts")
    unpacker.add_argument("path")
    unpacker.add_argument("--out-dir", required=True)
    unpacker.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "pack":
        lora = resolve(args.lora, "loras")
        refmods = [resolve(name, "refmods") for name in args.refmod]
        out = args.out
        if not out:
            stem = os.path.splitext(os.path.basename(lora))[0] + "_reflora.safetensors"
            out = os.path.join(_models_root(), "loras", "MinimaxH3", "reflora", stem)
        result = pack(lora, refmods, out, name=args.name, force=args.force)
        print(f"wrote {result['path']}  ({result['size'] / 1e6:.1f} MB)")
        _print_info(describe(result["path"]))
        total = sum(m["tokens"] for m in describe(result["path"])["members"])
        if total > 6000:
            print(f"\nNOTE: {total} reference tokens. Every reference is attended to in "
                  "every DiT block, so a container this heavy is slow to sample. Consider "
                  "selecting fewer members at load time.")
        return 0

    if args.command == "inspect":
        _print_info(describe(resolve(args.path, "loras")))
        return 0

    if args.command == "verify":
        lora = resolve(args.lora, "loras") if args.lora else None
        info = verify(resolve(args.path, "loras"), lora)
        _print_info(info)
        print("\nOK" if info["ok"] else "\nPROBLEMS FOUND")
        return 0 if info["ok"] else 1

    written = unpack(resolve(args.path, "loras"), args.out_dir, force=args.force)
    for path in written:
        print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContainerError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
