#!/usr/bin/env python3
"""Convert an EAGLE3 draft checkpoint to GGUF without torch/transformers.

The K3 board is riscv64 with no torch wheels, so convert_hf_to_gguf.py cannot run
here. This reads model.safetensors with numpy and lifts the tokenizer straight out
of an already-converted target GGUF instead of going through AutoTokenizer.

Mirrors the EAGLE3 logic in conversion/llama.py (midlayer.* rename, hidden_norm ->
attn_norm_2, t2d dropped, d2t kept as I64, llama Q/K undo_permute).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))

import gguf  # noqa: E402


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
        out: dict[str, np.ndarray] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            raw = f.read(end - begin)
            dtype = meta["dtype"]
            if dtype == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = arr.view(np.float32)
            elif dtype in ("F16", "F32", "I64", "I32", "BOOL"):
                arr = np.frombuffer(raw, dtype={
                    "F16": np.float16, "F32": np.float32,
                    "I64": np.int64, "I32": np.int32, "BOOL": np.bool_,
                }[dtype])
            else:
                raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
            out[name] = arr.reshape(meta["shape"])
        return out


def permute(w: np.ndarray, n_head: int, n_head_kv: int) -> np.ndarray:
    # llama.cpp undoes the HF llama Q/K interleaving; see LlamaModel.permute
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
             .swapaxes(1, 2)
             .reshape(w.shape))


def copy_tokenizer(writer: gguf.GGUFWriter, target_gguf: Path) -> int:
    reader = gguf.GGUFReader(target_gguf, "r")
    copied = 0
    for key, field in reader.fields.items():
        if not (key.startswith("tokenizer.") or key == "general.name"):
            continue
        if not field.types:
            continue
        main = field.types[0]
        if main == gguf.GGUFValueType.ARRAY:
            sub = field.types[-1]
            if sub == gguf.GGUFValueType.STRING:
                val = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
            else:
                val = [field.parts[i].tolist()[0] for i in field.data]
            writer.add_key_value(key, val, gguf.GGUFValueType.ARRAY, sub_type=sub)
        elif main == gguf.GGUFValueType.STRING:
            writer.add_key_value(key, field.contents(), gguf.GGUFValueType.STRING)
        else:
            writer.add_key_value(key, field.contents(), main)
        copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("draft_dir", type=Path)
    ap.add_argument("--target-model-dir", type=Path, required=True,
                    help="target HF dir, read for config.json (layer count / hidden size)")
    ap.add_argument("--target-gguf", type=Path, required=True,
                    help="already-converted target GGUF, used as the tokenizer source")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--outtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--no-permute", action="store_true",
                    help="skip the llama Q/K undo_permute (diagnostic A/B)")
    args = ap.parse_args()

    cfg = json.loads((args.draft_dir / "config.json").read_text())
    tgt = json.loads((args.target_model_dir / "config.json").read_text())
    if "text_config" in tgt:
        tgt = {**tgt, **tgt["text_config"]}

    n_head = cfg["num_attention_heads"]
    n_head_kv = cfg["num_key_value_heads"]
    n_embd = cfg["hidden_size"]
    head_dim = cfg.get("head_dim") or n_embd // n_head

    tgt_layers = [2, tgt["num_hidden_layers"] // 2, tgt["num_hidden_layers"] - 3]
    tgt_hidden = cfg.get("target_hidden_size") or tgt["hidden_size"]

    writer = gguf.GGUFWriter(args.outfile, "eagle3")
    writer.add_block_count(cfg["num_hidden_layers"])
    writer.add_context_length(cfg["max_position_embeddings"])
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(cfg["intermediate_size"])
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(cfg["rope_theta"])
    writer.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    writer.add_vocab_size(cfg["vocab_size"])

    writer.add_array("eagle3.target_layers", tgt_layers)
    writer.add_uint32("eagle3.target_hidden_size", tgt_hidden)
    writer.add_bool("eagle3.norm_before_residual", cfg.get("norm_before_residual", False))

    n_kv = copy_tokenizer(writer, args.target_gguf)
    print(f"copied {n_kv} tokenizer/general KVs from {args.target_gguf.name}")
    print(f"target_layers={tgt_layers} target_hidden_size={tgt_hidden}")

    tensors = read_safetensors(args.draft_dir / "model.safetensors")
    np_out = np.float16 if args.outtype == "f16" else np.float32

    rename = {
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
        "midlayer.input_layernorm.weight": "blk.0.attn_norm.weight",
        "midlayer.hidden_norm.weight": "blk.0.attn_norm_2.weight",
        "midlayer.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "midlayer.self_attn.k_proj.weight": "blk.0.attn_k.weight",
        "midlayer.self_attn.v_proj.weight": "blk.0.attn_v.weight",
        "midlayer.self_attn.o_proj.weight": "blk.0.attn_output.weight",
        "midlayer.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
        "midlayer.mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
        "midlayer.mlp.up_proj.weight": "blk.0.ffn_up.weight",
        "midlayer.mlp.down_proj.weight": "blk.0.ffn_down.weight",
        "fc.weight": "fc.weight",
    }

    for name, arr in sorted(tensors.items()):
        if name == "t2d":
            continue  # not used at runtime
        if name == "d2t":
            # checkpoints store d2t as an offset; the graph indexes target-vocab rows with
            # it directly (ggml_set_rows), so it must be absolute target token ids
            data = arr.astype(np.int64).reshape(-1)
            data = data + np.arange(data.size, dtype=np.int64)
            if np.any((data < 0) | (data >= tgt["vocab_size"])):
                raise ValueError(f"d2t target ids out of range for target vocab {tgt['vocab_size']}")
            if np.unique(data).size != data.size:
                raise ValueError("d2t contains duplicate target ids")
            writer.add_tensor("d2t", data, raw_dtype=gguf.GGMLQuantizationType.I64)
            print(f"  d2t -> I64 {data.shape} (offsets -> absolute ids)")
            continue
        if not args.no_permute:
            if name.endswith("q_proj.weight"):
                arr = permute(arr, n_head, n_head)
            elif name.endswith("k_proj.weight"):
                arr = permute(arr, n_head, n_head_kv)
        new = rename.get(name)
        if new is None:
            raise KeyError(f"unmapped tensor {name}")
        writer.add_tensor(new, np.ascontiguousarray(arr).astype(np_out))
        print(f"  {name} -> {new} {arr.shape}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
