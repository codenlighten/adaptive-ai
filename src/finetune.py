"""Copy fp32 weights into a ternary model — for post-training quantization
and continued fine-tuning. This is how real BitNet b1.58 deployments work:
pretrain in fp32, then continue training with the BitLinear constraint.

Strategy: every fp32 Linear in the BitMLP's hidden positions becomes a
BitLinear initialized with the fp32 weights. The BitLinear's STE will
quantize the weights to {-1, 0, +1} * alpha at the next forward pass,
but the underlying float "shadow" weights inherit the pretrained values,
giving Adam a good starting point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .char_lm import BitCharLM, FPCharLM
from .ternary import BitLinear


def copy_fp_to_bit_char_lm(fp: FPCharLM, bit: BitCharLM) -> None:
    """Copy a trained FPCharLM's weights into a fresh BitCharLM.

    Shared modules (embeddings, positional, output head, LayerNorms): exact copy.
    Inside each block:
      - FP block has nn.MultiheadAttention (fused QKV) + nn.Linear FFN
      - Bit block has BitLinear QKV/proj + BitLinear FFN
    We extract the fused QKV from the fp Linear-like internals and copy.
    """
    bit.tok_embed.weight.data.copy_(fp.tok_embed.weight.data)
    bit.pos.data.copy_(fp.pos.data)
    bit.norm.weight.data.copy_(fp.norm.weight.data)
    bit.norm.bias.data.copy_(fp.norm.bias.data)
    bit.head.weight.data.copy_(fp.head.weight.data)

    for fp_blk, bit_blk in zip(fp.blocks, bit.blocks):
        # LayerNorms
        bit_blk.norm1.weight.data.copy_(fp_blk.norm1.weight.data)
        bit_blk.norm1.bias.data.copy_(fp_blk.norm1.bias.data)
        bit_blk.norm2.weight.data.copy_(fp_blk.norm2.weight.data)
        bit_blk.norm2.bias.data.copy_(fp_blk.norm2.bias.data)

        # Attention QKV (fp uses MultiheadAttention.in_proj_weight which is QKV stacked)
        bit_blk.attn.qkv.weight.data.copy_(fp_blk.attn.in_proj_weight.data)
        bit_blk.attn.qkv.bias.data.copy_(fp_blk.attn.in_proj_bias.data)
        bit_blk.attn.proj.weight.data.copy_(fp_blk.attn.out_proj.weight.data)
        bit_blk.attn.proj.bias.data.copy_(fp_blk.attn.out_proj.bias.data)

        # BitLinear's LayerNorm — initialize identity-like.
        bit_blk.attn.qkv.norm.weight.data.fill_(1.0)
        bit_blk.attn.qkv.norm.bias.data.zero_()
        bit_blk.attn.proj.norm.weight.data.fill_(1.0)
        bit_blk.attn.proj.norm.bias.data.zero_()

        # FFN
        bit_blk.fc1.weight.data.copy_(fp_blk.fc1.weight.data)
        bit_blk.fc1.bias.data.copy_(fp_blk.fc1.bias.data)
        bit_blk.fc2.weight.data.copy_(fp_blk.fc2.weight.data)
        bit_blk.fc2.bias.data.copy_(fp_blk.fc2.bias.data)
        bit_blk.fc1.norm.weight.data.fill_(1.0)
        bit_blk.fc1.norm.bias.data.zero_()
        bit_blk.fc2.norm.weight.data.fill_(1.0)
        bit_blk.fc2.norm.bias.data.zero_()
