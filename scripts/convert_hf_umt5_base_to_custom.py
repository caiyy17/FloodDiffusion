"""
Convert HuggingFace umt5-base model to custom T5Encoder format.
Usage: python scripts/convert_hf_umt5_base_to_custom.py
"""
import torch
from transformers import AutoModel


def convert_hf_to_custom(hf_model_path, output_path):
    """
    Convert HuggingFace T5 encoder weights to custom T5Encoder format.

    HuggingFace T5 encoder structure:
        encoder.embed_tokens.weight
        encoder.block.{i}.layer.0.SelfAttention.q.weight
        encoder.block.{i}.layer.0.SelfAttention.k.weight
        encoder.block.{i}.layer.0.SelfAttention.v.weight
        encoder.block.{i}.layer.0.SelfAttention.o.weight
        encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight (only block 0)
        encoder.block.{i}.layer.0.layer_norm.weight
        encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight (gate)
        encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight (fc1)
        encoder.block.{i}.layer.1.DenseReluDense.wo.weight (fc2)
        encoder.block.{i}.layer.1.layer_norm.weight
        encoder.final_layer_norm.weight

    Custom T5Encoder structure:
        token_embedding.weight
        blocks.{i}.norm1.weight
        blocks.{i}.attn.q.weight
        blocks.{i}.attn.k.weight
        blocks.{i}.attn.v.weight
        blocks.{i}.attn.o.weight
        blocks.{i}.pos_embedding.embedding.weight (each block has its own)
        blocks.{i}.norm2.weight
        blocks.{i}.ffn.gate.0.weight
        blocks.{i}.ffn.fc1.weight
        blocks.{i}.ffn.fc2.weight
        norm.weight
    """
    print(f"Loading HuggingFace model from {hf_model_path}...")
    hf_model = AutoModel.from_pretrained(hf_model_path)
    hf_state = hf_model.encoder.state_dict()

    custom_state = {}

    # Token embedding
    custom_state["token_embedding.weight"] = hf_state["embed_tokens.weight"]

    # Get number of layers
    num_layers = 12  # umt5-base has 12 layers

    for i in range(num_layers):
        # Self-attention
        custom_state[f"blocks.{i}.attn.q.weight"] = hf_state[f"block.{i}.layer.0.SelfAttention.q.weight"]
        custom_state[f"blocks.{i}.attn.k.weight"] = hf_state[f"block.{i}.layer.0.SelfAttention.k.weight"]
        custom_state[f"blocks.{i}.attn.v.weight"] = hf_state[f"block.{i}.layer.0.SelfAttention.v.weight"]
        custom_state[f"blocks.{i}.attn.o.weight"] = hf_state[f"block.{i}.layer.0.SelfAttention.o.weight"]

        # Layer norms (before attention and before FFN)
        custom_state[f"blocks.{i}.norm1.weight"] = hf_state[f"block.{i}.layer.0.layer_norm.weight"]
        custom_state[f"blocks.{i}.norm2.weight"] = hf_state[f"block.{i}.layer.1.layer_norm.weight"]

        # FFN
        custom_state[f"blocks.{i}.ffn.gate.0.weight"] = hf_state[f"block.{i}.layer.1.DenseReluDense.wi_0.weight"]
        custom_state[f"blocks.{i}.ffn.fc1.weight"] = hf_state[f"block.{i}.layer.1.DenseReluDense.wi_1.weight"]
        custom_state[f"blocks.{i}.ffn.fc2.weight"] = hf_state[f"block.{i}.layer.1.DenseReluDense.wo.weight"]

        # Relative position embedding (each block has its own)
        custom_state[f"blocks.{i}.pos_embedding.embedding.weight"] = hf_state[f"block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"]

    # Final layer norm
    custom_state["norm.weight"] = hf_state["final_layer_norm.weight"]

    print(f"Saving converted model to {output_path}...")
    torch.save(custom_state, output_path)
    print("Done!")

    # Print some stats
    print(f"\nConverted {len(custom_state)} parameters")
    total_params = sum(p.numel() for p in custom_state.values())
    print(f"Total parameters: {total_params:,}")


def main():
    hf_model_path = "deps/t5_umt5-base-enc/google/umt5-base"
    output_path = "deps/t5_umt5-base-enc/models_t5_umt5-base-enc.pth"
    convert_hf_to_custom(hf_model_path, output_path)


if __name__ == "__main__":
    main()
