import os
import io
import zlib
import torch
from train_gpt import GPT, quantize_state_dict_int8, Hyperparameters

def check_model_size(dim, num_heads):
    code_bytes = 50000  # Estimate script size
    
    # Initialize model with test config
    model = GPT(
        vocab_size=1024,
        num_layers=9,      # not used for params anymore, but required argument
        model_dim=dim,
        num_heads=num_heads,
        num_kv_heads=4,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
    ).bfloat16()

    params = sum(p.numel() for p in model.parameters())
    
    # Quantize and compress
    quant_obj, quant_stats = quantize_state_dict_int8(model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_file_bytes = len(quant_blob)
    
    total_size = quant_file_bytes + code_bytes
    total_mb = total_size / (1024 * 1024)
    
    print(f"dim={dim:4d}, heads={num_heads:2d} -> {params/1e6:5.2f}M params -> {total_mb:5.2f} MiB")
    return total_size

if __name__ == "__main__":
    print(f"Target limit is 16.00 MiB (16777216 bytes)")
    
    # Sweep over dimensions and heads
    test_configs = [
        (2560, 40),
        (3072, 48),
        (3584, 56),
        (4096, 64),
        (4608, 72)
    ]
    
    best_config = None
    best_size = 0
    TARGET = 16.0 * 1024 * 1024
    
    for dim, heads in test_configs:
        size = check_model_size(dim, heads)
        if size <= TARGET and size > best_size:
            best_size = size
            best_config = (dim, heads)
            
    if best_config:
        print(f"\nOptimal configuration below 16MiB:")
        print(f"model_dim = {best_config[0]}")
        print(f"num_heads = {best_config[1]}")
    else:
        print("All configurations exceed 16MB!")
