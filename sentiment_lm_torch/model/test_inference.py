import torch
from sentiment_lm_torch.model.attention import AttentionBlock, causal_block_mask

def main():
    batch_size = 3
    seq_len = 4
    d_model = 64
    num_heads = 2
    device = "cuda"

    attention_block = AttentionBlock(num_heads=num_heads, d_model=d_model)
    attention_block.to(device)
    block_mask = causal_block_mask(seq_len, device=device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    attention_block.train()
    training_positions = torch.arange(seq_len, dtype=torch.int64, device=device)
    training_out = attention_block(x, training_positions, block_mask)
    # print(training_out)

    attention_block.eval()
    attention_block.init_kv_cache(batch_size, seq_len, device=torch.device(device), dtype=torch.float32)
    inference_out = torch.zeros_like(training_out, device=device)
    for i in range(seq_len):
        inference_positions = torch.full((batch_size, 1), i, dtype=torch.int64, device=device)
        inference_in = x[:, i, :][:, None, :]
        out_one = attention_block(inference_in, inference_positions)
        # print(out_one.shape)
        inference_out[:, i, :] = out_one.squeeze(1)
        # break

        # inference_out[:, i] = attention_block(x[:, i, :][:, None, :], inference_positions)
    print(training_out)
    print(inference_out)
    print(torch.abs(training_out - inference_out) < 1e-6)
    

if __name__ == '__main__':
    main()
