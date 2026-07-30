import torch.nn as nn
from einops import einsum, rearrange
import torch 
import math




def softmax(x , dim):
    x_max = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x-x_max)

    return exp/torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    QK = einsum(q, k, "... seq_len d_k, ... seq d_k -> ... seq_len seq")
    d_k = q.shape[-1]
    root_d_k = 1/math.sqrt(d_k)

    s = QK * root_d_k
    if mask is not None:
        s = s.masked_fill(mask == False, float('-inf'))
    s_max = softmax(s, -1)

    return einsum(s_max, v, "... q k, ... k d_v -> ... q d_v")



class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device = device, dtype = dtype)
            )
        nn.init.trunc_normal_(
            self.weight,
              mean = 0.0,
                std = math.sqrt(2/(in_features + out_features)), 
                a = -3.0 * math.sqrt(2/(in_features + out_features)),
                  b = 3.0 * math.sqrt(2/(in_features + out_features))
            )
    def forward(self, A):
        return einsum(A, self.weight, "... d_in, d_out d_in -> ... d_out")



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = 1.0, a = -3.0, b = 3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]



class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5 , device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(d_model) 
        )
        self.eps = eps
        # No trunc normal since we're not drawing from a distribution.

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32) # convert to float32 to avoid overflow
        sq = x*x
        m = sq.mean(dim=-1, keepdim= True)
        result = x * torch.rsqrt(m + self.eps) * self.weight
        return result.to(in_dtype)



class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        w1x = self.w1(x)
        w3x = self.w3(x)
        siluw1x = w1x * torch.sigmoid(w1x)

        return self.w2(siluw1x * w3x)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)



# RoPE
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        value_vector = torch.arange(0, d_k, 2).float()
        frequency_vector = theta ** (-(value_vector/d_k))
        positions_vector = torch.arange(max_seq_len).float()
        angles = torch.outer(positions_vector, frequency_vector)

        cos_table = angles.cos()
        sin_table = angles.sin()

        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x, token_positions):
        pairs = rearrange(x, '... seq (half two) -> ... seq half two', two = 2) # Get a 2d vector of pairs. 
        a = pairs[..., 0]
        b = pairs[..., 1]

        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        a_rot = a * cos - b * sin
        b_rot = a * sin + b * cos

        stacked = torch.stack((a_rot, b_rot), dim = -1)

        return rearrange(stacked, '... seq half two -> ... seq (half two)')





class Multihead_attention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len = None, theta = None, do_rope: bool = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_K = Linear(d_model, d_model)
        self.W_Q = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self.do_rope = do_rope
        if self.do_rope == True:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = rearrange(Q, "batch seq (h d_k) -> batch h seq d_k", h=self.num_heads)
        K = rearrange(K, "batch seq (h d_k) -> batch h seq d_k", h=self.num_heads)
        V = rearrange(V, "batch seq (h d_k) -> batch h seq d_k", h=self.num_heads)

        # Apply RoPE.
        seq = x.shape[1]
        token_positions = torch.arange(seq, device= x.device)
        if self.do_rope == True:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Apply causal masking
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device = x.device))


        out = scaled_dot_product_attention(Q, K, V, mask)


        out = rearrange(out, "batch h seq d_k -> batch seq (h d_k)")

        return self.W_O(out)



class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.attn = Multihead_attention(d_model, num_heads, max_seq_len, theta, do_rope = True)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x



class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, theta):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)