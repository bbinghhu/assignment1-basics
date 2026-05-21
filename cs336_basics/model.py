import math
import torch
from torch import nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        # Store W (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialize W
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        # Allocate embedding matrix
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Initialize embedding matrix
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Look up the embedding vector for the given token IDs
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter g, shape (d_model,)
        self.weight = nn.Parameter(
            torch.ones(self.d_model, device=device, dtype=dtype)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Rescale
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)
    
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model

        if d_ff is None:
            self.d_ff = int(round((8 / 3) * self.d_model / 64) * 64)
        else:
            self.d_ff = d_ff

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w3(x)

        silu = gate * torch.sigmoid(gate)
        out = self.w2(silu * value)

        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k 
        self.max_seq_len = max_seq_len

        # d_k has to be even because we rotate pairs of coordinates
        assert d_k % 2 == 0

        # frequencies for coordinate pairs: shape (d_k // 2,)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # positions: shape (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device).float()

        angles = torch.outer(positions, inv_freq)

        # Store precomputed sine and cosine values
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x -> (.., seq_len, d_k)
        # token_positions -> (.., seq_len)

        # Pick out the cosine and sine values
        # Indexes the first dimension 
        cos = self.cos[token_positions] # Shape: (max_seq_len, d_k // 2)
        sin = self.sin[token_positions]

        # shape: (.., seq_len, d_k // 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin 
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out

def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    """
    Softmax that subtracts the maximum value in the 𝑖-th dimension from all elements 
    of the 𝑖-th dimension to avoid numerical stability issues
    """
    
    # Subtract max value
    max_val = torch.max(x, dim=dim, keepdim=True).values
    new_x = x - max_val
    
    exp_x = torch.exp(new_x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # Compute attention scores
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    
    # Apply softmax on the keys dimension
    softmaxed_scores = softmax(scores, dim=-1)

    # Multiply with value matrix V
    attention = einsum(softmaxed_scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention


class MHA(nn.Module):
    def __init__(
    self,
    d_model: int,
    num_heads: int,
    theta: float | None = None,
    max_seq_len: int | None = None,
    use_rope: bool = False,
    device=None,
    dtype=None,
):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.use_rope = use_rope

        # Learnable weights
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

        # Initalize rope if needed
        if self.use_rope:
            assert theta is not None
            assert max_seq_len is not None
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: shape (..., seq_len, d_model)
        returns: shape (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]

        # Compute the query, key, and value vectors
        # shape (..., seq_len, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Split d_model into num_heads * d_k
        Q = rearrange(Q, "... seq_len (num_heads  d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads,)
        K = rearrange(K, "... seq_len (num_heads  d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads,)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        # Apply RoPE to Q, K
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(-2) # shape: (..., 1, seq_len)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Build causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # Apply dot product attention on each head
        # shape (..., num_heads, seq_len, d_v)
        multi_head_attention = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate all the heads
        # shape (..., seq_len, d_model) again
        multi_head_attention = rearrange(
            multi_head_attention,
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
        )

        # Apply W_O
        out = self.W_O(multi_head_attention)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Pre-mha norm
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # MHA with rope
        self.mha = MHA(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            use_rope=True,
            device=device,
            dtype=dtype,
        )

        # Pre-feedforward norm
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        # Feedforward layer
        self.ffwd = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (..., seq_len, d_model)
        output shape: (..., seq_len, d_model)
        """
        z = x + self.mha(self.norm1(x))
        out = z + self.ffwd(self.norm2(z))
        return out
    
class Transformer_LM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int, # determining the dimensionality of the token embedding matrix
        context_length: int, # The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.
        num_layers: int, #The number of Transformer blocks to use
        device=None,
        dtype=None,
):
        super().__init__()

        # Initialize embedding matrix
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        # Initialize transformer blocks. Want num_layers of them.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,  
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Initialize final norm
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # Initialize final linear layer
        self.output_embedding = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x)
        
        logits = self.output_embedding(self.norm(x))
        return logits
        





















