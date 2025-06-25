"""
Transformer decoder implementation for Canary models.

This module implements the Transformer decoder architecture used in 
NVIDIA's Canary models, adapted for MLX.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TransformerDecoderArgs:
    """Configuration for Transformer decoder."""
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    ff_expansion_factor: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048
    use_bias: bool = True
    norm_eps: float = 1e-5


class MultiHeadAttention(nn.Module):
    """Multi-head attention for Transformer decoder."""
    
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.head_dim = args.d_model // args.n_heads
        
        assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.wq = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
        self.wk = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
        self.wv = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
        self.wo = nn.Linear(args.d_model, args.d_model, bias=args.use_bias)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, T, C = x.shape
        
        # Compute queries, keys, values
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Handle cache for autoregressive generation
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
        
        # Compute attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / mx.sqrt(self.head_dim)
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = mx.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.wo(out)
        
        return out, (k, v)


class FeedForward(nn.Module):
    """Feed-forward network for Transformer decoder."""
    
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()
        hidden_dim = args.d_model * args.ff_expansion_factor
        
        self.w1 = nn.Linear(args.d_model, hidden_dim, bias=args.use_bias)
        self.w2 = nn.Linear(hidden_dim, args.d_model, bias=args.use_bias)
        self.dropout = nn.Dropout(args.dropout)
        
    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(self.dropout(nn.relu(self.w1(x))))


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer."""
    
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.LayerNorm(args.d_model, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.d_model, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Self-attention with residual connection
        attn_out, new_cache = self.attention(self.attention_norm(x), mask, cache)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.ffn_norm(x))
        x = x + self.dropout(ff_out)
        
        return x, new_cache


class TransformerDecoder(nn.Module):
    """Transformer decoder for Canary models."""
    
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()
        self.args = args
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(args.vocab_size, args.d_model)
        
        # Positional embeddings
        self.position_embeddings = nn.Embedding(args.max_seq_len, args.d_model)
        
        # Decoder layers
        self.layers = [TransformerDecoderLayer(args) for _ in range(args.n_layers)]
        
        # Final layer norm
        self.norm = nn.LayerNorm(args.d_model, eps=args.norm_eps)
        
        # Output projection
        self.output = nn.Linear(args.d_model, args.vocab_size, bias=False)
        
        self.dropout = nn.Dropout(args.dropout)
        
    def __call__(
        self, 
        tokens: mx.array, 
        encoder_output: Optional[mx.array] = None,
        cache: Optional[list] = None
    ) -> Tuple[mx.array, list]:
        B, T = tokens.shape
        
        # Create causal mask
        mask = mx.triu(mx.full((T, T), -mx.inf), k=1)
        
        # Token embeddings
        x = self.token_embeddings(tokens)
        
        # Add positional embeddings
        positions = mx.arange(T)
        if cache is not None and len(cache) > 0:
            # For autoregressive generation, offset positions
            positions = positions + cache[0][0].shape[2] if cache[0] is not None else positions
        
        pos_emb = self.position_embeddings(positions)
        x = x + pos_emb
        x = self.dropout(x)
        
        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)
        
        new_cache = []
        
        # Pass through decoder layers
        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, cache[i])
            new_cache.append(layer_cache)
        
        # Final layer norm
        x = self.norm(x)
        
        # Output projection
        logits = self.output(x)
        
        return logits, new_cache
    
    def generate(
        self, 
        encoder_output: mx.array,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> mx.array:
        """Generate tokens autoregressively."""
        B = encoder_output.shape[0]
        
        # Start with BOS token (assuming 0)
        tokens = mx.zeros((B, 1), dtype=mx.int32)
        cache = None
        
        for _ in range(max_length):
            # Forward pass
            logits, cache = self(tokens[:, -1:], encoder_output, cache)
            
            # Apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                top_k_logits, top_k_indices = mx.topk(logits, top_k)
                logits = mx.full_like(logits, -mx.inf)
                logits = logits.at[mx.arange(B)[:, None], top_k_indices].set(top_k_logits)
            
            # Sample next token
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs)
            
            # Append to sequence
            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)
            
            # Check for EOS token (assuming vocab_size - 1)
            if mx.all(next_token == self.args.vocab_size - 1):
                break
        
        return tokens