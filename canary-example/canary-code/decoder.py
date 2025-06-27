"""
Transformer Decoder for Canary MLX
A lightweight 4-layer transformer decoder for the Canary ASR model.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TransformerDecoderArgs:
    """Configuration for the Transformer Decoder."""
    n_layers: int = 4
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion_factor: int = 4
    vocab_size: int = 5248
    max_seq_len: int = 448
    dropout: float = 0.1


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        
    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(self.activation(self.w1(x)))


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiHeadAttention(
            dims=args.d_model,
            num_heads=args.n_heads,
            bias=True
        )
        self.attn_norm = nn.LayerNorm(args.d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiHeadAttention(
            dims=args.d_model,
            num_heads=args.n_heads,
            bias=True
        )
        self.cross_attn_norm = nn.LayerNorm(args.d_model)
        
        # Feed-forward
        d_ff = args.d_model * args.ff_expansion_factor
        self.ffn = FeedForward(args.d_model, d_ff)
        self.ffn_norm = nn.LayerNorm(args.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        mask: Optional[mx.array] = None,
        encoder_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with causal mask
        residual = x
        x = self.attn_norm(x)
        x = self.self_attn(x, x, x, mask=mask)
        x = self.dropout(x) + residual
        
        # Cross-attention to encoder
        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, encoder_output, encoder_output, mask=encoder_mask)
        x = self.dropout(x) + residual
        
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder for Canary."""
    
    def __init__(self, args: TransformerDecoderArgs):
        super().__init__()
        self.args = args
        
        # Token embedding
        self.token_embed = nn.Embedding(args.vocab_size, args.d_model)
        
        # Positional encoding (learned)
        self.pos_embed = nn.Embedding(args.max_seq_len, args.d_model)
        
        # Decoder layers
        self.layers = [
            TransformerDecoderLayer(args) 
            for _ in range(args.n_layers)
        ]
        
        # Final layer norm
        self.norm = nn.LayerNorm(args.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
    def create_causal_mask(self, seq_len: int) -> mx.array:
        """Create a causal attention mask."""
        mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
        return mask * -1e9  # Large negative value for softmax
        
    def __call__(
        self,
        tokens: mx.array,
        encoder_output: mx.array,
        encoder_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L = tokens.shape
        
        # Token + position embeddings
        positions = mx.arange(L)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(L)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, mask=causal_mask, encoder_mask=encoder_mask)
            
        # Final norm
        x = self.norm(x)
        
        return x


# Simple usage example:
if __name__ == "__main__":
    # Test the decoder
    args = TransformerDecoderArgs()
    decoder = TransformerDecoder(args)
    
    # Dummy inputs
    batch_size = 2
    seq_len = 10
    encoder_len = 100
    encoder_dim = 512  # Note: different from decoder
    
    tokens = mx.random.randint(0, args.vocab_size, (batch_size, seq_len))
    encoder_output = mx.random.normal((batch_size, encoder_len, args.d_model))
    
    # Forward pass
    output = decoder(tokens, encoder_output)
    print(f"Decoder output shape: {output.shape}")
    # Should be: (batch_size, seq_len, d_model)
