"""
Canary MLX Model Implementation
Combines FastConformer encoder with Transformer decoder for multilingual ASR.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import from parakeet (these files should be copied to your project)
from parakeet_mlx.conformer import Conformer, ConformerArgs
from parakeet_mlx.audio import PreprocessArgs, get_logmel, load_audio

# Import the new decoder
from decoder import TransformerDecoder, TransformerDecoderArgs


@dataclass
class CanaryArgs:
    """Configuration for the Canary model."""
    preprocessor: PreprocessArgs
    encoder: ConformerArgs
    decoder: TransformerDecoderArgs
    vocab_size: int = 5248
    languages: List[str] = field(default_factory=lambda: ["en", "de", "es", "fr"])
    tasks: List[str] = field(default_factory=lambda: ["transcribe", "translate"])


class CanaryModel(nn.Module):
    """Canary ASR Model with FastConformer encoder and Transformer decoder."""
    
    def __init__(self, args: CanaryArgs):
        super().__init__()
        self.args = args
        
        # Special tokens (you'll need to verify these from the tokenizer)
        self.sot_token = 1  # Start of transcript
        self.eot_token = 2  # End of transcript
        self.blank_token = 0  # Blank/padding
        
        # Language tokens (these are examples - verify from actual tokenizer)
        self.lang_tokens = {
            "en": 5240,  # <|en|>
            "de": 5241,  # <|de|>
            "es": 5242,  # <|es|>
            "fr": 5243,  # <|fr|>
        }
        
        # Task tokens
        self.task_tokens = {
            "transcribe": 5244,  # <|transcribe|>
            "translate": 5245,   # <|translate|>
        }
        
        # Model components
        self.encoder = Conformer(args.encoder)
        self.decoder = TransformerDecoder(args.decoder)
        
        # Encoder to decoder projection (512 -> 1024 for Canary)
        self.encoder_proj = nn.Linear(
            args.encoder.d_model,
            args.decoder.d_model
        )
        
        # Output projection
        self.output_proj = nn.Linear(
            args.decoder.d_model,
            args.vocab_size
        )
        
    def encode(self, mel: mx.array) -> tuple[mx.array, mx.array]:
        """Encode audio features using the FastConformer encoder."""
        # Add batch dimension if needed
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)
            
        # Encode
        features, lengths = self.encoder(mel)
        
        # Project to decoder dimension
        features = self.encoder_proj(features)
        
        return features, lengths
        
    def decode_greedy(
        self,
        encoder_output: mx.array,
        max_tokens: int = 448,
        language: str = "en",
        task: str = "transcribe",
    ) -> List[int]:
        """Simple greedy decoding."""
        # Start with special tokens
        tokens = [
            self.sot_token,
            self.lang_tokens.get(language, self.lang_tokens["en"]),
            self.task_tokens.get(task, self.task_tokens["transcribe"]),
        ]
        
        # Decode loop
        for _ in range(max_tokens):
            # Convert tokens to array
            token_array = mx.array([tokens])
            
            # Run decoder
            decoder_out = self.decoder(token_array, encoder_output)
            
            # Get logits for last position
            logits = self.output_proj(decoder_out[:, -1, :])
            
            # Greedy selection
            next_token = mx.argmax(logits, axis=-1).item()
            
            # Check for end of transcript
            if next_token == self.eot_token:
                break
                
            tokens.append(next_token)
            
        return tokens[3:]  # Remove special tokens
        
    def generate(
        self,
        mel: mx.array,
        language: str = "en",
        task: str = "transcribe",
        max_tokens: int = 448,
    ) -> str:
        """Generate text from audio features."""
        # Encode audio
        encoder_output, lengths = self.encode(mel)
        
        # Decode tokens
        tokens = self.decode_greedy(
            encoder_output,
            max_tokens=max_tokens,
            language=language,
            task=task,
        )
        
        # TODO: Convert tokens to text using tokenizer
        # For now, return token IDs as string
        return f"Generated tokens: {tokens}"
        
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        task: str = "transcribe",
    ) -> str:
        """Transcribe an audio file."""
        # Load audio
        audio = load_audio(
            Path(audio_path),
            self.args.preprocessor.sample_rate,
            dtype=mx.float32
        )
        
        # Get mel spectrogram
        mel = get_logmel(audio, self.args.preprocessor)
        
        # Generate transcription
        return self.generate(mel, language=language, task=task)


def create_canary_180m_config() -> CanaryArgs:
    """Create configuration for Canary-180M-Flash model."""
    preprocessor = PreprocessArgs(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=128,
        f_min=0,
        f_max=8000,
    )
    
    encoder = ConformerArgs(
        feat_in=128,
        n_layers=17,  # 180M model has 17 layers
        d_model=512,
        n_heads=8,
        ff_expansion_factor=4,
        subsampling_factor=8,
        self_attention_model="rel_pos",
        subsampling="dw_striding",
        conv_kernel_size=9,
        subsampling_conv_channels=256,
        pos_emb_max_len=5000,
        use_bias=True,
        xscaling=True,
    )
    
    decoder = TransformerDecoderArgs(
        n_layers=4,
        d_model=1024,
        n_heads=8,
        ff_expansion_factor=4,
        vocab_size=5248,
        max_seq_len=448,
        dropout=0.1,
    )
    
    return CanaryArgs(
        preprocessor=preprocessor,
        encoder=encoder,
        decoder=decoder,
    )


# Example usage
if __name__ == "__main__":
    # Create model
    config = create_canary_180m_config()
    model = CanaryModel(config)
    
    # Test with dummy audio
    dummy_mel = mx.random.normal((1, 1000, 128))  # (batch, time, mels)
    
    # Test encoding
    encoder_out, lengths = model.encode(dummy_mel)
    print(f"Encoder output shape: {encoder_out.shape}")
    
    # Test generation
    result = model.generate(dummy_mel[0], language="en", task="transcribe")
    print(result)
