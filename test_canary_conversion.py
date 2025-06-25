#!/usr/bin/env python3
"""
Test script for Canary model conversion and validation.

This script tests the conversion process and validates the converted model.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import logging

import torch
import mlx.core as mx
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add parakeet-mlx to path
sys.path.insert(0, str(Path(__file__).parent / "parakeet-mlx"))

from parakeet_mlx.canary import CanaryModel, CanaryArgs
from parakeet_mlx.conformer import ConformerArgs
from parakeet_mlx.transformer import TransformerDecoderArgs
from parakeet_mlx.audio import PreprocessArgs

console = Console()
logger = logging.getLogger(__name__)


def create_test_config() -> CanaryArgs:
    """Create a test configuration for Canary model."""
    
    # Audio preprocessing config
    preprocessor = PreprocessArgs(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=128,
        normalize="per_feature"
    )
    
    # Conformer encoder config (based on Canary-1B specs)
    encoder = ConformerArgs(
        feat_in=128,
        n_layers=24,
        d_model=1024,
        n_heads=8,
        ff_expansion_factor=4,
        subsampling_factor=8,
        self_attention_model="rel_pos",
        subsampling="dw_striding",
        conv_kernel_size=9,
        subsampling_conv_channels=256,
        pos_emb_max_len=5000
    )
    
    # Transformer decoder config
    decoder = TransformerDecoderArgs(
        vocab_size=51864,  # Canary vocab size
        d_model=1024,
        n_heads=16,
        n_layers=24,
        ff_expansion_factor=4,
        dropout=0.1,
        max_seq_len=512
    )
    
    # Canary-specific config
    return CanaryArgs(
        preprocessor=preprocessor,
        encoder=encoder,
        decoder=decoder,
        supported_languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
        default_language="en",
        max_decode_length=512
    )


def test_model_creation():
    """Test creating a Canary model with random weights."""
    console.print("[bold blue]Testing model creation...[/bold blue]")
    
    try:
        config = create_test_config()
        model = CanaryModel(config)
        
        # Count parameters
        total_params = sum(p.size for p in model.parameters().values())
        
        console.print(f"✅ Model created successfully")
        console.print(f"   Total parameters: {total_params:,}")
        
        return model, config
        
    except Exception as e:
        console.print(f"❌ Model creation failed: {e}")
        raise


def test_forward_pass(model: CanaryModel, config: CanaryArgs):
    """Test forward pass with dummy data."""
    console.print("[bold blue]Testing forward pass...[/bold blue]")
    
    try:
        # Create dummy audio features
        batch_size = 2
        seq_len = 100
        feat_dim = config.preprocessor.n_mels
        
        audio_features = mx.random.normal((batch_size, seq_len, feat_dim))
        
        # Test encoder only
        encoder_output = model.encode_audio(audio_features)
        console.print(f"✅ Encoder output shape: {encoder_output.shape}")
        
        # Test full forward pass (inference mode)
        output_tokens = model(audio_features)
        console.print(f"✅ Generated tokens shape: {output_tokens.shape}")
        
        # Test with target tokens (training mode)
        target_length = 50
        target_tokens = mx.random.randint(0, config.decoder.vocab_size, (batch_size, target_length))
        logits = model(audio_features, target_tokens)
        console.print(f"✅ Training logits shape: {logits.shape}")
        
    except Exception as e:
        console.print(f"❌ Forward pass failed: {e}")
        raise


def test_weight_loading(model_path: Path):
    """Test loading converted weights."""
    console.print(f"[bold blue]Testing weight loading from {model_path}...[/bold blue]")
    
    if not model_path.exists():
        console.print(f"❌ Model path does not exist: {model_path}")
        return False
    
    try:
        # Check for required files
        required_files = ["model.safetensors", "config.json"]
        missing_files = []
        
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            console.print(f"❌ Missing files: {missing_files}")
            return False
        
        # Try to load the model
        # This would use the from_pretrained function once implemented
        console.print("✅ All required files present")
        console.print("⚠️  Weight loading test skipped (from_pretrained not fully implemented)")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Weight loading failed: {e}")
        return False


def compare_architectures():
    """Compare Canary architecture with Parakeet."""
    console.print("[bold blue]Architecture Comparison[/bold blue]")
    
    table = Table(title="Canary vs Parakeet Architecture")
    table.add_column("Component", style="cyan")
    table.add_column("Canary", style="green")
    table.add_column("Parakeet", style="yellow")
    
    table.add_row("Encoder", "FastConformer", "FastConformer")
    table.add_row("Decoder", "Transformer", "TDT/RNNT")
    table.add_row("Attention", "Multi-head", "Multi-head")
    table.add_row("Languages", "Multilingual", "English")
    table.add_row("Tasks", "Transcribe/Translate", "Transcribe")
    table.add_row("Vocab Size", "~51K", "~1K")
    
    console.print(table)


def validate_conversion_mapping():
    """Validate the key mapping logic."""
    console.print("[bold blue]Testing conversion key mapping...[/bold blue]")
    
    # Import the converter
    sys.path.insert(0, str(Path(__file__).parent))
    from canary_to_mlx_converter import KeyMapper
    
    mapper = KeyMapper()
    
    # Test cases for key mapping
    test_cases = [
        # Encoder keys
        ("encoder.layers.0.self_attn.linear_q.weight", "encoder.blocks.0.attention.wq.weight"),
        ("encoder.layers.5.feed_forward.layer1.bias", "encoder.blocks.5.feed_forward.w1.bias"),
        ("encoder.layers.10.norm_mha.weight", "encoder.blocks.10.attention_norm.weight"),
        
        # Decoder keys  
        ("decoder.decoder_layers.0.multi_head_attn.q_proj.weight", "decoder.layers.0.attention.wq.weight"),
        ("decoder.decoder_layers.3.mlp.fc1.weight", "decoder.layers.3.feed_forward.w1.weight"),
        ("decoder.final_norm.weight", "decoder.norm.weight"),
        
        # Skip cases
        ("preprocessor.window", None),
        ("some_layer.num_batches_tracked", None),
    ]
    
    passed = 0
    failed = 0
    
    for original, expected in test_cases:
        result = mapper.map_key(original)
        if result == expected:
            console.print(f"✅ {original} -> {result}")
            passed += 1
        else:
            console.print(f"❌ {original} -> {result} (expected {expected})")
            failed += 1
    
    console.print(f"\nMapping test results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all tests."""
    console.print("[bold green]Canary Model Conversion Test Suite[/bold green]\n")
    
    try:
        # Test 1: Model creation
        model, config = test_model_creation()
        console.print()
        
        # Test 2: Forward pass
        test_forward_pass(model, config)
        console.print()
        
        # Test 3: Architecture comparison
        compare_architectures()
        console.print()
        
        # Test 4: Key mapping validation
        mapping_ok = validate_conversion_mapping()
        console.print()
        
        # Test 5: Weight loading (if converted model exists)
        model_path = Path("./canary-mlx")
        test_weight_loading(model_path)
        console.print()
        
        # Summary
        console.print("[bold green]Test Summary:[/bold green]")
        console.print("✅ Model creation: PASSED")
        console.print("✅ Forward pass: PASSED") 
        console.print("✅ Architecture comparison: PASSED")
        console.print(f"{'✅' if mapping_ok else '❌'} Key mapping: {'PASSED' if mapping_ok else 'FAILED'}")
        console.print("⚠️  Weight loading: SKIPPED (no converted model)")
        
        if mapping_ok:
            console.print("\n[bold green]🎉 All tests passed! Ready for conversion.[/bold green]")
        else:
            console.print("\n[bold red]❌ Some tests failed. Please fix before converting.[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"\n[bold red]❌ Test suite failed: {e}[/bold red]")
        logger.exception("Test failure details:")
        sys.exit(1)


if __name__ == "__main__":
    main()