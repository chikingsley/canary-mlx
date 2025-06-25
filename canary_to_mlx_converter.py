#!/usr/bin/env python3
"""
Canary-1B-Flash NeMo to MLX Converter

This script converts NVIDIA NeMo Canary-1B-Flash model weights to MLX format
for use on Apple Silicon. It uses Pydantic for robust configuration management
and validation.

Usage:
    python canary_to_mlx_converter.py --nemo-file canary-1b-flash.nemo
"""

import os
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import torch
from safetensors.torch import save_file
from pydantic import BaseModel, Field, validator, ConfigDict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
import typer

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Convert NVIDIA NeMo Canary models to MLX format")


class ConversionConfig(BaseModel):
    """Configuration for the Canary to MLX conversion process."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Input/Output paths
    nemo_file: Path = Field(..., description="Path to the .nemo model file")
    output_dir: Path = Field(default=Path("./canary-mlx"), description="Output directory for converted model")
    extract_dir: Path = Field(default=Path("./canary_nemo_extracted"), description="Temporary extraction directory")
    
    # Model configuration
    checkpoint_name: str = Field(default="model_weights.ckpt", description="Name of checkpoint file in .nemo archive")
    config_name: str = Field(default="model_config.yaml", description="Name of config file in .nemo archive")
    
    # Conversion options
    skip_preprocessor: bool = Field(default=True, description="Skip preprocessor weights")
    skip_batch_norm_tracking: bool = Field(default=True, description="Skip batch norm tracking stats")
    preserve_original_keys: bool = Field(default=False, description="Keep original key names for debugging")
    
    # Output options
    safetensors_name: str = Field(default="model.safetensors", description="Output safetensors filename")
    config_json_name: str = Field(default="config.json", description="Output config JSON filename")
    
    @validator('nemo_file')
    def nemo_file_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"NeMo file does not exist: {v}")
        if not v.suffix == '.nemo':
            raise ValueError(f"File must have .nemo extension: {v}")
        return v
    
    @validator('output_dir', 'extract_dir')
    def create_dirs(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v


class KeyMapper:
    """Handles mapping of NeMo parameter keys to MLX format."""
    
    def __init__(self, preserve_original: bool = False):
        self.preserve_original = preserve_original
        self.mapping_stats = {
            'encoder_keys': 0,
            'decoder_keys': 0,
            'other_keys': 0,
            'skipped_keys': 0
        }
    
    def map_key(self, original_key: str) -> Optional[str]:
        """Map a NeMo key to MLX format."""
        if self._should_skip_key(original_key):
            self.mapping_stats['skipped_keys'] += 1
            return None
        
        new_key = original_key
        
        # Apply encoder mappings
        if self._is_encoder_key(original_key):
            new_key = self._map_encoder_key(new_key)
            self.mapping_stats['encoder_keys'] += 1
        
        # Apply decoder mappings  
        elif self._is_decoder_key(original_key):
            new_key = self._map_decoder_key(new_key)
            self.mapping_stats['decoder_keys'] += 1
        
        else:
            new_key = self._map_other_key(new_key)
            self.mapping_stats['other_keys'] += 1
        
        return new_key
    
    def _should_skip_key(self, key: str) -> bool:
        """Check if key should be skipped during conversion."""
        skip_patterns = [
            "preprocessor",
            "num_batches_tracked",
            "_forward_module",  # Some NeMo internal keys
            "_orig_mod",        # Torch compile artifacts
        ]
        return any(pattern in key for pattern in skip_patterns)
    
    def _is_encoder_key(self, key: str) -> bool:
        """Check if key belongs to encoder."""
        return "encoder" in key and "decoder" not in key
    
    def _is_decoder_key(self, key: str) -> bool:
        """Check if key belongs to decoder."""
        return "decoder" in key
    
    def _map_encoder_key(self, key: str) -> str:
        """Map encoder keys (FastConformer architecture)."""
        # FastConformer encoder mappings (reused from Parakeet)
        mappings = {
            "encoder.layers": "encoder.blocks",
            "self_attn": "attention", 
            "linear_q": "wq",
            "linear_k": "wk", 
            "linear_v": "wv",
            "linear_out": "wo",
            "feed_forward.layer1": "feed_forward.w1",
            "feed_forward.layer2": "feed_forward.w2",
            "norm_mha": "attention_norm",
            "norm_feed_forward": "ffn_norm",
            "norm1": "attention_norm",
            "norm2": "ffn_norm",
        }
        
        for old, new in mappings.items():
            key = key.replace(old, new)
        
        return key
    
    def _map_decoder_key(self, key: str) -> str:
        """Map decoder keys (Transformer architecture)."""
        # Transformer decoder mappings (new for Canary)
        mappings = {
            "decoder.decoder_layers": "decoder.layers",
            "multi_head_attn": "attention",
            "self_attn": "attention",
            "q_proj": "wq",
            "k_proj": "wk", 
            "v_proj": "wv",
            "out_proj": "wo",
            "decoder.final_norm": "decoder.norm",
            "log_softmax": "output",
            "norm1": "attention_norm",
            "norm2": "ffn_norm",
            "mlp.fc1": "feed_forward.w1",
            "mlp.fc2": "feed_forward.w2",
        }
        
        for old, new in mappings.items():
            key = key.replace(old, new)
            
        return key
    
    def _map_other_key(self, key: str) -> str:
        """Map other keys (embeddings, etc.)."""
        # General mappings
        mappings = {
            "word_embeddings": "token_embeddings",
            "position_embeddings": "position_embeddings", 
        }
        
        for old, new in mappings.items():
            key = key.replace(old, new)
            
        return key


class TensorProcessor:
    """Handles tensor shape transformations for MLX compatibility."""
    
    @staticmethod
    def process_tensor(key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor for MLX compatibility."""
        original_shape = tensor.shape
        
        # Convolutional layer permutations
        if "conv" in key and "pointwise" not in key:
            if len(tensor.shape) == 4:  # (out, in, h, w) -> (out, h, w, in)
                tensor = tensor.permute(0, 2, 3, 1)
            elif len(tensor.shape) == 3:  # (out, in, kernel) -> (out, kernel, in)
                tensor = tensor.permute(0, 2, 1)
            elif len(tensor.shape) == 2 and "depthwise" in key:  # Depthwise conv
                tensor = tensor.unsqueeze(1)  # (kernel, groups) -> (kernel, 1, groups)
        
        # LSTM weight handling (if present)
        elif any(lstm_key in key for lstm_key in ["weight_ih_l", "weight_hh_l"]):
            # LSTM weights need special handling for MLX
            pass  # Will implement if needed
        
        # Log shape changes for debugging
        if tensor.shape != original_shape:
            logger.debug(f"Reshaped {key}: {original_shape} -> {tensor.shape}")
        
        return tensor


class CanaryToMLXConverter:
    """Main converter class for Canary NeMo to MLX conversion."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.key_mapper = KeyMapper(preserve_original=config.preserve_original_keys)
        self.tensor_processor = TensorProcessor()
        
    def convert(self) -> bool:
        """Run the complete conversion process."""
        try:
            console.print(f"[bold green]Starting Canary to MLX conversion[/bold green]")
            console.print(f"Input: {self.config.nemo_file}")
            console.print(f"Output: {self.config.output_dir}")
            
            # Step 1: Extract NeMo archive
            checkpoint_path = self._extract_nemo_archive()
            
            # Step 2: Load checkpoint
            state_dict = self._load_checkpoint(checkpoint_path)
            
            # Step 3: Convert weights
            mlx_state_dict = self._convert_weights(state_dict)
            
            # Step 4: Save converted model
            self._save_converted_model(mlx_state_dict)
            
            # Step 5: Print conversion summary
            self._print_summary()
            
            console.print(f"[bold green]✅ Conversion completed successfully![/bold green]")
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Conversion failed: {e}[/bold red]")
            logger.exception("Conversion error details:")
            return False
    
    def _extract_nemo_archive(self) -> Path:
        """Extract the .nemo archive and return path to checkpoint."""
        console.print(f"[yellow]Extracting {self.config.nemo_file.name}...[/yellow]")
        
        with tarfile.open(self.config.nemo_file, 'r') as tar:
            tar.extractall(path=self.config.extract_dir)
        
        checkpoint_path = self.config.extract_dir / self.config.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        console.print(f"[green]✅ Extracted to {self.config.extract_dir}[/green]")
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
        """Load the PyTorch checkpoint."""
        console.print(f"[yellow]Loading checkpoint from {checkpoint_path}...[/yellow]")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if 'state_dict' not in checkpoint:
            raise KeyError("'state_dict' not found in checkpoint")
        
        state_dict = checkpoint['state_dict']
        console.print(f"[green]✅ Loaded {len(state_dict)} tensors[/green]")
        
        return state_dict
    
    def _convert_weights(self, state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
        """Convert NeMo weights to MLX format."""
        console.print(f"[yellow]Converting weights to MLX format...[/yellow]")
        
        mlx_state_dict = OrderedDict()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Converting tensors...", total=len(state_dict))
            
            for original_key, tensor in state_dict.items():
                # Map key to MLX format
                new_key = self.key_mapper.map_key(original_key)
                
                if new_key is None:
                    progress.advance(task)
                    continue
                
                # Process tensor for MLX compatibility
                processed_tensor = self.tensor_processor.process_tensor(new_key, tensor)
                
                mlx_state_dict[new_key] = processed_tensor
                progress.advance(task)
        
        console.print(f"[green]✅ Converted {len(mlx_state_dict)} tensors[/green]")
        return mlx_state_dict
    
    def _save_converted_model(self, mlx_state_dict: OrderedDict):
        """Save the converted model in MLX format."""
        console.print(f"[yellow]Saving converted model...[/yellow]")
        
        # Save weights as safetensors
        safetensors_path = self.config.output_dir / self.config.safetensors_name
        save_file(mlx_state_dict, safetensors_path)
        
        # TODO: Generate and save config.json
        # This would require parsing the original NeMo config and converting it
        
        console.print(f"[green]✅ Saved weights to {safetensors_path}[/green]")
    
    def _print_summary(self):
        """Print conversion summary statistics."""
        stats = self.key_mapper.mapping_stats
        
        console.print("\n[bold]Conversion Summary:[/bold]")
        console.print(f"  Encoder keys: {stats['encoder_keys']}")
        console.print(f"  Decoder keys: {stats['decoder_keys']}")
        console.print(f"  Other keys: {stats['other_keys']}")
        console.print(f"  Skipped keys: {stats['skipped_keys']}")
        console.print(f"  Total processed: {sum(stats.values())}")


@app.command()
def convert(
    nemo_file: Path = typer.Argument(..., help="Path to the .nemo model file"),
    output_dir: Path = typer.Option("./canary-mlx", help="Output directory"),
    extract_dir: Path = typer.Option("./canary_nemo_extracted", help="Temporary extraction directory"),
    preserve_keys: bool = typer.Option(False, help="Preserve original key names for debugging"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging")
):
    """Convert NVIDIA NeMo Canary model to MLX format."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = ConversionConfig(
        nemo_file=nemo_file,
        output_dir=output_dir,
        extract_dir=extract_dir,
        preserve_original_keys=preserve_keys
    )
    
    # Run conversion
    converter = CanaryToMLXConverter(config)
    success = converter.convert()
    
    if not success:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()