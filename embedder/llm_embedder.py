"""
LLM Embedder - Extract hidden state embeddings from Large Language Models.

This module provides a flexible interface for extracting embeddings from LLMs
by passing text through the model and capturing the last hidden state. This
is fundamentally different from text embeddings (BERT-style) which use encoder
models - here we use decoder models (GPT-2, Qwen, LLaMA) and extract internal
representations.

ARCHITECTURE OVERVIEW:
======================

The LLM embedding system is designed with multiple input sources in mind:

1. TIME SERIES → PROMPTS (Current: TimeCMA)
   - Time series values are converted to text prompts via PromptBuilder
   - Prompts describe the values, trends, and temporal context
   - Used by: TimeCMA and similar models

2. RAW TEXT → EMBEDDINGS (Future: News, Reports)
   - Pre-existing text (news articles, weather reports) passed directly to LLM
   - Captures deeper semantic understanding than BERT encoders
   - Use case: When you want LLM-quality understanding of text
   
3. MULTIMODAL COMBINATIONS (Future)
   - Combine time series context with external text
   - Example: "Given weather: {news}, the values were: {ts_values}"

The key abstraction is that ALL inputs become text strings before LLM processing.
The LLMEmbedder handles the LLM inference; input preparation is handled by:
- PromptBuilder: for time series → text conversion
- Direct text: for pre-existing text sources (future)
- Custom adapters: for domain-specific conversions (future)

IMPORTANT: This embedder is designed for PRECOMPUTATION only.
Embeddings should be generated before training, not on-the-fly.
Use `cli/inference.py` to run precomputation.

DATASET CONSIDERATIONS:
======================

Different dataset types have different structures:

1. Time-MMD Datasets (e.g., ETTh1, ETTm1, Weather):
   - Simple CSV structure with timestamp + features
   - Embeddings indexed by sample index
   - Standard data loading via data_provider

2. Fidel-TS Datasets (e.g., Bear_room, California_ISO):
   - Complex nested directory structures per subdataset
   - Hardcoded path resolution via FidelTSPathResolver
   - May have existing text embeddings in hetero/ directories
   - LLM embeddings stored separately in llm_embeddings/

Example:
    # Time series → prompts (TimeCMA style)
    embedder = LLMEmbedder.from_config('configs/llm_embedding/default.yaml')
    embedder.generate_ts_embeddings(dataset='ETTh1', split='train')
    
    # Raw text → embeddings (future)
    embeddings = embedder.embed_texts(["Weather is sunny", "Storm approaching"])
"""

import gc
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Iterator, Tuple, TYPE_CHECKING

import numpy as np
import torch

from .llm_registry import LLMRegistry
from .llm_cache import LLMEmbeddingCache, LLMEmbeddingMetadata, StreamingEmbeddingWriter
from .prompt_builder import TSPromptBuilder

if TYPE_CHECKING:
    from rich.console import Console


class LLMEmbedder:
    """
    Extract hidden state embeddings from Large Language Models.
    
    This class provides the core LLM inference functionality for extracting
    embeddings from text. It supports multiple input modalities:
    
    1. TIME SERIES INPUT (via TSPromptBuilder):
       - Converts time series to text prompts
       - Uses templates like TimeCMA format
       - Primary use case for forecasting models
       - Method: generate_ts_embeddings()
    
    2. RAW TEXT INPUT (direct):
       - Pass text strings directly to LLM
       - Useful for news, reports, descriptions
       - Future extension for richer multimodal models
       - Method: embed_texts()
    
    3. CUSTOM INPUT SOURCES (extensible):
       - Subclass or compose with custom input adapters
       - Any data that can be converted to text strings
    
    The core operation is always: text → LLM → hidden state → embedding
    
    Attributes:
        model_name: HuggingFace model name
        device: Target device for inference
        extraction_mode: 'last_token' or 'pooled'
        ts_prompt_builder: TSPromptBuilder for time series conversion (optional)
    
    Example:
        # For time series (TimeCMA style)
        embedder = LLMEmbedder(model_name='gpt2', prompt_template='timecma_v1')
        embeddings = embedder.generate_ts_embeddings('ETTh1', 'train')
        
        # For raw text (future use)
        embeddings = embedder.embed_texts(["text1", "text2"])
    """
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        device: str = 'cuda:0',
        cache_dir: str = './LLM_cache/',
        data_root: str = './data/',
        quantization: Optional[str] = None,
        extraction_mode: str = 'last_token',
        prompt_template: Optional[str] = 'timecma_v1',
        prompt_config: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
        # Experiment-driven parameters (from experiment config)
        input_len: int = 96,
        output_len: int = 96,
        scale: bool = True,
        data_config_path: Optional[str] = None,
        console: Optional['Console'] = None,
    ):
        """
        Initialize LLM embedder.
        
        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'Qwen/Qwen2.5-72B-Instruct')
            device: Target device ('cuda:0', 'cpu', etc.)
            cache_dir: Directory for LLM model weights cache
            data_root: Root directory for datasets
            quantization: Quantization mode ('4bit', '8bit', or None)
            extraction_mode: Embedding extraction method:
                - 'last_token': Use last non-padding token's hidden state (default)
                - 'pooled': Mean pooling over all non-padding tokens
            prompt_template: Prompt template name for time series conversion.
                - 'timecma_v1': TimeCMA paper format
                - 'simple': Minimal format
                - None: No prompt builder (for raw text input only)
            prompt_config: Configuration dict for prompt template
            max_length: Maximum token length for inputs
            input_len: Input sequence length (from experiment config)
            output_len: Output/prediction sequence length (from experiment config)
            scale: Whether to scale the data (from experiment config)
            data_config_path: Path to data config YAML (from experiment config)
            console: Optional Rich Console for progress bar display
        
        Note:
            If prompt_template is None, the embedder can only be used for raw
            text input via embed_texts(). Time series methods will raise errors.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.data_root = data_root
        self.quantization = quantization
        self.extraction_mode = extraction_mode
        self.prompt_template = prompt_template
        self.prompt_config = prompt_config or {}
        self.max_length = max_length
        self.console = console  # Rich Console for progress bars
        
        # Experiment-driven parameters
        self.input_len = input_len
        self.output_len = output_len
        self.scale = scale
        self.data_config_path = data_config_path
        
        # Initialize time series prompt builder (optional - only for TS input)
        # If prompt_template is None, this embedder is for raw text only
        if prompt_template is not None:
            self.ts_prompt_builder = TSPromptBuilder(
                template_name=prompt_template,
                template_config=prompt_config
            )
        else:
            self.ts_prompt_builder = None
        
        # Default batch size (can be overridden by from_experiment_config or at call time)
        self.default_batch_size = 16
        
        # Lazy load model and tokenizer
        self._model = None
        self._tokenizer = None
        self._embed_dim = None
    
    @classmethod
    def from_experiment_config(
        cls,
        experiment_config_path: str,
        device: Optional[str] = None,
        console: Optional['Console'] = None,
    ) -> 'LLMEmbedder':
        """
        Create embedder from experiment configuration file.
        
        This is the preferred method for creating an LLMEmbedder. It extracts
        all required parameters from the experiment config, ensuring consistency
        between training and embedding generation.
        
        Args:
            experiment_config_path: Path to experiment YAML configuration file
            device: Override device from config (defaults to cuda:0 or experiment's GPU)
            console: Optional Rich Console for progress bar display
        
        Returns:
            Configured LLMEmbedder instance
        
        Raises:
            ValueError: If experiment config doesn't have llm_embedding section
        
        Example:
            embedder = LLMEmbedder.from_experiment_config(
                'configs/experiments/timecma_test.yaml'
            )
            embedder.generate_ts_embeddings('time_mmd_traffic', 'train')
        """
        with open(experiment_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract llm_embedding section (required)
        llm_config = config.get('llm_embedding')
        if llm_config is None:
            raise ValueError(
                f"Experiment config '{experiment_config_path}' does not have an 'llm_embedding' section. "
                f"Add llm_embedding configuration to generate LLM embeddings."
            )
        
        # Extract training parameters
        training = config.get('training', {})
        input_len = training.get('input_len', 96)
        output_len = training.get('output_len', 96)
        scale = training.get('scale', True)
        
        # Extract data config path
        data = config.get('data', {})
        data_config_path = data.get('config_path')
        
        # Extract data_root from base_data_path or default
        base_data_path = config.get('base_data_path', './data/')
        
        # Determine device: CLI override > experiment config > default
        if device is None:
            device_config = config.get('device', {})
            gpu = device_config.get('gpu', 0)
            use_gpu = device_config.get('use_gpu', True)
            device = f'cuda:{gpu}' if use_gpu else 'cpu'
        
        embedder = cls(
            model_name=llm_config.get('model_name', 'gpt2'),
            device=device,
            cache_dir=llm_config.get('cache_dir', './LLM_cache/'),
            data_root=base_data_path,
            quantization=llm_config.get('quantization'),
            extraction_mode=llm_config.get('extraction_mode', 'last_token'),
            prompt_template=llm_config.get('prompt_template', 'timecma_v1'),
            prompt_config=llm_config.get('prompt_config', {'value_format': 'integer', 'include_timestamps': True}),
            max_length=llm_config.get('max_length', 512),
            input_len=input_len,
            output_len=output_len,
            scale=scale,
            data_config_path=data_config_path,
            console=console,
        )
        
        # Store batch_size from config for use by generate_ts_embeddings
        embedder.default_batch_size = llm_config.get('batch_size', 64)
        
        # Store experiment config path for reference
        embedder.experiment_config_path = experiment_config_path
        
        return embedder
    
    @classmethod
    def for_raw_text(
        cls,
        model_name: str = 'gpt2',
        device: str = 'cuda:0',
        cache_dir: str = './LLM_cache/',
        quantization: Optional[str] = None,
        extraction_mode: str = 'last_token',
        max_length: int = 512,
    ) -> 'LLMEmbedder':
        """
        Create an embedder configured for raw text input only.
        
        This factory method creates an LLMEmbedder without a PromptBuilder,
        suitable for embedding pre-existing text (news, reports, etc.)
        rather than generating prompts from time series.
        
        Args:
            model_name: HuggingFace model name
            device: Target device
            cache_dir: LLM model weights cache directory
            quantization: Quantization mode
            extraction_mode: 'last_token' or 'pooled'
            max_length: Maximum token length
        
        Returns:
            LLMEmbedder configured for raw text embedding
        
        Example:
            embedder = LLMEmbedder.for_raw_text(model_name='gpt2')
            embeddings = embedder.embed_texts(["News article 1", "News article 2"])
        
        Note:
            This is a FUTURE EXTENSION point. The core infrastructure is in place,
            but integration with specific text sources (news, weather reports)
            should be implemented as needed.
        """
        return cls(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            data_root='',  # Not used for raw text
            quantization=quantization,
            extraction_mode=extraction_mode,
            prompt_template=None,  # No prompt builder
            prompt_config=None,
            max_length=max_length,
        )
    
    def _ensure_model_loaded(self):
        """Load model and tokenizer if not already loaded."""
        if self._model is None:
            self._model, self._embed_dim = LLMRegistry.get_model(
                model_name=self.model_name,
                device=self.device,
                cache_dir=self.cache_dir,
                quantization=self.quantization,
            )
            self._tokenizer = LLMRegistry.get_tokenizer(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
            )
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (loads model if needed)."""
        if self._embed_dim is None:
            self._embed_dim = LLMRegistry.get_embedding_dim(self.model_name)
        return self._embed_dim
    
    # =========================================================================
    # CORE EMBEDDING METHODS
    # These are the fundamental operations - all input types eventually use these
    # =========================================================================
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a list of text strings using the LLM.
        
        This is the CORE embedding method. All other methods (time series,
        structured data, etc.) ultimately convert their input to text and
        call this method.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for LLM inference
        
        Returns:
            embeddings: np.ndarray of shape [N, embed_dim]
        
        Example:
            embeddings = embedder.embed_texts([
                "Weather is sunny with clear skies",
                "Heavy rain expected tomorrow"
            ])
            # Returns: [2, 768] for GPT-2
        
        Note:
            This method is the foundation for future extensions. To add
            new input types (news, reports, custom formats), create an
            adapter that converts your data to text strings, then call
            this method.
        """
        self._ensure_model_loaded()
        
        all_embeddings = []
        
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Tokenize batch
            inputs = self._tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                
                if self.extraction_mode == 'last_token':
                    # Get last non-padding token for each sequence
                    seq_lengths = inputs.attention_mask.sum(dim=1) - 1
                    batch_embeddings = hidden_states[
                        torch.arange(hidden_states.size(0), device=self.device),
                        seq_lengths
                    ]
                else:  # pooled
                    # Mean pooling over non-padding tokens
                    mask = inputs.attention_mask.unsqueeze(-1).float()
                    batch_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def embed_text_dict(
        self,
        text_dict: Dict[str, str],
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        Embed a dictionary of texts (key → text mapping).
        
        Useful for timestamp-keyed text data or channel-specific descriptions.
        
        Args:
            text_dict: Dictionary mapping keys to text strings
            batch_size: Batch size for LLM inference
        
        Returns:
            Dictionary mapping keys to embedding arrays [embed_dim]
        
        Example:
            embeddings = embedder.embed_text_dict({
                "2023-01-01": "Weather report for January 1st...",
                "2023-01-02": "Weather report for January 2nd..."
            })
        
        Note:
            This is a FUTURE EXTENSION point for embedding pre-existing text
            data (weather reports, news articles) through an LLM instead of
            BERT. The infrastructure is in place; specific integrations with
            Fidel-TS text sources can be added as needed.
        """
        keys = list(text_dict.keys())
        texts = [text_dict[k] for k in keys]
        
        embeddings = self.embed_texts(texts, batch_size=batch_size)
        
        return {k: embeddings[i] for i, k in enumerate(keys)}
    
    # =========================================================================
    # TIME SERIES EMBEDDING METHODS
    # These methods use PromptBuilder to convert time series to text
    # =========================================================================
    
    def generate_ts_embeddings(
        self,
        dataset: str,
        split: str,
        batch_size: int = 32,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        Generate LLM embeddings from time series data (TS → prompts → embeddings).
        
        This method implements the TimeCMA-style workflow:
        1. Load time series data from dataset
        2. Convert each (sample, channel) to a text prompt via TSPromptBuilder
        3. Run batched LLM inference to extract hidden states
        4. Cache results for training
        
        The "ts" in the name emphasizes this is for TIME SERIES input.
        For raw text input, use embed_texts() instead.
        
        Args:
            dataset: Dataset name (e.g., 'ETTh1' for Time-MMD, 'Bear_room' for Fidel-TS)
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size for LLM inference
            force: Force regeneration even if cache exists
            progress_callback: Optional callback(current, total) for progress updates
        
        Returns:
            Embeddings array [N, embed_dim, C] where:
                - N = number of samples
                - embed_dim = LLM hidden dimension
                - C = number of channels
        
        Raises:
            ValueError: If ts_prompt_builder is None (use embed_texts for raw text)
        
        Note:
            This method handles both Time-MMD datasets (simple structure) and
            Fidel-TS datasets (complex structure). The data loading adapts to
            the dataset type automatically.
        """
        if self.ts_prompt_builder is None:
            raise ValueError(
                "Cannot generate time series embeddings without a TSPromptBuilder. "
                "Either set prompt_template in __init__, or use embed_texts() for raw text."
            )
        
        # Get actual data directory from config (e.g., './data/time_mmd/Climate' for 'time_mmd_climate')
        data_dir = self._get_data_directory(dataset)
        
        # Create cache manager with actual data directory
        cache = LLMEmbeddingCache(str(data_dir), dataset)
        
        # Build metadata for cache key
        metadata = self._build_metadata(dataset)
        
        # Check cache - return early if cache exists (WITHOUT loading full array into memory!)
        if not force and cache.cache_exists(metadata, split):
            cache_info = cache.get_cache_info(metadata, split)
            if cache_info is not None:
                num_samples, embed_dim, num_channels = cache_info
                if self.console is not None:
                    self.console.print(f"  [dim]Cache exists for {dataset}/{split}: {num_samples} embeddings[/dim]")
                # Return None - embeddings will be loaded by LLMEmbeddingProvider during training
                return None
        
        # Load data with progress display
        values, timestamps, data_metadata = self._load_dataset_with_progress(dataset, split)
        
        N, L, C = values.shape
        total_prompts = N * C
        
        # Ensure model loaded with progress display
        self._ensure_model_loaded_with_progress()
        
        # Generate embeddings with full progress bar
        start_time = time.time()
        embeddings = self._generate_embeddings_from_timeseries(
            values=values,
            timestamps=timestamps,
            metadata=data_metadata,
            batch_size=batch_size,
            progress_callback=progress_callback,
            split=split,
        )
        generation_time = time.time() - start_time
        
        # Print completion message with time taken
        if self.console is not None:
            self.console.print(f"  [dim]Generated {N} embeddings in {generation_time:.1f}s[/dim]")
        
        # Update metadata with generation info
        metadata.generation_time_seconds = generation_time
        metadata.num_samples = N
        metadata.num_channels = C
        metadata.seq_len = L
        metadata.device = self.device
        metadata.prompt_example = self.ts_prompt_builder.get_example_prompt(
            values, timestamps, data_metadata
        )
        
        # Save to cache (with progress display if console available)
        if self.console is not None:
            self.console.print(f"  [dim]Saving {N} embeddings to cache...[/dim]")
        cache.save_embeddings(embeddings, metadata, split)
        
        return embeddings
    
    def _load_dataset_with_progress(
        self,
        dataset: str,
        split: str,
    ) -> tuple:
        """
        Load dataset with progress indication via Rich console.
        
        Wraps _load_dataset with a spinner/status indicator if console is available.
        
        Args:
            dataset: Dataset name
            split: Data split
        
        Returns:
            Tuple of (values, timestamps, metadata) from _load_dataset
        """
        if self.console is not None:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"Loading {dataset}/{split}...", total=None)
                result = self._load_dataset(dataset, split)
                progress.update(task, completed=True)
            
            N, L, C = result[0].shape
            self.console.print(f"  [dim]Loaded {N} samples ({C} channels, seq_len={L})[/dim]")
            return result
        else:
            # Fallback: print statement
            print(f"[ LLM Embedder ] Loading data for {dataset}/{split}")
            result = self._load_dataset(dataset, split)
            print(f"[ LLM Embedder ] Loaded {result[0].shape[0]} samples")
            return result
    
    def _ensure_model_loaded_with_progress(self):
        """
        Load model and tokenizer with progress indication via Rich console.
        
        Shows a spinner while loading the model if console is available.
        """
        if self._model is not None:
            return  # Already loaded
        
        if self.console is not None:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"Loading model {self.model_name}...", total=None)
                self._ensure_model_loaded()
                progress.update(task, completed=True)
            
            self.console.print(f"  [dim]Model loaded (embed_dim={self._embed_dim})[/dim]")
        else:
            # Fallback: use standard _ensure_model_loaded
            self._ensure_model_loaded()
    
    def _generate_embeddings_from_timeseries(
        self,
        values: np.ndarray,
        timestamps: Optional[np.ndarray],
        metadata: Dict[str, Any],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        split: str = '',
    ) -> np.ndarray:
        """
        Generate LLM embeddings from time series data.
        
        Converts time series to prompts using PromptBuilder, then extracts
        embeddings via the LLM.
        
        OPTIMIZATION: Flattens all (sample, channel) pairs into a single
        batch dimension for efficient GPU inference.
        
        Args:
            values: [N, seq_len, C] time series values
            timestamps: [N, seq_len, features] or None
            metadata: Dataset metadata dict (contains 'freq', etc.)
            batch_size: Batch size for inference
            progress_callback: Optional progress callback
            split: Data split name (for progress display)
        
        Returns:
            embeddings: [N, embed_dim, C] LLM embeddings
        """
        N, L, C = values.shape
        
        # Step 1: Generate ALL prompts (flattened across samples and channels)
        if self.console is not None:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task("Building prompts...", total=None)
                all_prompts = self.ts_prompt_builder.build_flat_prompts(values, timestamps, metadata)
                progress.update(task, completed=True)
        else:
            all_prompts = self.ts_prompt_builder.build_flat_prompts(values, timestamps, metadata)
        
        total_prompts = len(all_prompts)  # N * C
        num_batches = (total_prompts + batch_size - 1) // batch_size
        
        # Step 2: Embed all prompts with progress bar
        all_embeddings = []
        
        if self.console is not None:
            from rich.progress import (
                Progress, BarColumn, TextColumn, TimeElapsedColumn, 
                TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
            )
            
            # Create a visually distinct progress bar for embedding generation
            progress_columns = (
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            )
            
            with Progress(*progress_columns, console=self.console, transient=False) as progress:
                task = progress.add_task(
                    f"Embedding {split}",
                    total=num_batches
                )
                
                for start_idx in range(0, total_prompts, batch_size):
                    end_idx = min(start_idx + batch_size, total_prompts)
                    batch_prompts = all_prompts[start_idx:end_idx]
                    
                    # Tokenize batch
                    inputs = self._tokenizer(
                        batch_prompts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.max_length
                    ).to(self.device)
                    
                    # Get embeddings
                    with torch.no_grad():
                        outputs = self._model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]  # Last layer
                        
                        if self.extraction_mode == 'last_token':
                            seq_lengths = inputs.attention_mask.sum(dim=1) - 1
                            batch_embeddings = hidden_states[
                                torch.arange(hidden_states.size(0), device=self.device),
                                seq_lengths
                            ]
                        else:  # pooled
                            mask = inputs.attention_mask.unsqueeze(-1).float()
                            batch_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
                    
                    all_embeddings.append(batch_embeddings.cpu().numpy())
                    
                    # Update progress bar
                    progress.advance(task)
                    
                    # Also call external callback if provided
                    if progress_callback:
                        progress_callback(end_idx, total_prompts)
        else:
            # Fallback: no progress bar, use callback if provided
            for start_idx in range(0, total_prompts, batch_size):
                end_idx = min(start_idx + batch_size, total_prompts)
                batch_prompts = all_prompts[start_idx:end_idx]
                
                # Tokenize batch
                inputs = self._tokenizer(
                    batch_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self._model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # Last layer
                    
                    if self.extraction_mode == 'last_token':
                        seq_lengths = inputs.attention_mask.sum(dim=1) - 1
                        batch_embeddings = hidden_states[
                            torch.arange(hidden_states.size(0), device=self.device),
                            seq_lengths
                        ]
                    else:  # pooled
                        mask = inputs.attention_mask.unsqueeze(-1).float()
                        batch_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
                
                all_embeddings.append(batch_embeddings.cpu().numpy())
                
                # Progress callback
                if progress_callback:
                    progress_callback(end_idx, total_prompts)
        
        # Step 3: Concatenate and reshape to [N, embed_dim, C]
        flat_embeddings = np.concatenate(all_embeddings, axis=0)  # [N*C, embed_dim]
        embeddings = flat_embeddings.reshape(N, C, -1).transpose(0, 2, 1)  # [N, embed_dim, C]
        
        return embeddings
    
    def _build_metadata(self, dataset: str) -> LLMEmbeddingMetadata:
        """Build metadata object for caching."""
        if self.ts_prompt_builder is not None:
            template_info = self.ts_prompt_builder.get_template_info()
            prompt_template = template_info['template_name']
            prompt_version = template_info['template_version']
            prompt_config = template_info['template_config']
        else:
            prompt_template = 'raw_text'
            prompt_version = '1.0.0'
            prompt_config = {}
        
        return LLMEmbeddingMetadata(
            model_name=self.model_name,
            extraction_mode=self.extraction_mode,
            embedding_dim=self._embed_dim or LLMRegistry.get_embedding_dim(self.model_name),
            quantization=self.quantization,
            prompt_template=prompt_template,
            prompt_template_version=prompt_version,
            prompt_config=prompt_config,
            dataset_name=dataset,
        )
    
    def _resolve_dataset_config(self, dataset: str) -> Path:
        """
        Resolve dataset name to its configuration file path.
        
        This method supports three dataset types with distinct naming conventions:
        
        1. Time-MMD datasets (time_mmd_<domain>):
           - Config: data_configs/time_mmd/<Domain>/config.yaml
           - Example: time_mmd_traffic -> data_configs/time_mmd/Traffic/config.yaml
        
        2. TTC datasets (ttc_<domain>):
           - Config: data_configs/ttc/<domain>/config.yaml
           - Example: ttc_climate -> data_configs/ttc/climate/config.yaml
        
        3. Fidel-TS datasets (fidel_<dataset> or fidel_<dataset>:<config>):
           - Config: data_configs/<Dataset>/<config>.yaml
           - Example: fidel_ETT -> data_configs/ETT/fullETT_H.yaml (default)
           - Example: fidel_ETT:fullETT_M -> data_configs/ETT/fullETT_M.yaml
        
        Args:
            dataset: Dataset identifier string
        
        Returns:
            Path to the configuration YAML file
        
        Raises:
            FileNotFoundError: If the configuration file does not exist
            ValueError: If the dataset type cannot be determined
        """
        
        # ==========================================================================
        # Time-MMD datasets: time_mmd_<domain>
        # ==========================================================================
        if dataset.startswith('time_mmd_'):
            domain = dataset.replace('time_mmd_', '')
            # Handle case variants: traffic -> Traffic, public_health -> Public_Health
            domain_parts = domain.split('_')
            domain_capitalized = '_'.join(p.capitalize() for p in domain_parts)
            config_path = Path(f'data_configs/time_mmd/{domain_capitalized}/config.yaml')
            
            if not config_path.exists():
                # List available domains for helpful error message
                time_mmd_dir = Path('data_configs/time_mmd')
                available = []
                if time_mmd_dir.exists():
                    available = [d.name for d in time_mmd_dir.iterdir() 
                                if d.is_dir() and (d / 'config.yaml').exists()]
                raise FileNotFoundError(
                    f"Time-MMD dataset config not found: {config_path}\n"
                    f"Available Time-MMD domains: {available}"
                )
            return config_path
        
        # ==========================================================================
        # TTC datasets: ttc_<domain>
        # ==========================================================================
        if dataset.startswith('ttc_'):
            domain = dataset.replace('ttc_', '')
            config_path = Path(f'data_configs/ttc/{domain}/config.yaml')
            
            if not config_path.exists():
                # List available domains for helpful error message
                ttc_dir = Path('data_configs/ttc')
                available = []
                if ttc_dir.exists():
                    available = [d.name for d in ttc_dir.iterdir() 
                                if d.is_dir() and (d / 'config.yaml').exists()]
                raise FileNotFoundError(
                    f"TTC dataset config not found: {config_path}\n"
                    f"Available TTC domains: {available}"
                )
            return config_path
        
        # ==========================================================================
        # Fidel-TS datasets: fidel_<dataset> or fidel_<dataset>:<config_name>
        # ==========================================================================
        if dataset.startswith('fidel_'):
            parts = dataset.replace('fidel_', '').split(':')
            dataset_name = parts[0]
            
            # Default config mappings for Fidel-TS datasets
            # Maps dataset folder name to default config file (without .yaml)
            default_configs = {
                'Bear_room': 'fullBear',
                'California_ISO': 'fullCAISO',
                'Canada_photovoltaics_plants': 'fullCPP',
                'electricity': 'fullelectricity',
                'ETT': 'fullETT_H',
                'Germany_Renewable_Power_Grid': 'fullGRPG',
                'Jena_Atmospheric_Physics': 'fullJAP',
                'NYC_traffic_speed': 'fullNYCTS',
                'traffic': 'fulltraffic',
                'weather': 'weather',
            }
            
            if len(parts) > 1:
                # Explicit config specified: fidel_ETT:fullETT_M
                config_name = parts[1]
                # Remove .yaml if user included it
                if config_name.endswith('.yaml'):
                    config_name = config_name[:-5]
            elif dataset_name in default_configs:
                # Use default config
                config_name = default_configs[dataset_name]
            else:
                raise ValueError(
                    f"Unknown Fidel-TS dataset: {dataset_name}\n"
                    f"Available Fidel-TS datasets: {list(default_configs.keys())}\n"
                    f"Specify config explicitly: fidel_{dataset_name}:<config_name>"
                )
            
            config_path = Path(f'data_configs/{dataset_name}/{config_name}.yaml')
            
            if not config_path.exists():
                # List available configs for this dataset
                dataset_dir = Path(f'data_configs/{dataset_name}')
                available = []
                if dataset_dir.exists():
                    available = [f.stem for f in dataset_dir.glob('*.yaml')]
                raise FileNotFoundError(
                    f"Fidel-TS dataset config not found: {config_path}\n"
                    f"Available configs for {dataset_name}: {available}"
                )
            return config_path
        
        # ==========================================================================
        # Fallback: Try direct path interpretation
        # ==========================================================================
        # Try as a direct config path (for backwards compatibility or custom paths)
        if dataset.endswith('.yaml'):
            config_path = Path(dataset)
            if config_path.exists():
                return config_path
        
        raise ValueError(
            f"Unknown dataset format: {dataset}\n"
            f"Supported formats:\n"
            f"  - time_mmd_<domain>: Time-MMD datasets (e.g., time_mmd_traffic)\n"
            f"  - ttc_<domain>: TTC datasets (e.g., ttc_climate)\n"
            f"  - fidel_<dataset>: Fidel-TS datasets (e.g., fidel_ETT)\n"
            f"  - fidel_<dataset>:<config>: Fidel-TS with specific config\n"
            f"Use 'python -m cli.inference list-datasets' to see available datasets."
        )
    
    def _get_data_directory(self, dataset: str) -> Path:
        """
        Get the actual data directory path for a dataset.
        
        This resolves the dataset name to its actual data directory by reading
        the `root_path` from the dataset's config file. This is important for
        storing LLM embeddings in the correct location alongside the data.
        
        Args:
            dataset: Dataset identifier string (e.g., 'time_mmd_climate')
        
        Returns:
            Path to the actual data directory (e.g., './data/time_mmd/Climate')
        
        Example:
            >>> embedder._get_data_directory('time_mmd_climate')
            Path('./data/time_mmd/Climate')
        """
        import yaml
        
        # Get the config file path
        config_path = self._resolve_dataset_config(dataset)
        
        # Read the config to get root_path
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get root_path from config (this is the actual data directory)
        root_path = data_config.get('root_path')
        
        if not root_path:
            raise ValueError(
                f"Dataset config {config_path} is missing 'root_path'. "
                f"Cannot determine data directory for LLM embedding cache."
            )
        
        return Path(root_path)
    
    def _load_dataset(
        self, 
        dataset: str, 
        split: str
    ) -> tuple:
        """
        Load dataset values and timestamps.
        
        This method handles three dataset types:
        
        1. Time-MMD datasets (time_mmd_<domain>):
           - Simple CSV structure with text columns
           - Uses TimeMMD_Dataset via Data_Provider
        
        2. TTC datasets (ttc_<domain>):
           - Similar structure to Time-MMD with text descriptions
           - Uses TimeMMD_Dataset via Data_Provider
        
        3. Fidel-TS datasets (fidel_<dataset>):
           - Complex nested directory structures
           - Path resolution via FidelTSPathResolver for embeddings
           - Uses Universal_Dataset via Data_Provider
        
        Args:
            dataset: Dataset name (e.g., 'time_mmd_traffic', 'ttc_climate', 'fidel_ETT')
            split: Data split ('train', 'val', 'test')
        
        Returns:
            Tuple of (values, timestamps, metadata)
            - values: [N, seq_len, channels]
            - timestamps: [N, seq_len, features] or None
            - metadata: Dict with 'freq' and other info
        
        Note:
            This uses the fidel-ts Data_Provider infrastructure for proper
            data loading with all preprocessing steps applied.
        """
        from utils.tools import dotdict
        from data_provider.data_factory import Data_Provider
        
        # Resolve dataset name to config path using unified resolver
        config_path = self._resolve_dataset_config(dataset)
        
        # Load data config
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Extract frequency from data config
        freq = data_config.get('sampling_rate', 'h')
        
        # Build minimal args structure for Data_Provider
        # Uses instance attributes from experiment config for consistency
        # Note: dotdict returns None for missing keys (not AttributeError),
        # so we must explicitly set all required fields
        args = dotdict({
            'data_config': dotdict(data_config),
            'model_config': dotdict({
                'task': 'TimeCMA',  # Use TimeCMA task for simple loading
                'stride': 1,
                'hetero_align_stride': False,
                'custom_input': None,  # Use default task-based input (None, not False)
                'freq': freq,
            }),
            'model': 'TimeCMA',  # Required for model name checks in data loader
            'batch_size': 32,
            'input_len': self.input_len,    # From experiment config
            'output_len': self.output_len,  # From experiment config
            'scale': self.scale,            # From experiment config
            'noise': None,
            'num_workers': 0,
            'prefetch_factor': None,
            'disable_buffer': True,
            'preload_hetero': False,  # Don't preload hetero data for embedding generation
            'gpu': 0,
            'use_gpu': False,  # CPU is fine for data loading
        })
        
        try:
            # Create Data_Provider
            data_provider = Data_Provider(args, buffer=False)
            
            # Get the appropriate split
            if split == 'train':
                datasets = data_provider.get_train(return_type='set')
            elif split == 'val':
                datasets = data_provider.get_val(return_type='set')
            elif split == 'test':
                datasets = data_provider.get_test(return_type='set')
            else:
                raise ValueError(f"Unknown split: {split}")
            
            # Collect all samples from all entities
            all_values = []
            all_timestamps = []
            
            for entity_id, entity_dataset in datasets.items():
                for i in range(len(entity_dataset)):
                    sample = entity_dataset[i]
                    # Sample format: (sample_id, seq_x, seq_y, x_time, y_time, 
                    #                 x_hetero, y_hetero, hetero_x_time, hetero_y_time,
                    #                 hetero_general, hetero_channel)
                    seq_x = sample[1]  # [seq_len, channels]
                    x_time = sample[3]  # [seq_len, time_features]
                    
                    all_values.append(seq_x)
                    all_timestamps.append(x_time)
            
            if not all_values:
                raise ValueError(
                    f"No samples found for {dataset}/{split}. "
                    f"This may occur if the split is too small (needs at least seq_len + pred_len samples)."
                )
            
            values = np.stack(all_values, axis=0)  # [N, seq_len, channels]
            timestamps = np.stack(all_timestamps, axis=0) if all_timestamps else None
            
            # Build metadata (freq already extracted above)
            metadata = {'freq': freq, 'dataset': dataset, 'split': split}
            
            return values, timestamps, metadata
            
        except Exception as e:
            import traceback
            raise RuntimeError(f"Failed to load dataset {dataset}/{split}: {e}") from e
    
    # =========================================================================
    # MEMORY-EFFICIENT CHUNKED PROCESSING
    # For large datasets that don't fit in memory
    # =========================================================================
    
    def _count_dataset_samples(
        self,
        dataset: str,
        split: str,
    ) -> Tuple[int, int, int]:
        """
        Count dataset samples without loading data into memory.
        
        This method iterates through the dataset to count samples,
        getting the shape information without storing all the data.
        
        Args:
            dataset: Dataset name
            split: Data split ('train', 'val', 'test')
        
        Returns:
            Tuple of (num_samples, num_channels, seq_len)
        """
        from utils.tools import dotdict
        from data_provider.data_factory import Data_Provider
        
        # Resolve dataset config
        config_path = self._resolve_dataset_config(dataset)
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        freq = data_config.get('sampling_rate', 'h')
        
        # Build args for Data_Provider
        args = dotdict({
            'data_config': dotdict(data_config),
            'model_config': dotdict({
                'task': 'TimeCMA',
                'stride': 1,
                'hetero_align_stride': False,
                'custom_input': None,
                'freq': freq,
            }),
            'model': 'TimeCMA',
            'batch_size': 32,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'scale': self.scale,
            'noise': None,
            'num_workers': 0,
            'prefetch_factor': None,
            'disable_buffer': True,
            'preload_hetero': False,
            'gpu': 0,
            'use_gpu': False,
        })
        
        # Create Data_Provider
        data_provider = Data_Provider(args, buffer=False)
        
        # Get the appropriate split
        if split == 'train':
            datasets = data_provider.get_train(return_type='set')
        elif split == 'val':
            datasets = data_provider.get_val(return_type='set')
        elif split == 'test':
            datasets = data_provider.get_test(return_type='set')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Count samples and get shape from first sample
        total_samples = 0
        num_channels = 0
        seq_len = 0
        
        for entity_id, entity_dataset in datasets.items():
            entity_len = len(entity_dataset)
            total_samples += entity_len
            
            # Get shape from first sample (if not already known)
            if num_channels == 0 and entity_len > 0:
                first_sample = entity_dataset[0]
                seq_x = first_sample[1]  # [seq_len, channels]
                seq_len, num_channels = seq_x.shape
        
        return total_samples, num_channels, seq_len
    
    def _iter_dataset_chunks(
        self,
        dataset: str,
        split: str,
        chunk_size: int = 1000,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]]:
        """
        Iterate over dataset in memory-efficient chunks.
        
        Instead of loading all samples at once, this generator yields
        chunks of samples, allowing processing of datasets that don't
        fit in memory.
        
        Args:
            dataset: Dataset name
            split: Data split
            chunk_size: Number of samples per chunk
        
        Yields:
            Tuple of (values_chunk, timestamps_chunk, metadata, chunk_start_idx):
                - values_chunk: [chunk_size, seq_len, channels]
                - timestamps_chunk: [chunk_size, seq_len, features]
                - metadata: Dict with 'freq', 'dataset', 'split'
                - chunk_start_idx: Starting index of this chunk in the full dataset
        """
        from utils.tools import dotdict
        from data_provider.data_factory import Data_Provider
        
        # Resolve dataset config
        config_path = self._resolve_dataset_config(dataset)
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        freq = data_config.get('sampling_rate', 'h')
        metadata = {'freq': freq, 'dataset': dataset, 'split': split}
        
        # Build args for Data_Provider
        args = dotdict({
            'data_config': dotdict(data_config),
            'model_config': dotdict({
                'task': 'TimeCMA',
                'stride': 1,
                'hetero_align_stride': False,
                'custom_input': None,
                'freq': freq,
            }),
            'model': 'TimeCMA',
            'batch_size': 32,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'scale': self.scale,
            'noise': None,
            'num_workers': 0,
            'prefetch_factor': None,
            'disable_buffer': True,
            'preload_hetero': False,
            'gpu': 0,
            'use_gpu': False,
        })
        
        # Create Data_Provider
        data_provider = Data_Provider(args, buffer=False)
        
        # Get the appropriate split
        if split == 'train':
            datasets = data_provider.get_train(return_type='set')
        elif split == 'val':
            datasets = data_provider.get_val(return_type='set')
        elif split == 'test':
            datasets = data_provider.get_test(return_type='set')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Iterate through entities and accumulate chunks
        chunk_values = []
        chunk_timestamps = []
        chunk_start_idx = 0
        current_idx = 0
        
        for entity_id, entity_dataset in datasets.items():
            for i in range(len(entity_dataset)):
                sample = entity_dataset[i]
                seq_x = sample[1]  # [seq_len, channels]
                x_time = sample[3]  # [seq_len, time_features]
                
                chunk_values.append(seq_x)
                chunk_timestamps.append(x_time)
                current_idx += 1
                
                # Yield when chunk is full
                if len(chunk_values) >= chunk_size:
                    yield (
                        np.stack(chunk_values, axis=0),
                        np.stack(chunk_timestamps, axis=0),
                        metadata,
                        chunk_start_idx,
                    )
                    chunk_start_idx = current_idx
                    chunk_values = []
                    chunk_timestamps = []
        
        # Yield remaining samples
        if chunk_values:
            yield (
                np.stack(chunk_values, axis=0),
                np.stack(chunk_timestamps, axis=0),
                metadata,
                chunk_start_idx,
            )
    
    def generate_ts_embeddings_chunked(
        self,
        dataset: str,
        split: str,
        batch_size: int = 32,
        chunk_size: int = 1000,
        force: bool = False,
        enable_gpu_monitor: bool = True,
    ) -> np.ndarray:
        """
        Memory-efficient embedding generation using chunked processing.
        
        This method processes large datasets without loading everything into
        memory at once. It:
        1. Counts total samples (lightweight pass)
        2. Creates a streaming HDF5 writer
        3. Processes data in chunks, writing embeddings incrementally
        4. Cleans up GPU memory between chunks
        5. Monitors GPU utilization in background (optional)
        
        Use this method for large datasets like fidel_NYC_traffic_speed.
        
        Args:
            dataset: Dataset name
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size for LLM inference
            chunk_size: Number of samples to load at once (memory vs speed tradeoff)
            force: Force regeneration even if cache exists
            enable_gpu_monitor: Enable GPU utilization monitoring (default: True)
        
        Returns:
            Embeddings array [N, embed_dim, C] loaded from disk
        
        Memory Usage:
            - Data: ~chunk_size * seq_len * channels * 4 bytes
            - Prompts: ~batch_size * ~500 bytes
            - Embeddings: Written to disk incrementally
        """
        if self.ts_prompt_builder is None:
            raise ValueError(
                "Cannot generate time series embeddings without a TSPromptBuilder. "
                "Either set prompt_template in __init__, or use embed_texts() for raw text."
            )
        
        # Get data directory and create cache
        data_dir = self._get_data_directory(dataset)
        cache = LLMEmbeddingCache(str(data_dir), dataset)
        metadata = self._build_metadata(dataset)
        
        # Check cache - return early if exists (WITHOUT loading full array into memory!)
        if not force and cache.cache_exists(metadata, split):
            cache_info = cache.get_cache_info(metadata, split)
            if cache_info is not None:
                num_samples, embed_dim, num_channels = cache_info
                if self.console is not None:
                    self.console.print(f"  [dim]Cache exists for {dataset}/{split}: {num_samples} embeddings[/dim]")
                # Return None to indicate "already cached" - caller shouldn't need the data
                # The actual embeddings will be loaded by LLMEmbeddingProvider during training
                return None
        
        # Count samples first (lightweight)
        if self.console is not None:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"Counting samples in {dataset}/{split}...", total=None)
                total_samples, num_channels, seq_len = self._count_dataset_samples(dataset, split)
                progress.update(task, completed=True)
            
            self.console.print(f"  [dim]Found {total_samples} samples ({num_channels} channels)[/dim]")
        else:
            print(f"[ LLM Embedder ] Counting samples for {dataset}/{split}")
            total_samples, num_channels, seq_len = self._count_dataset_samples(dataset, split)
            print(f"[ LLM Embedder ] Found {total_samples} samples ({num_channels} channels)")
        
        if total_samples == 0:
            raise ValueError(f"No samples found for {dataset}/{split}")
        
        # Ensure model loaded
        self._ensure_model_loaded_with_progress()
        
        # Calculate total prompts for progress
        total_prompts = total_samples * num_channels
        num_prompt_batches = (total_prompts + batch_size - 1) // batch_size
        num_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        # Create cache directory
        cache_dir = cache.get_cache_dir(metadata)
        
        # Initialize GPU monitor if requested and GPU is available
        gpu_monitor = None
        if enable_gpu_monitor and 'cuda' in self.device and torch.cuda.is_available():
            try:
                from utils.gpu_monitor import GpuMonitor
                
                # Extract GPU index from device string (e.g., 'cuda:0' -> 0)
                gpu_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                
                # Create monitor (don't write CSV, just for live metrics)
                gpu_csv_path = str(cache_dir / f"gpu_telemetry_{split}.csv")
                gpu_monitor = GpuMonitor(device_index=gpu_idx, out_csv=gpu_csv_path, interval_s=1.0)
                gpu_monitor.start()
                
                if self.console is not None:
                    self.console.print(f"  [dim]GPU monitoring enabled (device: cuda:{gpu_idx})[/dim]")
            except Exception as e:
                if self.console is not None:
                    self.console.print(f"  [dim yellow]GPU monitoring unavailable: {e}[/dim yellow]")
                gpu_monitor = None
        
        # Start timing
        start_time = time.time()
        last_gpu_log_time = start_time
        gpu_log_interval = 10.0  # Log GPU stats every 10 seconds
        
        try:
            # Create streaming writer
            with StreamingEmbeddingWriter(
                cache_dir=cache_dir,
                split=split,
                num_samples=total_samples,
                embed_dim=self._embed_dim,
                num_channels=num_channels,
            ) as writer:
                
                # Process chunks with progress bar
                if self.console is not None:
                    from rich.progress import (
                        Progress, BarColumn, TextColumn, TimeElapsedColumn,
                        TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
                    )
                    
                    progress_columns = (
                        SpinnerColumn(),
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(bar_width=40),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn(),
                        TextColumn("•"),
                        TimeRemainingColumn(),
                    )
                    
                    with Progress(*progress_columns, console=self.console, transient=False) as progress:
                        task = progress.add_task(f"Embedding {split}", total=total_samples)
                        
                        for values_chunk, ts_chunk, data_meta, chunk_start in self._iter_dataset_chunks(
                            dataset, split, chunk_size
                        ):
                            # Process this chunk
                            chunk_embeddings = self._process_chunk_streaming(
                                values_chunk, ts_chunk, data_meta, batch_size
                            )
                            
                            # Write to disk
                            writer.write_batch(chunk_embeddings)
                            
                            # Update progress
                            progress.update(task, completed=writer.samples_written)
                            
                            # Log GPU metrics periodically
                            current_time = time.time()
                            if gpu_monitor and (current_time - last_gpu_log_time) >= gpu_log_interval:
                                gpu_metrics = gpu_monitor.get_latest_metrics()
                                if gpu_metrics:
                                    util = gpu_metrics.get('util_gpu_pct', 0)
                                    mem_used = gpu_metrics.get('mem_used_mib', 0)
                                    mem_total = gpu_metrics.get('mem_total_mib', 1)
                                    mem_pct = (mem_used / mem_total * 100) if mem_total else 0
                                    self.console.print(
                                        f"  [dim cyan]GPU: {util}% util | "
                                        f"{mem_used:,} MiB / {mem_total:,} MiB ({mem_pct:.1f}%)[/dim cyan]"
                                    )
                                last_gpu_log_time = current_time
                            
                            # Cleanup
                            del values_chunk, ts_chunk, chunk_embeddings
                            gc.collect()
                            torch.cuda.empty_cache()
                else:
                    # Fallback without console
                    for values_chunk, ts_chunk, data_meta, chunk_start in self._iter_dataset_chunks(
                        dataset, split, chunk_size
                    ):
                        chunk_embeddings = self._process_chunk_streaming(
                            values_chunk, ts_chunk, data_meta, batch_size
                        )
                        writer.write_batch(chunk_embeddings)
                        del values_chunk, ts_chunk, chunk_embeddings
                        gc.collect()
                        torch.cuda.empty_cache()
        
        finally:
            # Stop GPU monitor and get summary
            if gpu_monitor:
                gpu_monitor.stop()
                summary = gpu_monitor.summary()
                if summary and self.console is not None:
                    avg_util = summary.get('avg_util_gpu_pct', 0)
                    max_mem = summary.get('max_mem_used_mib', 0)
                    self.console.print(
                        f"  [dim]GPU Summary: avg {avg_util:.1f}% util | "
                        f"peak mem {max_mem:,} MiB[/dim]"
                    )
        
        generation_time = time.time() - start_time
        
        # Print completion
        if self.console is not None:
            self.console.print(f"  [dim]Generated {total_samples} embeddings in {generation_time:.1f}s[/dim]")
        
        # Update and save metadata
        metadata.generation_time_seconds = generation_time
        metadata.num_samples = total_samples
        metadata.num_channels = num_channels
        metadata.seq_len = seq_len
        metadata.device = self.device
        metadata.split = split
        metadata.save(cache_dir / "metadata.json")
        
        # Load and return from disk
        return cache.load_embeddings(metadata, split, quiet=True)
    
    def _process_chunk_streaming(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        metadata: Dict[str, Any],
        batch_size: int,
    ) -> np.ndarray:
        """
        Process a single chunk of data and return embeddings.
        
        Uses lazy prompt generation to minimize memory usage.
        
        Args:
            values: [chunk_size, seq_len, channels]
            timestamps: [chunk_size, seq_len, features]
            metadata: Dataset metadata
            batch_size: LLM batch size
        
        Returns:
            embeddings: [chunk_size, embed_dim, channels]
        """
        chunk_size, seq_len, num_channels = values.shape
        
        # Pre-allocate output array for this chunk
        chunk_embeddings = np.zeros(
            (chunk_size, self._embed_dim, num_channels),
            dtype=np.float32
        )
        
        # Process using lazy prompt generation
        for prompt_batch, sample_idxs, channel_idxs in self.ts_prompt_builder.iter_batch_prompts(
            values, timestamps, metadata, batch_size
        ):
            # Tokenize batch
            inputs = self._tokenizer(
                prompt_batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                if self.extraction_mode == 'last_token':
                    seq_lengths = inputs.attention_mask.sum(dim=1) - 1
                    batch_embeddings = hidden_states[
                        torch.arange(hidden_states.size(0), device=self.device),
                        seq_lengths
                    ]
                else:  # pooled
                    mask = inputs.attention_mask.unsqueeze(-1).float()
                    batch_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            
            # Place embeddings in correct positions
            batch_emb_np = batch_embeddings.cpu().numpy()
            for i, (sample_idx, channel_idx) in enumerate(zip(sample_idxs, channel_idxs)):
                chunk_embeddings[sample_idx, :, channel_idx] = batch_emb_np[i]
            
            # Cleanup GPU memory
            del inputs, outputs, hidden_states, batch_embeddings, batch_emb_np
        
        return chunk_embeddings
    
    def verify_cache(self, dataset: str, splits: List[str] = None) -> Dict[str, Any]:
        """
        Verify that embeddings are cached for a dataset.
        
        Args:
            dataset: Dataset name
            splits: List of splits to check (default: all)
        
        Returns:
            Verification status dictionary
        """
        # Get actual data directory from config
        data_dir = self._get_data_directory(dataset)
        
        cache = LLMEmbeddingCache(str(data_dir), dataset)
        metadata = self._build_metadata(dataset)
        return cache.verify_all_splits(metadata, splits)
