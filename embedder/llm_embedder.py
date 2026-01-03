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

import time
import yaml
from typing import Dict, Any, Optional, Callable, List

import numpy as np
import torch

from .llm_registry import LLMRegistry
from .llm_cache import LLMEmbeddingCache, LLMEmbeddingMetadata
from .prompt_builder import TSPromptBuilder


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
        
        # Initialize time series prompt builder (optional - only for TS input)
        # If prompt_template is None, this embedder is for raw text only
        if prompt_template is not None:
            self.ts_prompt_builder = TSPromptBuilder(
                template_name=prompt_template,
                template_config=prompt_config
            )
        else:
            self.ts_prompt_builder = None
        
        # Default batch size (can be overridden by from_config or at call time)
        self.default_batch_size = 16
        
        # Lazy load model and tokenizer
        self._model = None
        self._tokenizer = None
        self._embed_dim = None
    
    @classmethod
    def from_config(
        cls,
        config_path: str,
        model_override: Optional[str] = None,
        device: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> 'LLMEmbedder':
        """
        Create embedder from YAML configuration file.
        
        Args:
            config_path: Path to configuration file
            model_override: Override model name from config
            device: Override device from config
            quantization: Override quantization from config
        
        Returns:
            Configured LLMEmbedder instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        llm_config = config.get('llm_embedding', config)
        
        embedder = cls(
            model_name=model_override or llm_config.get('model_name', 'gpt2'),
            device=device or llm_config.get('device', 'cuda:0'),
            cache_dir=llm_config.get('cache_dir', './LLM_cache/'),
            data_root=llm_config.get('data_root', './data/'),
            quantization=quantization or llm_config.get('quantization'),
            extraction_mode=llm_config.get('extraction_mode', 'last_token'),
            prompt_template=llm_config.get('prompt_template', 'timecma_v1'),
            prompt_config=llm_config.get('prompt_config', {}),
            max_length=llm_config.get('max_length', 512),
        )
        
        # Store batch_size from config for use by generate_ts_embeddings
        embedder.default_batch_size = llm_config.get('batch_size', 16)
        
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
        
        # Create cache manager
        cache = LLMEmbeddingCache(self.data_root, dataset)
        
        # Build metadata for cache key
        metadata = self._build_metadata(dataset)
        
        # Check cache
        if not force and cache.cache_exists(metadata, split):
            print(f"[ LLM Embedder ] Loading cached embeddings for {dataset}/{split}")
            return cache.load_embeddings(metadata, split)
        
        # Load data
        print(f"[ LLM Embedder ] Loading data for {dataset}/{split}")
        values, timestamps, data_metadata = self._load_dataset(dataset, split)
        
        N, L, C = values.shape
        print(f"[ LLM Embedder ] Generating embeddings: {N} samples, {C} channels")
        
        # Ensure model loaded
        self._ensure_model_loaded()
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self._generate_embeddings_from_timeseries(
            values=values,
            timestamps=timestamps,
            metadata=data_metadata,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
        generation_time = time.time() - start_time
        
        # Update metadata with generation info
        metadata.generation_time_seconds = generation_time
        metadata.num_samples = N
        metadata.num_channels = C
        metadata.seq_len = L
        metadata.device = self.device
        metadata.prompt_example = self.ts_prompt_builder.get_example_prompt(
            values, timestamps, data_metadata
        )
        
        # Save to cache
        cache.save_embeddings(embeddings, metadata, split)
        
        print(f"[ LLM Embedder ] Generated {N} embeddings in {generation_time:.1f}s")
        
        return embeddings
    
    def _generate_embeddings_from_timeseries(
        self,
        values: np.ndarray,
        timestamps: Optional[np.ndarray],
        metadata: Dict[str, Any],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
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
        
        Returns:
            embeddings: [N, embed_dim, C] LLM embeddings
        """
        N, L, C = values.shape
        
        # Step 1: Generate ALL prompts (flattened across samples and channels)
        all_prompts = self.ts_prompt_builder.build_flat_prompts(values, timestamps, metadata)
        total_prompts = len(all_prompts)  # N * C
        
        # Step 2: Embed all prompts
        all_embeddings = []
        
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
    
    def _load_dataset(
        self, 
        dataset: str, 
        split: str
    ) -> tuple:
        """
        Load dataset values and timestamps.
        
        This method handles both Time-MMD datasets (simple CSV structure)
        and Fidel-TS datasets (complex nested directories).
        
        For Time-MMD datasets:
            - Uses data_provider/data_factory for standard loading
            - Simple root_path/data_path structure
        
        For Fidel-TS datasets:
            - Has complex nested directory structures
            - Path resolution is handled by FidelTSPathResolver
            - Each subdataset (Bear_room, California_ISO, etc.) has unique paths
        
        Args:
            dataset: Dataset name
            split: Data split
        
        Returns:
            Tuple of (values, timestamps, metadata)
            - values: [N, seq_len, channels]
            - timestamps: [N, seq_len, features] or None
            - metadata: Dict with 'freq' and other info
        
        Note:
            This is a simplified implementation. Production use should
            integrate more tightly with the fidel-ts data loading infrastructure
            and handle Fidel-TS dataset structures properly.
        """
        # Import data loading utilities
        from data_provider.data_factory import data_provider
        
        # Build minimal config for data loading
        class DataConfig:
            def __init__(self, dataset_name, split, data_root):
                self.data = dataset_name
                self.root_path = f'{data_root}/{dataset_name}/'
                self.data_path = f'{dataset_name}.csv'
                self.features = 'M'
                self.target = 'OT'
                self.freq = 'h'
                self.seq_len = 96
                self.label_len = 48
                self.pred_len = 96
                self.scale = True
                self.timeenc = 0
                self.batch_size = 32
                self.num_workers = 0
                self.flag = split
        
        try:
            config = DataConfig(dataset, split, self.data_root)
            data_set, _ = data_provider(config, config.flag)
            
            # Extract all samples
            all_values = []
            all_timestamps = []
            
            for i in range(len(data_set)):
                seq_x, seq_y, seq_x_mark, seq_y_mark = data_set[i]
                all_values.append(seq_x)
                all_timestamps.append(seq_x_mark)
            
            values = np.stack(all_values, axis=0)
            timestamps = np.stack(all_timestamps, axis=0) if all_timestamps[0] is not None else None
            
            metadata = {'freq': config.freq}
            
            return values, timestamps, metadata
            
        except Exception as e:
            print(f"[ warning ] Could not load dataset {dataset}: {e}")
            print("[ warning ] Using dummy data for testing")
            
            # Return dummy data for testing
            N, L, C = 100, 96, 7
            values = np.random.randn(N, L, C).astype(np.float32)
            timestamps = None
            metadata = {'freq': 'h'}
            
            return values, timestamps, metadata
    
    def verify_cache(self, dataset: str, splits: List[str] = None) -> Dict[str, Any]:
        """
        Verify that embeddings are cached for a dataset.
        
        Args:
            dataset: Dataset name
            splits: List of splits to check (default: all)
        
        Returns:
            Verification status dictionary
        """
        cache = LLMEmbeddingCache(self.data_root, dataset)
        metadata = self._build_metadata(dataset)
        return cache.verify_all_splits(metadata, splits)
