Tensor Cache Entities: Detailed Trace and Fix Plan

Purpose
- Document the entity-mixing bug paths in Time-MMD/TTC and Fidel-TS.
- Provide a rigorous fix plan that preserves shapes and model expectations.

Scope
- Time-MMD / TTC datasets (TimeMMD_Dataset, TimeMMD_HeteroGetter).
- Fidel-TS datasets (Heterogeneous_Dataset, FidelTSEmbeddingLoader).
- Tensor cache generation and loading (data_provider/tensor_cache.py).
- Models: TGTSF, lynx_film, lynx_film_raw, lynx_film_enhanced.

Definitions
- entity_id: Identifier for a distinct time series (patient, site, etc.).
- timestamp: Integer in format YYYYMMDDHHMMSS (Time-MMD/TTC) or dataset-specific for Fidel-TS.
- general_info: Static, entity-level or dataset-level info text.
- channel_info: Static per-variable descriptions (one per channel).
- news/historical_events: Dynamic text embeddings aligned to prediction window (y_hetero) or input window (x_hetero).

-------------------------------------------------------------------------------
1) Detailed Trace: Time-MMD / TTC
-------------------------------------------------------------------------------

1.1 Dataset creation
- Data_Provider detects Time-MMD datasets and builds TimeMMD_Dataset:
  - data_provider/data_factory.py -> _create_time_mmd_dataset()
  - output_format depends on timemmd_text_output
  - For embedding models: output_format = "embedding"

1.2 CSV loading and text extraction
- TimeMMD_Dataset.__read_data__ loads CSV and identifies the text column:
  - Time-MMD format: Final_Search_* or Final_Output
  - TTC format: "text"
- Text column is excluded from time series data.
- Text is stored as pd.Series indexed by timestamp (int64 YYYYMMDDHHMMSS).

1.3 Embedding formation
- TimeMMD_HeteroGetter._load_or_create_embeddings:
  - Builds text_dict = {str(timestamp): text}
  - Calls TextEmbedder.embed_text_dict(text_dict, format_for_time_mmd=True)
- embed_text_dict returns embeddings keyed exactly by those timestamps.
- Output shape for dynamic text:
  - output_dynamic: (num_timesteps, 1, embed_dim)

1.4 Static embeddings (general_info, channel_info)
- _prepare_channel_and_general_info embeds:
  - general_info as (1, embed_dim)
  - channel_info as:
    - (num_channels, embed_dim) if channel_names are available
    - otherwise repeated or single embedding

1.5 Hetero data output to Universal_Dataset
- hetero_data_getter returns:
  - matched_times: List[str]
  - general_info_emb: (1, embed_dim)
  - channel_info_emb: (num_channels, embed_dim) or (1, embed_dim)
  - output_dynamic: (num_timesteps, 1, embed_dim)

1.6 Entity-mixing risk in Time-MMD / TTC
- New text embedding cache keys are timestamps only.
- Cache directory is based on model metadata, not entity or file.
- If two entities share timestamps, embeddings may collide and mix.

-------------------------------------------------------------------------------
2) Detailed Trace: Fidel-TS
-------------------------------------------------------------------------------

2.1 Dataset creation
- Data_Provider builds Heterogeneous_Dataset (Fidel-TS only):
  - data_provider/data_factory.py
  - output_format is always "embedding" for Fidel-TS

2.2 Dynamic embeddings
- FidelTSEmbeddingLoader loads text JSONs and flattens nested structures.
- Uses TextEmbedder.embed_text_dict(text_data, format_for_time_mmd=False).
- Dynamic embeddings are keyed by timestamp strings.
- For Fidel-TS, hetero_type = all_for_one for new cache:
  - Same dynamic text applies to all entities (intentional).

2.3 Static embeddings
- static_info.json provides general_info, channel_info, downtime_prompt.
- Embedded in batch and stored as:
  - general_info: (embed_dim,)
  - channel_info: dict[channel_id -> embedding]
  - downtime_prompt: (embed_dim,)
- init_hetero_data(id) selects channel_info[id] per entity.

2.4 Entity-mixing risk in Fidel-TS
- Dynamic text is global, so sharing across entities is expected.
- Time series values are still entity-specific and must not mix.
- Tensor cache currently deduplicates time series by timestamp only.

-------------------------------------------------------------------------------
3) Tensor Cache Generation and Loading Trace
-------------------------------------------------------------------------------

3.1 Shared table collection (tensor_cache.py)
- SharedTableCollector holds:
  - timestamp_to_idx: Dict[int, int]
  - timeseries: List[np.ndarray]
  - embeddings: List[np.ndarray]
  - entity_general, entity_channel per entity
- _process_sample_for_collection registers:
  - time series per timestamp
  - hetero embeddings per timestamp
  - entity-level static data per entity_id

3.2 Current dedup key
- _register_timestamp_data uses:
  - ts_key = int(timestamp)
  - if ts_key already present: skip
- No entity_id in the key.
- Result: timestamps collide across entities, mixing:
  - timeseries values
  - dynamic text embeddings
  - hetero_time features

3.3 TensorCacheDataset __getitem__
- Loads shared tables into RAM and index arrays per split.
- Reconstructs sample data using shared indices:
  - seq_x, seq_y from shared['timeseries'][x_idx/y_idx]
  - hetero_x, hetero_y from shared['embeddings'][strided indices]
  - hetero_general, hetero_channel from entity tables
- If timestamp indices are shared across entities, seq_x/seq_y and
  hetero_x/hetero_y are pulled from the wrong entity.

-------------------------------------------------------------------------------
4) Model Input Contracts (Shapes and Semantics)
-------------------------------------------------------------------------------

4.1 TGTSF
- Chooses text input by timestamp_semantics:
  - t_about: use y_hetero (news)
  - t_known: use x_hetero (historical_events)
- Expects:
  - text_input: [B, L, N, input_text_dim]
  - channel_description: [B, C, input_text_dim]
- Projects input_text_dim to text_dim if needed.

4.2 lynx_film, lynx_film_raw, lynx_film_enhanced
- Same timestamp_semantics logic as TGTSF.
- Expects:
  - text_input: [B, L, N, input_text_dim]
  - channel_description: [B, C, input_text_dim] (or [B, 1, C, D])
- Adds projection if input_text_dim != text_dim.

4.3 Dynamic text alignment
- Time-MMD/TTC: t_known (historical text, aligned to input window).
- Fidel-TS: t_about (forecast-style text about prediction window).

-------------------------------------------------------------------------------
5) Fix Plan (Rigorous and Entity-Safe)
-------------------------------------------------------------------------------

Goals
- Prevent cross-entity mixing in:
  - Time-MMD/TTC text embedding cache
  - Tensor cache shared tables
- Preserve existing model input shapes and semantics.
- Maintain performance benefits of deduplication.

Plan Overview
1) Make Time-MMD/TTC text embedding cache entity-safe.
2) Make tensor cache deduplication entity-aware for entity-specific data.
3) Preserve global (shared) embedding behavior for Fidel-TS where appropriate.
4) Validate shapes and semantics end-to-end.

-------------------------------------------------------------------------------
5.1 Fix: Time-MMD/TTC Embedding Cache Keying
-------------------------------------------------------------------------------

Problem
- Text embeddings are keyed by timestamp only.
- Cache directory is shared by model metadata only.

Fix Options (choose one; both are compatible)
Option A: Prefix timestamp keys with entity_id
- In TimeMMD_HeteroGetter._load_or_create_embeddings:
  - Build text_dict with keys "{entity_id}|{timestamp}"
  - Store embeddings keyed by composite key
- In _match_timestamps / _fetch_and_validate_embeddings:
  - Convert matched timestamp to composite key before lookup

Option B: Separate cache directory per entity
- Extend TextEmbedder cache_path to include entity_id or data_path filename:
  - cache_path = f"{subdir}/{entity_id}" or use data_path stem
- Prevents collision even if timestamp keys are the same.

Recommended Approach
- Use both:
  - Composite keys prevent collisions within the embedding dict.
  - Per-entity cache path prevents collision across cache directories.
- This yields safe behavior even if a user toggles cache_path or changes data_path.

Robustness
- No change to embedding shapes.
- Only the keys and cache directories change.
- No effect on models or data loaders, only on embedding retrieval.

-------------------------------------------------------------------------------
5.2 Fix: Tensor Cache Deduplication by (entity_id, timestamp)
-------------------------------------------------------------------------------

Problem
- Shared tables are keyed by timestamp only.
- This mixes time series and embeddings across entities.

Fix
- Change shared-table key from:
  - timestamp -> index
  to:
  - (entity_id, timestamp) -> index

Implementation Details
- SharedTableCollector:
  - Replace timestamp_to_idx: Dict[int, int]
  - With entity_timestamp_to_idx: Dict[str, int]
  - Key format: f"{entity_id}|{timestamp}"
- _register_timestamp_data:
  - Accept entity_id and use composite key
  - Store timestamps as the raw timestamp for readability
  - Keep a parallel array of entity_id for each timestamp if needed for debug
- _process_sample_for_collection:
  - Pass entity_id into _register_timestamp_data
- _write_sample_indices:
  - Convert x_time/y_time to indices using composite keys
  - Key = f"{entity_id}|{timestamp}"

Data Layout and Shapes (unchanged)
- shared['timeseries']: (N_unique, n_features)
- shared['embeddings']: (N_unique, embed_dim) or (N_unique, 2, embed_dim)
- shared['timestamps']: (N_unique,)  (timestamps only, no change)
- shared['entity_general']: (N_entities, embed_dim)
- shared['entity_channel']: (N_entities, embed_dim or n_channels, embed_dim)

Effect
- No changes to per-sample shapes.
- Models receive identical shapes but correct entity-aligned data.

-------------------------------------------------------------------------------
5.3 Handling Global Text (Fidel-TS all_for_one)
-------------------------------------------------------------------------------

Observation
- Fidel-TS dynamic text is intentionally shared across entities.

Decision
- Do NOT duplicate embeddings per entity for Fidel-TS dynamic text.
- Instead:
  - Use entity-aware deduplication only for entity-specific time series.
  - Allow embeddings to remain shared when hetero_type == all_for_one.

Implementation Note
- When hetero_type == all_for_one and embeddings are global:
  - For embeddings, use timestamp-only keys.
  - For time series, use entity|timestamp keys.
- This requires splitting dedup logic for time series vs embeddings.

-------------------------------------------------------------------------------
5.4 Compatibility and Migration
-------------------------------------------------------------------------------

Cache versioning
- Bump tensor cache metadata version or format tag.
- Invalidate old caches to prevent silent reuse.
- Include a config flag (e.g., tensor_cache_entity_keying: "entity_timestamp")
  to make the behavior explicit and hash it into config_hash.

Backwards compatibility
- Loader should reject mixed-format caches.
- Clear error message instructing to regenerate cache.

-------------------------------------------------------------------------------
5.5 Validation Plan (Correctness + Shape Guarantees)
-------------------------------------------------------------------------------

Unit tests
- Construct two entities with same timestamp but different:
  - time series values
  - text embeddings
- Generate cache and assert:
  - seq_x from each entity matches its original values
  - hetero_x from each entity matches its original embeddings

Shape checks (must not change)
- x_hetero / y_hetero: [L, N, D] where N is 1 or 2 (downtime)
- channel_description: [C, D] or [1, C, D]
- general_info: [1, D] or [D] depending on existing pipeline
- Ensure models still accept inputs with same shapes.

End-to-end smoke
- Run TGTSF and lynx_film variants on:
  - Time-MMD (t_known)
  - Fidel-TS (t_about)
- Verify no shape mismatch and model runs without data leakage.

-------------------------------------------------------------------------------
6) Expected Outcomes
-------------------------------------------------------------------------------

Correctness
- Entity-specific time series and embeddings no longer mix across entities.
- Time-MMD/TTC caches are safe even with shared timestamps.

Performance
- Deduplication remains effective, but keys are now entity-aware.
- Minor increase in unique entries (expected and correct).

Model compatibility
- No changes required in TGTSF or lynx_film variants.
- Inputs retain expected shapes and semantics.

