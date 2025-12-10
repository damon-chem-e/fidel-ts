# Text Noise Injection Implementation Plan

## 1. Overview
The goal is to inject irrelevant text (noise) into time-series datasets to study model robustness and the efficacy of different architectures in filtering noise. The system must support distinguishable (Type 1) and indistinguishable (Type 2) noise, with configurable probabilities and types.

## 2. Design Decisions & Improvements

### 2.1 Offline Generation (Recommended)
**Recommendation:** Generate injected datasets offline rather than on-the-fly during training.
*   **Reasoning:**
    *   **Performance:** LLM generation (even via API) is too slow for the training loop.
    *   **Reproducibility:** A fixed dataset ensures that changes in model performance are due to architecture, not random noise variations between runs.
    *   **Modularity:** Decouples "noise generation" from "data loading", requiring minimal changes to the complex `data_provider` logic.
    *   **Embedding:** Embeddings can be pre-computed (batch processed) which is orders of magnitude faster than single-sample embedding during loading.

### 2.2 Noise Types Refined
*   **Type 1 (Distinguishable):**
    *   **Goal:** Easy to spot.
    *   **Method:** "Out-of-Domain" text (e.g., excerpts from Wikipedia, news about non-weather topics) or "Fake Locations" (as proposed).
    *   **Improvement:** Use a corpus of "clearly not weather" text (e.g., generic news dataset) to avoid LLM costs for this simple case.
*   **Type 2 (Indistinguishable):**
    *   **Goal:** Plausible but irrelevant.
    *   **Method:** "Counterfactuals".
        *   **Shuffle:** Use weather reports from different years or different cities (free, fast, highly indistinguishable).
        *   **LLM Hallucination:** Generate plausible but false weather reports (expensive but controllable).

### 2.3 Metadata for Evaluation
Instead of just a "flag", we generate a rich metadata sidecar file (JSON/Parquet) mapping `(timestamp, entity_id)` to:
*   `has_injection`: bool
*   `injection_type`: "type1" | "type2" | "none"
*   `noise_source`: "shuffled_date_2019" | "llm_generated_id_123"
*   `original_text_kept`: bool

## 3. Implementation Architecture

The module `text_noise_injection` will contain:

### 3.1 `config.py`
Defines the `InjectionConfig` data class.

```python
@dataclass
class InjectionConfig:
    # Probabilities
    injection_prob: float = 0.5        # Probability a timepoint gets noise
    replace_prob: float = 0.0          # Probability noise replaces original text (vs appending)
    
    # Type distribution (conditional on injection)
    type1_ratio: float = 0.5           # 50% Type 1, 50% Type 2
    
    # Sources
    type1_source: str = "generic_news" # or "fake_location_generator"
    type2_source: str = "shuffle_time" # or "llm_counterfactual"
    
    # LLM Settings (if used)
    llm_model: str = "gpt-3.5-turbo"
    api_key_env_var: str = "OPENAI_API_KEY"
```

### 3.2 `generators.py`
Abstract strategy pattern for generating text.

*   `BaseGenerator`: Interface.
*   `CorpusGenerator`: Picks random sentences from a provided text list (fast Type 1).
*   `ShuffleGenerator`: Picks data from the dataset itself but different timestamps (fast Type 2).
*   `LLMGenerator`: Calls external API to generate text.

### 3.3 `injector.py`
The main processor class `DataInjector`.

*   **Inputs:** Path to original `json` data, `InjectionConfig`.
*   **Logic:**
    1.  Load original data.
    2.  Iterate `(timestamp, entity)`.
    3.  Roll dice based on `config.injection_prob`.
    4.  If inject:
        *   Roll dice for Type 1 vs Type 2.
        *   Call appropriate `Generator`.
        *   Append or Replace text.
        *   Record metadata.
    5.  Save new JSON file.
    6.  Save metadata JSON file.

### 3.4 `embedder.py` (Optional but recommended)
A script to pre-compute embeddings for the newly generated JSON file, mirroring the `NYC_traffic_speed/weather/merged_report_embedding` structure. This ensures the `Data_Provider` can load `.pkl` files directly.

## 4. Implementation Steps

1.  **Step 1: Setup:** Create `text_noise_injection` directory and `__init__.py`.
2.  **Step 2: Config:** Implement `InjectionConfig`.
3.  **Step 3: Generators:** Implement `ShuffleGenerator` (easiest/highest value) and `CorpusGenerator`. Leave `LLMGenerator` as a stub or basic implementation.
4.  **Step 4: Main Loop:** Implement `injector.py` to process the `NYC_traffic_speed` JSON structure.
5.  **Step 5: Integration:**
    *   Run the script to generate `.../weather/merged_general_report_with_injection_v1/`.
    *   (Optional) Run embedding script.
    *   Create a new YAML data config in `data_configs/NYC_traffic_speed/` pointing to the new directory.

## 5. Directory Structure

```
text_noise_injection/
├── __init__.py
├── config.py           # Configuration classes
├── generators.py       # Noise generation logic
├── injector.py         # Main processing loop
└── utils.py            # File I/O helpers
```

## 6. Usage Example (Conceptual)

```bash
# Generate the noisy dataset
python -m text_noise_injection.injector \
  --config configs/injection/high_noise.yaml \
  --input data/NYC_traffic_speed/weather/merged_general_report/merged_general_weather_report.json \
  --output data/NYC_traffic_speed/weather/merged_general_report_noise_v1

# Pre-compute embeddings (if not doing on-the-fly)
python -m text_noise_injection.embedder \
  --input data/NYC_traffic_speed/weather/merged_general_report_noise_v1 \
  --output data/NYC_traffic_speed/weather/merged_report_embedding_noise_v1
```

