# Text Noise Injection Implementation Plan (v2)

## 1. Overview
This module enables the injection of irrelevant text (noise) into time-series datasets to evaluate model robustness and attention mechanisms. The system generates "noisy" versions of datasets offline, ensuring reproducibility and efficiency.

## 2. Noise Strategies

We define two distinct types of noise, both focusing on reusing existing data (shuffling) to maintain textual distribution while destroying correlation or relevance.

### 2.1 Type 1: Semi-Distinguishable (Structure/Source Irrelevance)
*   **Goal:** The text itself looks like valid data (e.g., valid weather report), but comes from an irrelevant source or context. The model must learn to ignore specific "keys" or "locations".
*   **Mechanism (NYC Traffic Example):**
    *   Inject new, fake locations (e.g., "bronx", "staten_island", "fake_loc_1") alongside the valid ones ("brooklyn", "queens", "new-york").
    *   **Content:** Populate these new locations with weather reports randomly sampled from the original dataset (temporal shuffling).
    *   **Distinguishability:** The content is indistinguishable from real weather, but the *key/location* is distinguishable. The model learns "ignore 'fake_loc_1'".

### 2.2 Type 2: Indistinguishable (Temporal Irrelevance)
*   **Goal:** The text is from a valid source but irrelevant to the current time step.
*   **Mechanism:**
    *   Target existing valid locations ("brooklyn", etc.).
    *   **Content:** Replace the valid report at time $t$ with a report from time $t'$ (randomly sampled from the same location's history).
    *   **Distinguishability:** Extremely hard. The source is valid, the text is valid, but the information is wrong for the specific time.

## 3. Architecture & Workflow

### 3.1 Identification & Versioning
*   **Dataset ID:** We will not use simple versions like 'v1'. Each generated dataset gets a unique ID:
    *   `{timestamp}_{config_hash}` (e.g., `20251210_a1b2c3d4`)
*   **Directory Structure:**
    ```
    data/NYC_traffic_speed/
    └── {INJECTED_DATASET_ID}/
        ├── config.yaml             # The injection config used
        ├── metadata.json           # Detailed per-sample injection logs
        ├── raw_text/               # Structure mirroring original raw data
        │   └── weather/
        │       └── merged_general_report/
        │           └── merged_general_weather_report.json
        └── embeddings/             # Pre-computed embeddings
            └── weather/
                └── merged_report_embedding/
                    └── fast_general_formal_embeddings_2017.pkl
    ```

### 3.2 Components

#### `config.py`
Defines `InjectionConfig`:
*   `injection_prob`: Probability a timepoint has noise.
*   `replace_prob`: Probability noise replaces valid text (vs appending/adding new key).
*   `type1_params`: Configuration for fake locations (e.g., list of fake names, count).
*   `seed`: Master seed for reproducibility.

#### `injector.py`
The core logic engine.
1.  **Load:** Reads original JSON.
2.  **Plan:** Generates a "noise plan" (indices to shuffle) based on the seed.
3.  **Execute:**
    *   For **Type 1**: Creates new keys in the JSON dictionary for each timestamp, filling them with shuffled data.
    *   For **Type 2**: Overwrites existing keys with shuffled data.
4.  **Log:** Writes `metadata.json` mapping `(time, location)` -> `{is_injected, injection_type, source_time, original_kept}`.
5.  **Save:** Writes the new JSON structure to the unique ID directory.

#### `embedder.py`
Handles offline embedding generation to avoid runtime overhead.
*   **Refactoring:** We will extract the embedding logic from `data_provider/data_loader.py` into a shared utility `utils/text_embedding.py` (or similar).
*   **Operation:**
    *   Loads the *newly generated* noisy JSON.
    *   Uses the shared embedding utility to generate embeddings.
    *   Saves `.pkl` files matching the original file structure/naming convention.

## 4. Implementation Steps

1.  **Refactor Embedding Logic:**
    *   Create `utils/text_embedding.py`.
    *   Move `convert_df_text_to_embeddings` logic there from `Heterogeneous_Dataset`.
    *   Update `data_loader.py` to import and use this utility.

2.  **Implement Injection Module:**
    *   Create `text_noise_injection/config.py`.
    *   Create `text_noise_injection/injector.py` (implementing the shuffling logic).
    *   Create `text_noise_injection/utils.py` (hashing, ID generation).

3.  **Implement Offline Embedder:**
    *   Create `text_noise_injection/embedder.py` that utilizes `utils/text_embedding.py`.

4.  **Integration Test:**
    *   Generate a noisy dataset.
    *   Verify directory structure and metadata.
    *   Verify embeddings are loadable by the existing `Data_Provider`.

## 5. Metadata Schema (`metadata.json`)

```json
{
  "dataset_id": "20251210_a1b2c3d4",
  "base_dataset": "NYC_traffic_speed",
  "injections": {
    "20170101": {
      "brooklyn": {
        "is_injected": false,
        "type": "none"
      },
      "fake_loc_1": {
        "is_injected": true,
        "type": "type1",
        "source_time": "20180512",
        "source_location": "queens"
      }
    }
  }
}
```
