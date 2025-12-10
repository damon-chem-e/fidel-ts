import random
import argparse
import copy
from tqdm import tqdm
from .config import InjectionConfig
from .utils import generate_dataset_id, ensure_dir, save_json, load_json
import os

class DataInjector:
    def __init__(self, config: InjectionConfig):
        self.config = config
        random.seed(self.config.seed)
        
    def load_data(self, path):
        print(f"Loading original data from {path}...")
        self.data = load_json(path)
        # Gather all text for sampling pool
        # Structure: time -> location -> text (or dict with text)
        # We need to handle nested structure if necessary, but based on examples it seems to be:
        # "YYYYMMDD": { "loc": "text" or "loc": {"daily": "text", ...} }
        
        self.all_timestamps = sorted(list(self.data.keys()))
        self.text_pool = []
        
        # Flatten data to build a pool of (text, source_loc, source_time)
        print("Building text pool...")
        for t in self.all_timestamps:
            for loc, content in self.data[t].items():
                self.text_pool.append({
                    "text": content,
                    "source_time": t,
                    "source_loc": loc
                })
                
    def get_random_sample(self):
        return random.choice(self.text_pool)

    def inject(self, output_root):
        if not hasattr(self, 'data'):
            raise ValueError("Data not loaded. Call load_data() first.")

        dataset_id = generate_dataset_id(self.config)
        output_dir = os.path.join(output_root, dataset_id)
        ensure_dir(output_dir)
        
        # Save config
        save_json(self.config.to_dict(), os.path.join(output_dir, "config.json"))
        
        new_data = copy.deepcopy(self.data)
        metadata = {
            "dataset_id": dataset_id,
            "base_dataset_path": self.config.source_dataset_path,
            "injections": {}
        }
        
        print(f"Injecting noise (ID: {dataset_id})...")
        
        for t in tqdm(self.all_timestamps, desc="Processing timestamps"):
            metadata["injections"][t] = {}
            
            # Determine if we inject at this timestamp
            if random.random() < self.config.injection_prob:
                
                # Determine Type 1 or Type 2
                # If Type 1 is enabled and (Type 2 disabled OR dice roll favors Type 1)
                is_type1 = self.config.type1_enabled and (
                    not self.config.type2_enabled or random.random() < self.config.type1_ratio
                )
                
                if is_type1:
                    # Type 1: Fake Location
                    fake_loc = random.choice(self.config.fake_locations)
                    sample = self.get_random_sample()
                    
                    # Inject
                    new_data[t][fake_loc] = sample["text"]
                    
                    metadata["injections"][t][fake_loc] = {
                        "is_injected": True,
                        "type": "type1",
                        "source_time": sample["source_time"],
                        "source_loc": sample["source_loc"]
                    }
                    
                else:
                    # Type 2: Indistinguishable (Temporal Shuffle) on existing location
                    # Pick an existing location to corrupt
                    existing_locs = list(new_data[t].keys())
                    if not existing_locs:
                        continue
                        
                    target_loc = random.choice(existing_locs)
                    sample = self.get_random_sample()
                    
                    # Replace
                    new_data[t][target_loc] = sample["text"]
                    
                    metadata["injections"][t][target_loc] = {
                        "is_injected": True,
                        "type": "type2",
                        "source_time": sample["source_time"],
                        "source_loc": sample["source_loc"],
                        "original_replaced": True
                    }
            
            # Log non-injected existing locations for completeness (optional, keeps metadata dense)
            # For now, we only log injections to keep file size reasonable, unless requested otherwise.
            
        # Save output
        raw_output_dir = os.path.join(output_dir, "raw_text")
        ensure_dir(raw_output_dir)
        
        # We maintain the filename of the input but in the new directory
        input_filename = os.path.basename(self.config.source_dataset_path)
        save_path = os.path.join(raw_output_dir, input_filename)
        save_json(new_data, save_path)
        save_json(metadata, os.path.join(output_dir, "metadata.json"))
        
        print(f"Generation complete. Data saved to {output_dir}")
        return output_dir, save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original JSON data")
    parser.add_argument("--output_root", required=True, help="Root directory for generated datasets")
    parser.add_argument("--prob", type=float, default=0.5, help="Injection probability")
    parser.add_argument("--type1_ratio", type=float, default=0.5, help="Ratio of Type 1 injections")
    args = parser.parse_args()
    
    config = InjectionConfig(
        injection_prob=args.prob,
        type1_ratio=args.type1_ratio,
        source_dataset_path=args.input
    )
    
    injector = DataInjector(config)
    injector.load_data(args.input)
    injector.inject(args.output_root)

