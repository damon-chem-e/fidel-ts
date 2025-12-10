from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class InjectionConfig:
    # General Settings
    seed: int = 42
    
    # Injection Probabilities
    injection_prob: float = 0.5        # Probability a timepoint gets any noise
    replace_prob: float = 0.0          # Probability noise replaces original text (vs appending)
    
    # Type 1 (Fake Locations) Settings
    type1_enabled: bool = True
    type1_ratio: float = 0.5           # Ratio of Type 1 noise (vs Type 2) when injection occurs
    fake_locations: List[str] = field(default_factory=lambda: ["fake_loc_1", "fake_loc_2", "bronx", "staten_island"])
    
    # Type 2 (Temporal Shuffling) Settings
    type2_enabled: bool = True
    
    # Source Settings
    source_dataset_path: str = "" # Path to the original JSON file to sample from
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

