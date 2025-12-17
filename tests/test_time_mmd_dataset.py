"""
Test script for Time-MMD dataset loading.

This script tests the Time-MMD data provider to ensure it loads correctly
and provides data in the expected format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import Data_Provider
from utils.tools import dotdict
import yaml


def test_time_mmd_dataset(data_config_path, model_config_path=None):
    """
    Test Time-MMD dataset loading.
    
    Args:
        data_config_path: Path to data config YAML file
        model_config_path: Optional path to model config YAML file
    """
    print("=" * 60)
    print("Testing Time-MMD Dataset Loading")
    print("=" * 60)
    
    # Load data config
    print(f"\n1. Loading data config: {data_config_path}")
    with open(data_config_path, 'r') as f:
        data_config_dict = yaml.safe_load(f)
    data_config = dotdict(data_config_dict)
    
    # Check if it's a Time-MMD dataset
    if data_config.get('dataset_type') != 'time_mmd':
        print("⚠️  Warning: dataset_type is not 'time_mmd', will use standard dataset")
    else:
        print("✓ Time-MMD dataset detected")
    
    # Create minimal args object for testing
    class TestArgs:
        def __init__(self):
            self.data_config = data_config
            self.batch_size = 4
            self.input_len = 96
            self.output_len = 24
            self.scale = True
            self.num_workers = 0
            self.prefetch_factor = 2
            self.preload_hetero = False
            self.disable_buffer = False
            self.noise = 0.0
            
            # Model config (minimal)
            class ModelConfig:
                def __init__(self):
                    self.task = 'TSF'  # Start with time-series-only
                    self.custom_input = None
                    self.stride = 1
                    self.hetero_align_stride = False
                    self.name = 'DLinear'  # Default model name
            self.model_config = ModelConfig()
            
            # Data config hetero_info (for output_format)
            self.data_config.hetero_info = None
    
    args = TestArgs()
    
    # Load model config if provided
    if model_config_path:
        print(f"\n2. Loading model config: {model_config_path}")
        with open(model_config_path, 'r') as f:
            model_config_dict = yaml.safe_load(f)
        # Update model config in args
        for key, value in model_config_dict.items():
            setattr(args.model_config, key, value)
        print(f"   Task: {args.model_config.task}")
        if hasattr(args.model_config, 'custom_input'):
            print(f"   Custom input: {args.model_config.custom_input}")
    
    # Create data provider
    print("\n3. Creating Data_Provider...")
    try:
        data_provider = Data_Provider(args, buffer=False)
        print("✓ Data_Provider created successfully")
    except Exception as e:
        print(f"✗ Error creating Data_Provider: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading train dataset
    print("\n4. Loading train dataset...")
    try:
        train_dataset = data_provider.get_train(return_type='set')
        print(f"✓ Train dataset loaded: {len(train_dataset)} items")
        
        # Check first dataset item
        if len(train_dataset) > 0:
            first_key = list(train_dataset.keys())[0]
            first_dataset = train_dataset[first_key]
            print(f"   First dataset key: {first_key}")
            print(f"   Dataset length: {len(first_dataset)}")
            
            # Try to get one item
            if len(first_dataset) > 0:
                sample = first_dataset[0]
                print(f"\n5. Sample data structure:")
                print(f"   - sample_id: {type(sample[0])}, value: {sample[0]}")
                print(f"   - seq_x shape: {sample[1].shape}")
                print(f"   - seq_y shape: {sample[2].shape}")
                print(f"   - x_time shape: {sample[3].shape}")
                print(f"   - y_time shape: {sample[4].shape}")
                print(f"   - x_hetero type: {type(sample[5])}")
                print(f"   - y_hetero type: {type(sample[6])}")
                print(f"   - hetero_general: {sample[9]}")
                print(f"   - hetero_channel: {sample[10]}")
                
                # Check if text data is present
                if hasattr(first_dataset, 'hetero_data_getter') and first_dataset.hetero_data_getter is not None:
                    print(f"\n6. Text data getter present: ✓")
                    # Test getting text for a sample timestamp
                    test_timestamps = sample[3][:5]  # First 5 timestamps
                    try:
                        matched_times, general_info, channel_info, output_dynamic = first_dataset.hetero_data_getter(test_timestamps)
                        print(f"   Test text retrieval: ✓")
                        print(f"   - Matched {len(matched_times)} timestamps")
                        print(f"   - Output format: {type(output_dynamic)}")
                        if isinstance(output_dynamic, list) and len(output_dynamic) > 0:
                            print(f"   - First text sample: {str(output_dynamic[0])[:100]}...")
                    except Exception as e:
                        print(f"   ✗ Error retrieving text: {e}")
                else:
                    print(f"\n6. Text data getter: Not present (time-series-only mode)")
                
                # Print a couple training examples
                print(f"\n7. Training examples:")
                num_examples = min(2, len(first_dataset))
                for idx in range(num_examples):
                    example = first_dataset[idx]
                    print(f"\n   Example {idx + 1}:")
                    print(f"   - Sample ID: {example[0]}")
                    print(f"   - Input sequence (seq_x) shape: {example[1].shape}")
                    print(f"   - Output sequence (seq_y) shape: {example[2].shape}")
                    print(f"   - Input timestamps (x_time): {example[3][:3].tolist()}... (showing first 3)")
                    print(f"   - Output timestamps (y_time): {example[4][:3].tolist()}... (showing first 3)")
                    print(f"   - Input values (seq_x) min/max: {example[1].min():.2f} / {example[1].max():.2f}")
                    print(f"   - Output values (seq_y) min/max: {example[2].min():.2f} / {example[2].max():.2f}")
                    if hasattr(first_dataset, 'hetero_data_getter') and first_dataset.hetero_data_getter is not None:
                        # Get text for this example
                        try:
                            matched_times, gen_info, chan_info, text_data = first_dataset.hetero_data_getter(example[3])
                            if isinstance(text_data, list) and len(text_data) > 0:
                                print(f"   - Text data available: {len(text_data)} entries")
                                if len(text_data) > 0:
                                    print(f"   - First text: {str(text_data[0])[:80]}...")
                        except Exception:
                            pass
        else:
            print("⚠️  Warning: Train dataset is empty")
            
    except Exception as e:
        print(f"✗ Error loading train dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading validation dataset
    print("\n8. Loading validation dataset...")
    try:
        val_dataset = data_provider.get_val(return_type='set')
        print(f"✓ Validation dataset loaded: {len(val_dataset)} items")
    except Exception as e:
        print(f"✗ Error loading validation dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading test dataset
    print("\n9. Loading test dataset...")
    try:
        test_dataset = data_provider.get_test(return_type='set')
        print(f"✓ Test dataset loaded: {len(test_dataset)} items")
    except Exception as e:
        print(f"✗ Error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test creating dataloader
    print("\n10. Creating DataLoader...")
    try:
        train_loader = data_provider.get_train(return_type='loader')
        print(f"✓ DataLoader created successfully")
        
        # Try to get one batch
        print("\n11. Testing batch loading...")
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully")
        print(f"   Batch size: {len(batch[0])}")
        print(f"   seq_x shape: {batch[1].shape}")
        print(f"   seq_y shape: {batch[2].shape}")
    except Exception as e:
        print(f"✗ Error creating DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Time-MMD dataset is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Time-MMD dataset loading")
    parser.add_argument(
        'data_config',
        type=str,
        help='Path to data config YAML file (e.g., data_configs/time_mmd/Traffic/config.yaml)'
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default=None,
        help='Optional path to model config YAML file'
    )
    
    args = parser.parse_args()
    
    success = test_time_mmd_dataset(args.data_config, args.model_config)
    sys.exit(0 if success else 1)

