import importlib
import yaml
from utils.tools import dotdict
from .LLM_socket import LLM_Socket
from .Time_R1_socket import Time_R1_socket

def model_init(model_name, configs, all_args, is_LLM=False, is_FM=False):

    if is_LLM:
        return Time_R1_socket(configs) if model_name.startwith("Time-R1") else LLM_Socket(configs)

    elif is_FM:
        # Validate required parameters before setting
        if all_args.input_len is None:
            raise ValueError(f"input_len must be provided in training config for model {model_name}, got None")
        if all_args.output_len is None:
            raise ValueError(f"output_len must be provided in training config for model {model_name}, got None")
        
        configs['hist_len'] = all_args.input_len
        configs['pred_len'] = all_args.output_len
        configs['gpu'] = all_args.gpu if all_args.use_gpu else None

        module = importlib.import_module(f'models.{model_name}')
        model_class = getattr(module, 'Model')
        return model_class(configs)
    
    else:
        # Validate required parameters before setting
        if all_args.input_len is None:
            raise ValueError(f"input_len must be provided in training config for model {model_name}, got None")
        if all_args.output_len is None:
            raise ValueError(f"output_len must be provided in training config for model {model_name}, got None")
        
        configs['seq_len'] = all_args.input_len
        configs['pred_len'] = all_args.output_len
        configs['gpu'] = all_args.gpu if all_args.use_gpu else None
        
        # Automatically adjust enc_in for missing value indicator columns
        # enc_in in config represents the raw number of channels (without indicators)
        # We increment it by the number of indicator columns that will be added
        num_indicator_columns = getattr(all_args, 'num_indicator_columns', 0)
        if num_indicator_columns > 0 and 'enc_in' in configs:
            original_enc_in = configs['enc_in']
            configs['enc_in'] = original_enc_in + num_indicator_columns
            print(f"[ info ] Adjusted enc_in from {original_enc_in} to {configs['enc_in']} "
                  f"(added {num_indicator_columns} missing value indicator column(s))")

        data_configs = all_args.data_config
        try:
            configs['input_channel'] = data_configs.input_channel
        except: pass
        try:
            configs['base_T'] = data_configs.base_T
        except: pass
        try:
            configs['sampling_rate'] = data_configs.sampling_rate
        except: pass
        
        module = importlib.import_module(f'models.{model_name}')
        model_class = getattr(module, 'Model')
        
        return model_class(configs)
