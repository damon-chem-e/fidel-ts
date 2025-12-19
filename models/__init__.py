import importlib
import yaml
from utils.tools import dotdict
from .LLM_socket import LLM_Socket
from .Time_R1_socket import Time_R1_socket

def model_init(model_name, configs, all_args, is_LLM=False, is_FM=False):
    # BEGIN DEBUG
    print(f"[DEBUG] models.__init__.model_init: Entry point")
    print(f"[DEBUG] models.__init__.model_init: model_name: {model_name}")
    print(f"[DEBUG] models.__init__.model_init: configs type: {type(configs)}")
    print(f"[DEBUG] models.__init__.model_init: hasattr(configs, 'enc_in'): {hasattr(configs, 'enc_in')}")
    if hasattr(configs, 'enc_in'):
        print(f"[DEBUG] models.__init__.model_init: configs.enc_in: {configs.enc_in}")
    # Check if it's a dict-like object
    if hasattr(configs, '__dict__'):
        print(f"[DEBUG] models.__init__.model_init: configs.__dict__ keys: {list(configs.__dict__.keys())}")
        if 'enc_in' in configs.__dict__:
            print(f"[DEBUG] models.__init__.model_init: configs.__dict__['enc_in']: {configs.__dict__['enc_in']}")
    elif isinstance(configs, dict):
        print(f"[DEBUG] models.__init__.model_init: configs dict keys: {list(configs.keys())}")
        if 'enc_in' in configs:
            print(f"[DEBUG] models.__init__.model_init: configs['enc_in']: {configs['enc_in']}")
    # END DEBUG

    if is_LLM:
        return Time_R1_socket(configs) if model_name.startwith("Time-R1") else LLM_Socket(configs)

    elif is_FM:
        configs['hist_len'] = all_args.input_len
        configs['pred_len'] = all_args.output_len
        configs['gpu'] = all_args.gpu if all_args.use_gpu else None

        module = importlib.import_module(f'models.{model_name}')
        model_class = getattr(module, 'Model')
        return model_class(configs)
    
    else:
        configs['seq_len'] = all_args.input_len
        configs['pred_len'] = all_args.output_len
        configs['gpu'] = all_args.gpu if all_args.use_gpu else None

        # BEGIN DEBUG
        print(f"[DEBUG] models.__init__.model_init: After setting seq_len, pred_len, gpu")
        print(f"[DEBUG] models.__init__.model_init: configs.enc_in: {getattr(configs, 'enc_in', 'ATTRIBUTE NOT FOUND')}")
        # END DEBUG

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

        # BEGIN DEBUG
        print(f"[DEBUG] models.__init__.model_init: Before creating model class")
        print(f"[DEBUG] models.__init__.model_init: Final configs.enc_in: {getattr(configs, 'enc_in', 'ATTRIBUTE NOT FOUND')}")
        # END DEBUG
        
        module = importlib.import_module(f'models.{model_name}')
        model_class = getattr(module, 'Model')
        
        # BEGIN DEBUG
        print(f"[DEBUG] models.__init__.model_init: About to call {model_name}.Model(configs)")
        # END DEBUG
        
        return model_class(configs)
