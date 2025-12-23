import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class Model(nn.Module):
    """
    Time-MoE Pre-trained Model Wrapper

    This class encapsulates the Time-MoE model from Hugging Face to make it 
    compatible with the existing evaluation framework. It handles the specific
    normalization and de-normalization steps required by Time-MoE.
    """
    def __init__(self, configs):
        """
        Initializes the Time-MoE model wrapper.
        
        Args:
            configs (dotdict): Model configuration object, must contain 'pred_len'.
        """
        super(Model, self).__init__()
        
        # Get prediction length from config
        self.pred_len = configs.pred_len
        self.model_name = configs.model_name

        # --- Load the pre-trained Time-MoE model ---
        # Determine device and load model
        self.device = f"cuda:{int(configs.gpu)}" if configs.gpu is not None else "cpu"
        
        # Note: Time-MoE requires `trust_remote_code=True`
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
        )
        print(f"[ info ] {self.model_name} loaded successfully on device {next(self.model.parameters()).device}.")
        
        # Get hidden size from model config for representation extraction
        self.hidden_size = self.model.config.hidden_size
        
        # Set to evaluation mode for inference
        self.model.eval()

    def forward(self, x, return_representations=False, **kwargs):
        """
        Performs a forward pass (prediction) following Time-MoE's logic.

        Args:
            x (torch.Tensor): 
                Input context tensor. For univariate tasks, its shape is 
                [Batch, Input length] from the DataLoader.
            return_representations (bool): 
                If True, return internal representations instead of predictions.
                Returns aggregated hidden states [B, hidden_size] for use with ZhangHanBest.
            **kwargs: 
                Catches any extra arguments and ignores them.
        
        Returns:
            torch.Tensor: 
                If return_representations=False: 
                    Prediction tensor [Batch, pred_len]
                If return_representations=True:
                    Aggregated representation [Batch, hidden_size]
        """
        # --- Time-MoE Specific Pre-processing: Normalization ---
        # The model expects instance-wise normalization.
        
        # 1. Calculate mean and std for each sequence in the batch.
        #    keepdim=True ensures that mean/std have shape [Batch, 1] for broadcasting.
        #    We add a small epsilon to std to prevent division by zero.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8

        # 2. Normalize the input sequences
        normed_seqs = (x - mean) / std

        # --- Extract Representations (for ZhangHanBest integration) ---
        if return_representations:
            # Use forward pass with output_hidden_states to get internal representations
            # This extracts hidden states before the language model head
            with torch.no_grad():
                outputs = self.model.forward(
                    normed_seqs,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Get last hidden state: [B, seq_len, hidden_size]
            last_hidden_state = outputs.last_hidden_state
            
            # Aggregate over sequence length: mean pooling -> [B, hidden_size]
            # This gives us a global representation of the input sequence
            aggregated_repr = last_hidden_state.mean(dim=1)  # [B, hidden_size]
            
            return aggregated_repr

        # --- Model Generation (original prediction path) ---
        # 3. Generate future tokens. The model is causal, so it takes the context
        #    and generates `max_new_tokens` more.
        with torch.no_grad():
            # The `generate` method is part of the Hugging Face API
            output = self.model.generate(
                normed_seqs, 
                max_new_tokens=self.pred_len
            )
        
        # `output` shape is [Batch, Input length + Output length]
        
        # --- Time-MoE Specific Post-processing: De-normalization ---
        
        # 4. Extract only the newly generated (predicted) part of the sequence.
        normed_predictions = output[:, -self.pred_len:]

        # 5. Inverse normalize the predictions using the original mean and std.
        predictions = normed_predictions * std + mean
        
        # shape: [Batch, Pred_len]
        return predictions