import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import numpy as np

class TextEmbedder:
    def __init__(self, model_name, device='cpu', batch_size=200):
        self.model_name = model_name
        self.device = torch.device('cpu') if device == 'cpu' else torch.device(f'cuda:{device}' if isinstance(device, int) else device)
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embedding_single(self, text):
        encoded = self.tokenizer(text,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors='pt')

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # [CLS] token embedding
            text_embedding = outputs.last_hidden_state[:, 0, :].to('cpu')

        return text_embedding[0]

    def convert_df_text_to_embeddings(self, df):
        """
        Expects a DataFrame with 'time' column (optional) and other columns containing text.
        Merges non-time columns into a single string per row and embeds it.
        """
        # Preserve time if it exists
        if 'time' in df.columns:
            df_result = df[['time']].copy()
            df_to_embed = df.drop('time', axis=1)
        else:
            df_result = pd.DataFrame(index=df.index)
            df_to_embed = df

        # Merge text columns
        # Convert to string and join, handling potential non-string types gracefully
        df_merged = df_to_embed.astype(str).apply(' '.join, axis=1)

        batch_size = self.batch_size
        ls_embeddings = []
        
        # Use a list for iteration to avoid index issues
        text_list = df_merged.tolist()
        
        for i in tqdm(range(0, len(text_list), batch_size), desc="Processing embedding batches", unit="batch"):
            batch_texts = text_list[i:i+batch_size]

            encoded = self.tokenizer(batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt')

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].to('cpu')
                ls_embeddings.extend(batch_embeddings)

        df_result['embeddings'] = ls_embeddings

        return df_result

