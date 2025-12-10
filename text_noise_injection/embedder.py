import argparse
import os
import pandas as pd
import joblib
import json
from tqdm import tqdm
from utils.text_embedding import TextEmbedder
from .utils import ensure_dir

def generate_embeddings(input_path, output_dir, model_name, batch_size=200, device='cuda'):
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    # Structure: {timestamp: {loc1: text, loc2: text}}
    print("Converting to DataFrame...")
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df['time'] = df.index
    
    # Initialize embedder
    print(f"Initializing embedder with model {model_name}...")
    embedder = TextEmbedder(model_name=model_name, device=device, batch_size=batch_size)
    
    # Process by year to manage memory and match file structure
    years = df.index.year.unique()
    
    ensure_dir(output_dir)
    
    for year in tqdm(years, desc="Processing years"):
        df_year = df[df.index.year == year].copy()
        
        # Embed
        # Note: convert_df_text_to_embeddings expects 'time' column to preserve it, 
        # but merges all other columns.
        print(f"Embedding year {year} ({len(df_year)} samples)...")
        df_embedded = embedder.convert_df_text_to_embeddings(df_year)
        
        # Convert to dictionary {timestamp_str: embedding}
        # We need to revert index to string format to match original keys if they were strings
        # But usually .pkl uses the timestamp object or string. 
        # The data_loader loads pkl and does: self.embeddings.update(pkl_data)
        # Then maps matched_dynamic (which are timestamps) to embeddings.
        
        # Let's see data_loader:
        # matched_dynamic = self.dynamic_data.loc[matched_times]['time'].values
        # output_dynamic_ = np.array([self.embeddings[time] for time in matched_dynamic], ...)
        
        # So the keys in the dictionary must match the values in 'time' column of dynamic_data.
        # In data_loader.py:
        # self.dynamic_data['time'] = self.dynamic_data.index
        # self.dynamic_data['time'] = self.dynamic_data['time'].dt.strftime('%Y%m%d%H%M%S')
        
        # So the keys should be strings '%Y%m%d%H%M%S'? 
        # Or int64?
        # In Universal_Dataset.__read_data__:
        # self.data[self.timestamp_col] = self.data[self.timestamp_col].dt.strftime('%Y%m%d%H%M%S')
        # self.data[self.timestamp_col] = self.data[self.timestamp_col].astype(np.int64)
        
        # However, Heterogeneous_Dataset seems to treat time as datetime index mainly.
        # But look at data_loader line 729:
        # matched_dynamic = self.dynamic_data.loc[matched_times]['time'].values
        # output_dynamic_ = np.array([self.embeddings[time] for time in matched_dynamic]
        
        # If 'time' column contains strings (strftime), then keys must be strings.
        # In Heterogeneous_Dataset.load_data (line 494):
        # self.dynamic_data['time'] = self.dynamic_data['time'].dt.strftime('%Y%m%d%H%M%S')
        
        # So yes, keys in pkl should be 'YYYYMMDDHHMMSS' strings.
        
        # Ensure 'time' in df_embedded is in the right format
        # It currently is datetime objects because we did df['time'] = df.index
        
        embeddings_dict = {}
        for idx, row in df_embedded.iterrows():
            # timestamp is idx (datetime)
            ts_str = idx.strftime('%Y%m%d%H%M%S')
            embeddings_dict[ts_str] = row['embeddings'].numpy() # Convert to numpy
            
        output_path = os.path.join(output_dir, f"fast_general_formal_embeddings_{year}.pkl")
        joblib.dump(embeddings_dict, output_path)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to injected JSON data")
    parser.add_argument("--output", required=True, help="Output directory for .pkl files")
    parser.add_argument("--model", default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", default="cpu") # default to cpu for safety, user can override
    
    args = parser.parse_args()
    
    generate_embeddings(args.input, args.output, args.model, args.batch_size, args.device)

