import os
import pandas as pd
import pickle

def load_or_compute(file_path, compute_func, force_recompute=False, **kwargs):
    """
    Checks if a file exists. 
    - If it exists and force_recompute is False: loads and returns it.
    - If it doesn't exist (or force_recompute is True): runs compute_func, saves the result, and returns it.
    
    Args:
        file_path (str): Path to save/load the file (e.g., 'data/processed/pagerank.csv').
        compute_func (callable): The function to run if data is missing (e.g., get_pagerank).
        force_recompute (bool): If True, ignores existing file and re-runs computation.
        **kwargs: Arguments to pass to compute_func (e.g., G=G_loaded, alpha=0.85).
        
    Returns:
        The data (DataFrame, dict, or whatever compute_func returns).
    """
    
    #if the file exists i load it 

    if os.path.exists(file_path) and not force_recompute:
        print("File found.")
        
        # Handle CSVs 
        if file_path.endswith('.csv'):
            # Assume first column is index (ASIN)
            return pd.read_csv(file_path, index_col=0)
            
        # Handle Pickles 
        elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        else:
            raise ValueError("Unsupported file extension. Use .csv or .pickle")

    # 2. If we are here, we need to COMPUTE
    print("Computing")
    result = compute_func(**kwargs)
    
    # 3. Save the result
    print(f"Saving to {file_path}...")
    
    # If result is a dict (like your scores), convert to DataFrame first for CSVs
    if file_path.endswith('.csv'):
        if isinstance(result, dict):
            # Convert {ASIN: score} dict to DataFrame
            # infer column name from filename (e.g. pagerank_scores.csv -> PageRank)
            col_name = os.path.basename(file_path).replace('_scores.csv', '').replace('.csv', '')
            df = pd.DataFrame.from_dict(result, orient='index', columns=[col_name])
            df.index.name = 'ASIN'
            df.to_csv(file_path)
            return df
        elif isinstance(result, pd.DataFrame):
            result.to_csv(file_path)
            return result
            
    # For Pickles
    elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)
            
    return result