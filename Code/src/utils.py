import os
import pandas as pd
import pickle
from pathlib import Path
import time
from datetime import datetime
import time

TIME_FILE = "../data/times_log.csv"
RESULTS_FILE = "../data/results/clustering_results.csv"


def log_times(func_name, duration_sec, params):
    """
    Append to CSV (or create it if not existant) where the entries are the timestamp, function name, function and function args
    The timestamp is probably not necessary but it's for now a way to not overlap logs made in different computer

    Returns:
        Times logged in a csv file (TIME_FILE)
    
    """

    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'function': func_name,
        'duration_sec': round(duration_sec, 4),
         # Save kwargs as string
        'parameters': str(params)
    }

    df_entry= pd.DataFrame([entry])

    if not os.path.exists(TIME_FILE):
        df_entry.to_csv(TIME_FILE, index=False)
    else:
        df_entry.to_csv(TIME_FILE, mode='a', header=False, index=False)



def load_or_compute(file_path, compute_func, force_recompute=False, **kwargs):
    """
    Checks if a file exists. 
    - If it exists and force_recompute is False: loads and returns it.
    - If it doesn't exist (or force_recompute is True): runs compute_func, saves the result, and returns it.
    
    Args:
        file_path (Path): Path to save/load the file (e.g., 'data/processed/pagerank.csv').
        compute_func (callable): The function to run if data is missing (e.g., get_pagerank).
        force_recompute (bool): If True, ignores existing file and re-runs computation.
        **kwargs: Arguments to pass to compute_func (e.g., G=G_loaded, alpha=0.85).
        
    Returns:
        The data (DataFrame, dict, or whatever compute_func returns).
    """

    if file_path.exists() and not force_recompute:
        print("File found.")
        ext = file_path.suffix.lower()

        # Handle CSVs
        if ext == '.csv':
            # Assume first column is index (ASIN)
            return pd.read_csv(file_path, index_col=0)
            
        # Handle Pickles 
        elif ext in {".pickle", ".pkl"}:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        else:
            raise ValueError("Unsupported file extension. Use .csv or .pickle")

    # If we are here, we need to COMPUTE
    print("Computing")
    start_time = time.time()
    
    result = compute_func(**kwargs)
    duration = time.time() - start_time
    
    #NOTE: for now on pagerak there is no method to save platform
    log_times(compute_func.__name__, duration, kwargs)
    
    # Save the result
    print(f"Saving to {file_path}...")
    ext = file_path.suffix.lower()

    # If result is a dict (like your scores), convert to DataFrame first for CSVs
    if ext == ".csv":
        if isinstance(result, dict):
            # Convert {ASIN: score} dict to DataFrame
            # infer column name from filename (e.g. pagerank_scores.csv -> PageRank)
            col_name = file_path.stem.replace("_scores", "")
            df = pd.DataFrame.from_dict(result, orient='index', columns=[col_name])
            df.index.name = 'ASIN'
            df.to_csv(file_path)
            return df
        elif isinstance(result, pd.DataFrame):
            result.to_csv(file_path)
            return result
            
    # For Pickles
    elif ext in {".pickle", ".pkl"}:
        with file_path.open("wb") as f:
            pickle.dump(result, f)
        return result

    else:
        raise ValueError("Unsupported file extension. Use .csv or .pickle/.pkl")

def log_clustering_results(method_name, nmi, ari):
    """
    Save clustering evaluation results to a CSV file.
    Args:
        method_name (str): Name of the clustering method.
        nmi (float): Normalized Mutual Information score.
        ari (float): Adjusted Rand Index score.
    """
    new_entry = pd.DataFrame([{
        'method': method_name, 
        'NMI': round(nmi, 4), 
        'ARI': round(ari, 4)
    }])

    if not os.path.exists(RESULTS_FILE):
        new_entry.to_csv(RESULTS_FILE, index=False)
    else:
        new_entry.to_csv(RESULTS_FILE, mode='a', header=False, index=False)

    print(f"Clustering results for {method_name} saved corrctly.")