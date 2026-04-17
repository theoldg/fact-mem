import sys
import os
import numpy as np

# Add infini-gram-mini/engine to path
sys.path.append('/usr/local/google/home/lamort/Documents/fact-mem/infini-gram-mini/engine')
from src.engine import InfiniGramMiniEngine

def main():
    npy_path = "/usr/local/google/home/lamort/Documents/fact-mem/pile-tokenized/massive_tokens.npy"
    index_dir = "/usr/local/google/home/lamort/Documents/fact-mem/indexes/shard_0_1"
    
    print("Loading numpy file (memmap)...")
    data = np.memmap(npy_path, dtype=np.uint16, mode='r', shape=(95, 1024000, 2049))
    
    # Let's look for a sequence of tokens in shard 0 where all bytes are < 128
    # to avoid UTF-8 encoding issues with pybind11.
    
    shard_data = data[0]
    found_query = None
    query_tokens = None
    
    print("Searching for a safe query (all bytes < 128)...")
    for i in range(1000): # check first 1000 tokens
        tokens = shard_data.reshape(-1)[i:i+5]
        bytes_data = tokens.tobytes()
        if all(b < 128 for b in bytes_data):
            found_query = bytes_data.decode('latin1')
            query_tokens = tokens
            print(f"Found safe query at offset {i}!")
            print(f"Tokens: {query_tokens}")
            break
            
    if not found_query:
        print("Could not find a sequence of 5 tokens with all bytes < 128 in the first 1000 tokens.")
        print("Trying with a shorter sequence of 2 tokens...")
        for i in range(1000):
            tokens = shard_data.reshape(-1)[i:i+2]
            bytes_data = tokens.tobytes()
            if all(b < 128 for b in bytes_data):
                found_query = bytes_data.decode('latin1')
                query_tokens = tokens
                print(f"Found safe query at offset {i}!")
                print(f"Tokens: {query_tokens}")
                break
                
    if not found_query:
        print("Failed to find a safe query. Exiting.")
        return
        
    print(f"Query string (latin1): {found_query!r}")
    
    print("Initializing engine...")
    engine = InfiniGramMiniEngine(index_dirs=[index_dir], load_to_ram=False, get_metadata=True)
    
    print("Counting query...")
    count_res = engine.count(found_query)
    print("Count result:", count_res)
    
    print("Finding query...")
    find_res = engine.find(found_query)
    print("Find result:", find_res)
    
    if count_res.get('count', 0) > 0:
        print("SUCCESS: Index works and found the tokens!")
    else:
        print("FAILURE: Index did not find the tokens.")

if __name__ == '__main__':
    main()
