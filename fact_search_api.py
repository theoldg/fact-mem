import urllib.request
import json

def search_dolma(query_text):
    url = "https://api.infini-gram.io/"
    payload = {
        "index": "v4_dolma-v1_7_llama",
        "query_type": "find",
        "query": query_text
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = response.read().decode('utf-8')
            return json.loads(res_data)
    except Exception as e:
        print(f"Error: {e}")
        return None

def estimate_step(token_index, batch_size=4000000):
    """
    Estimates the training step for a given token index.
    Assumes a global batch size of ~4M tokens (e.g., OLMo 7B).
    """
    return token_index / batch_size

if __name__ == "__main__":
    query = "the capital of France is Paris"
    print(f"Searching for '{query}'...")
    result = search_dolma(query)
    
    if result and "segment_by_shard" in result:
        # Take the first occurrence as an example
        if result["segment_by_shard"]:
            first_occ = result["segment_by_shard"][0][0] # Start rank of first segment
            print(f"\nFirst occurrence token index: {first_occ}")
            step = estimate_step(first_occ)
            print(f"Estimated training step (assuming 4M batch size): {step:.2f}")
            
            # Checkpoints are usually every 1000 steps
            checkpoint_index = int(step / 1000) * 1000
            print(f"Likely first checkpoint index containing this data: {checkpoint_index}")
        else:
            print("No occurrences found.")
    else:
        print("No results or error.")
