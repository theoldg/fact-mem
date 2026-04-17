import json
import collections
from transformers import AutoTokenizer

class BPETokenSupersetSearcher:
    def __init__(self, model_name="EleutherAI/pythia-70m", merges_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        if merges_path is None:
            merges_path = "/usr/local/google/home/lamort/.gemini/jetski/brain/f5d171ba-e8de-4ae3-9f8c-0fd4c0c66002/scratch/merges.json"
        
        with open(merges_path, "r") as f:
            merges_list = json.load(f)
            
        self.merge_set = set(tuple(pair) for pair in merges_list)
        
        self.suffix_index = collections.defaultdict(list)
        self.prefix_index = collections.defaultdict(list)
        
        print("Building indexes...")
        for token, id in self.vocab.items():
            for i in range(1, len(token) + 1):
                self.suffix_index[token[-i:]].append(id)
                self.prefix_index[token[:i]].append(id)
        print("Indexes built.")

    def get_bpe_representation(self, text):
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text, add_special_tokens=False))
        return "".join(tokens)

    def search(self, target_string):
        S_bpe = self.get_bpe_representation(target_string)
        print(f"BPE representation of {target_string!r}: {S_bpe!r}")
        
        results = []
        L = len(S_bpe)
        
        def dfs(current_seq, remainder):
            if not remainder:
                results.append(current_seq)
                return

            for i in range(1, len(remainder) + 1):
                prefix = remainder[:i]
                if prefix in self.vocab:
                    token_id = self.vocab[prefix]
                    
                    t_prev = self.id_to_token[current_seq[-1]]
                    t_curr = self.id_to_token[token_id]
                    
                    if (t_prev, t_curr) in self.merge_set:
                        continue
                        
                    dfs(current_seq + [token_id], remainder[i:])

        # Case k = 1: S_bpe is a substring of a single token
        for token, id in self.vocab.items():
            if S_bpe in token:
                results.append([id])
                
        # Case k >= 2: DFS
        for i in range(1, L + 1):
            p1 = S_bpe[:i]
            candidates = self.suffix_index.get(p1, [])
            for v1 in candidates:
                remainder = S_bpe[i:]
                dfs([v1], remainder)
                
        # Filter for minimality and uniqueness
        unique_results = []
        seen = set()
        
        for seq in results:
            seq_tuple = tuple(seq)
            if seq_tuple in seen:
                continue
            seen.add(seq_tuple)
            
            full_str = "".join([self.id_to_token[id] for id in seq])
            if S_bpe not in full_str:
                continue
                
            minimal = True
            for i in range(len(seq)):
                sub_seq = seq[:i] + seq[i+1:]
                sub_str = "".join([self.id_to_token[id] for id in sub_seq])
                if S_bpe in sub_str:
                    minimal = False
                    break
            
            if minimal:
                unique_results.append(seq)
                
        return unique_results

if __name__ == "__main__":
    import sys
    searcher = BPETokenSupersetSearcher()
    
    target = "hello world"
    if len(sys.argv) > 1:
        target = sys.argv[1]
        
    res = searcher.search(target)
    print(f"Found {len(res)} minimal sequences.")
    for r in res[:10]:
        print([searcher.id_to_token[id] for id in r])
