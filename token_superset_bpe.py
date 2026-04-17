import json
import collections
import re
import fire
import os
import pickle
from transformers import AutoTokenizer


class BPETokenSupersetSearcher:
    """Searcher for token sequences that form a superset of a given string."""

    def __init__(self, model_name: str = "EleutherAI/pythia-70m") -> None:
        """Initialize the searcher with a model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Extract merges directly from the tokenizer's JSON representation
        tokenizer_json = json.loads(self.tokenizer.backend_tokenizer.to_str())
        model_data = tokenizer_json.get("model", {})
        merges_list = model_data.get("merges", [])
        self.merge_set = set(tuple(pair) for pair in merges_list)

        cache_dir = "/usr/local/google/home/lamort/Documents/fact-mem/.cache"
        prefix_cache_path = os.path.join(cache_dir, "prefix_index.pkl")
        suffix_cache_path = os.path.join(cache_dir, "suffix_index.pkl")

        if os.path.exists(prefix_cache_path) and os.path.exists(suffix_cache_path):
            print("Loading indexes from cache...")
            with open(prefix_cache_path, "rb") as f:
                self.prefix_index = pickle.load(f)
            with open(suffix_cache_path, "rb") as f:
                self.suffix_index = pickle.load(f)
            print("Indexes loaded from cache.")
        else:
            self.suffix_index = collections.defaultdict(list)
            self.prefix_index = collections.defaultdict(list)

            print("Building indexes...")
            for token, id in self.vocab.items():
                for i in range(1, len(token) + 1):
                    self.suffix_index[token[-i:]].append(id)
                    self.prefix_index[token[:i]].append(id)
            print("Indexes built.")

            print("Caching indexes...")
            os.makedirs(cache_dir, exist_ok=True)
            with open(prefix_cache_path, "wb") as f:
                pickle.dump(self.prefix_index, f)
            with open(suffix_cache_path, "wb") as f:
                pickle.dump(self.suffix_index, f)
            print("Indexes cached.")

    def get_bpe_representation(self, text: str) -> str:
        """Get the BPE representation of a string by concatenating its tokens."""
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text, add_special_tokens=False)
        )
        return "".join(tokens)

    def search(self, target_string: str, regex_pattern: str | None = None) -> list[list[int]]:
        """Search for minimal token sequences covering the target string."""
        S_bpe = self.get_bpe_representation(target_string)
        print(f"BPE representation of {target_string!r}: {S_bpe!r}")

        results = []
        L = len(S_bpe)

        def dfs(current_seq: list[int], remainder: str) -> None:
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

            # 1. Round-Trip Validation (The Ground Truth Check)
            # Decode the sequence to a raw string, then re-encode it.
            decoded_text = self.tokenizer.decode(seq)
            re_encoded = self.tokenizer.encode(decoded_text, add_special_tokens=False)

            # If the tokenizer doesn't naturally produce this exact sequence
            # for these exact bytes, it's an impossible sequence.
            if re_encoded != seq:
                continue

            if regex_pattern and not re.search(regex_pattern, decoded_text):
                continue

            # 2. Existing Substring Check
            full_str = "".join([self.id_to_token[id] for id in seq])
            if S_bpe not in full_str:
                continue

            # 3. Existing Minimality Check
            minimal = True
            for i in range(len(seq)):
                sub_seq = seq[:i] + seq[i + 1 :]
                sub_str = "".join([self.id_to_token[id] for id in sub_seq])
                if S_bpe in sub_str:
                    minimal = False
                    break

            if minimal:
                unique_results.append(seq)

        return unique_results


def main(target: str, regex: str | None = None) -> None:
    """Main function to run the search from CLI."""
    searcher = BPETokenSupersetSearcher()
    res = searcher.search(target, regex_pattern=regex)
    print(f"Found {len(res)} minimal sequences.")
    for r in res[:10]:
        decoded = searcher.tokenizer.decode(r)
        token_strs = [searcher.id_to_token[id] for id in r]
        print(f"IDs: {r}")
        print(f"Tokens: {token_strs}")
        print(f"Decoded: {decoded!r}")
        print("-" * 20)


if __name__ == "__main__":
    fire.Fire(main)
