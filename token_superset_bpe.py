import json
import collections
import functools
from tqdm import tqdm
import re
import fire
import pickle
from pathlib import Path
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

        cache_dir = Path(__file__).resolve().parent / ".cache"
        prefix_cache_path = cache_dir / "prefix_index.pkl"
        suffix_cache_path = cache_dir / "suffix_index.pkl"

        if prefix_cache_path.exists() and suffix_cache_path.exists():
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
            cache_dir.mkdir(parents=True, exist_ok=True)
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

    def search(
        self, target_string: str, regex_pattern: str | None = None
    ) -> list[list[int]]:
        """Search for minimal token sequences covering the target string."""
        S_bpe = self.get_bpe_representation(target_string)
        target_ids = self.tokenizer.encode(target_string, add_special_tokens=False)
        print(f"BPE representation of {target_string!r}: {S_bpe!r}")

        results = []
        L = len(S_bpe)
        max_depth = len(target_ids) + 3

        pbar = tqdm(desc="Exploring states", unit="states")

        # Case k = 1: S_bpe is a substring of a single token
        for token, id in self.vocab.items():
            if S_bpe in token:
                results.append([id])

        @functools.lru_cache(maxsize=None)
        def dfs(
            remainder: str, max_len: int, prev_token: int | None = None
        ) -> list[tuple[int, ...]]:
            pbar.update(1)
            if not remainder:
                return [()]
            if max_len <= 0:
                return []

            local_results = []
            for i in range(1, len(remainder) + 1):
                prefix = remainder[:i]
                if prefix in self.vocab:
                    token_id = self.vocab[prefix]

                    if prev_token is not None:
                        t_prev = self.id_to_token[prev_token]
                        t_curr = self.id_to_token[token_id]
                        if (t_prev, t_curr) in self.merge_set:
                            continue

                    for sub_seq in dfs(remainder[i:], max_len - 1, token_id):
                        local_results.append((token_id,) + sub_seq)
            return local_results

        # Case k >= 2: DFS
        for i in range(1, L + 1):
            p1 = S_bpe[:i]
            candidates = self.suffix_index.get(p1, [])
            for v1 in candidates:
                remainder = S_bpe[i:]
                for sub_seq in dfs(remainder, max_depth - 1, v1):
                    results.append([v1] + list(sub_seq))

        pbar.close()
        print("BPE search complete. Post-processing results...")

        # Filter for minimality and uniqueness
        unique_results = []
        seen = set()

        for seq in tqdm(results, desc="Post-processing results"):
            seq_tuple = tuple(seq)
            if seq_tuple in seen:
                continue
            seen.add(seq_tuple)

            # 1. Round-Trip Validation (The Ground Truth Check)
            # Decode the sequence to a raw string, then re-encode it.
            decoded_text = self.tokenizer.decode(seq)
            re_encoded = self.tokenizer.encode(decoded_text, add_special_tokens=False)

            if re_encoded != seq:
                continue

            if regex_pattern and not re.search(regex_pattern, decoded_text):
                continue

            # 3. Minimality Check (Decoded String)
            minimal = True
            for i in range(len(seq)):
                sub_seq = seq[:i] + seq[i + 1 :]
                decoded_sub = self.tokenizer.decode(sub_seq)
                if target_string in decoded_sub:
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
