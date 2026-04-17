import fire
import numpy as np
from token_superset_bpe import BPETokenSupersetSearcher
from query_massive_tokens import InfiniGramSearcher

class MassiveTokenSearcher:
    def __init__(self, index_dir: str = "pile-data/index_dir", model_name: str = "EleutherAI/pythia-70m", max_workers: int = 10):
        """
        Initializes the searcher with a BPE searcher and an Infini-Gram searcher.
        
        Args:
            index_dir: The directory containing the Infini-Gram index.
            model_name: The name of the model for the tokenizer.
            max_workers: Number of threads for parallel querying.
        """
        print("Initializing BPETokenSupersetSearcher...")
        self.bpe_searcher = BPETokenSupersetSearcher(model_name=model_name)
        self.tokenizer = self.bpe_searcher.tokenizer
        
        self.infinigram_searcher = InfiniGramSearcher(index_dir=index_dir, max_workers=max_workers)

    def search(self, query: str, regex: str | None = None):
        """
        Finds all token representations for a query and returns all matches in the dataset.
        
        Args:
            query: The string to search for.
            regex: Optional regex pattern to filter token sequences.
            
        Returns:
            A list of dicts containing the matching sequence and the result location.
        """
        sequences = self.bpe_searcher.search(query, regex_pattern=regex)
        print(f"Found {len(sequences)} token sequences for query {query!r}")
        
        return self.infinigram_searcher.query_sequences(sequences)

    def display(self, query: str, context_len: int = 50, limit: int = 10, regex: str | None = None):
        """
        Searches for a query and displays the results in context.

        Args:
            query: The string to search for.
            context_len: Number of tokens to show before and after the match.
            limit: Maximum number of results to display.
            regex: Optional regex pattern to filter token sequences.
        """
        results = self.search(query, regex=regex)

        if not results:
            print("No results found.")
            return

        print(f"\nTotal matches found across all sequences: {len(results)}")
        print(f"Displaying top {min(limit, len(results))} results for query: {query!r}")
        print("=" * 80)

        for item in results[:limit]:
            seq = item["sequence"]
            res = item["result"]

            shard = res.shard
            sample_idx = res.sample_index
            offset = res.token_offset

            doc_tokens = self.infinigram_searcher.massive_tokens[shard, sample_idx]

            match_len = len(seq)
            actual_offset = offset

            # Check offset
            if offset + match_len <= len(doc_tokens) and np.array_equal(
                doc_tokens[offset : offset + match_len], seq
            ):
                actual_offset = offset
            # Check offset - 1
            elif (
                offset > 0
                and offset - 1 + match_len <= len(doc_tokens)
                and np.array_equal(doc_tokens[offset - 1 : offset - 1 + match_len], seq)
            ):
                actual_offset = offset - 1
            else:
                print(
                    f"Warning: Sequence {seq} not found at offset {offset} or {offset - 1} in shard {shard}, sample {sample_idx}"
                )
                actual_offset = offset

            start = max(0, actual_offset - context_len)
            end = min(len(doc_tokens), actual_offset + match_len + context_len)

            before_match = doc_tokens[start:actual_offset]
            match_tokens = doc_tokens[actual_offset : actual_offset + match_len]
            after_match = doc_tokens[actual_offset + match_len : end]

            decoded_before = self.tokenizer.decode(before_match)
            decoded_match = self.tokenizer.decode(match_tokens)
            decoded_after = self.tokenizer.decode(after_match)

            print(f"Shard: {shard}, Sample: {sample_idx}, Offset: {offset}")
            print(f"Context: ...{decoded_before}**{decoded_match}**{decoded_after}...")
            print("-" * 40)


if __name__ == "__main__":
    fire.Fire(MassiveTokenSearcher)
