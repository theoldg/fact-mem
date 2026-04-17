import fire
import numpy as np
from token_superset_bpe import BPETokenSupersetSearcher
from query_massive_tokens import FILE_INDEX_TO_SHARD, QueryResult
from build_massive_tokens import (
    memmap_tokens,
    STEPS_PER_CHECKPOINT,
    TOKENS_PER_CHECKPOINT,
)
from infini_gram.engine import InfiniGramEngine
import time
from typing import List, Dict, Any


class MassiveTokenSearcher:
    def __init__(
        self,
        index_dir: str = "pile-data/index_dir",
        model_name: str = "EleutherAI/pythia-70m",
    ):
        """
        Initializes the searcher with a BPE searcher and an Infini-Gram engine.

        Args:
            index_dir: The directory containing the Infini-Gram index.
            model_name: The name of the model for the tokenizer.
        """
        print("Initializing BPETokenSupersetSearcher...")
        self.bpe_searcher = BPETokenSupersetSearcher(model_name=model_name)
        self.tokenizer = self.bpe_searcher.tokenizer

        print(f"Initializing InfiniGramEngine with index_dir={index_dir}...")
        t0 = time.time()
        self.engine = InfiniGramEngine(
            index_dir=index_dir, eos_token_id=0, vocab_size=65535
        )
        print(f"Engine initialized in {time.time() - t0:.2f} seconds.")

        print("Loading massive_tokens memmap...")
        self.massive_tokens = memmap_tokens(mode="r")
        self.samples_per_shard = STEPS_PER_CHECKPOINT * TOKENS_PER_CHECKPOINT

    def search(self, query: str, regex: str | None = None) -> List[Dict[str, Any]]:
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

        all_results = []

        for i, seq in enumerate(sequences):
            decoded_seq = self.tokenizer.decode(seq)
            print(
                f"\n[{i + 1}/{len(sequences)}] Querying sequence: {seq} (decodes to: {decoded_seq!r})"
            )

            if isinstance(seq, np.ndarray):
                seq_list = seq.tolist()
            else:
                seq_list = seq

            print("  Calling engine.find()...")
            t0 = time.time()
            find_res = self.engine.find(input_ids=seq_list)
            print(f"  engine.find() took {time.time() - t0:.2f} seconds.")

            results = []
            if isinstance(find_res, dict) and "segment_by_shard" in find_res:
                for s, (start, end) in enumerate(find_res["segment_by_shard"]):
                    num_matches = end - start
                    print(f"    Shard {s}: found {num_matches} raw matches")

                    for rank in range(start, end):
                        if rank - start > 0 and (rank - start) % 1000 == 0:
                            print(
                                f"      Processed {rank - start}/{num_matches} matches in shard {s}..."
                            )

                        doc = self.engine.get_doc_by_rank(s=s, rank=rank)
                        if isinstance(doc, dict) and "doc_ix" in doc:
                            doc_ix = doc["doc_ix"]
                            needle_offset = doc["needle_offset"]

                            file_idx = doc_ix // self.samples_per_shard
                            sample_index = doc_ix % self.samples_per_shard

                            shard = FILE_INDEX_TO_SHARD[file_idx]

                            results.append(
                                QueryResult(
                                    shard=shard,
                                    sample_index=sample_index,
                                    token_offset=needle_offset,
                                )
                            )
            print(f"  Processed {len(results)} results for this sequence.")
            for r in results:
                all_results.append({"sequence": seq, "result": r})

        return all_results

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

            doc_tokens = self.massive_tokens[shard, sample_idx]

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
