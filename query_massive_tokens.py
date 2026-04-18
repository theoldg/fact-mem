from dataclasses import dataclass
import numpy as np
from infini_gram.engine import InfiniGramEngine
from build_massive_tokens import (
    STEPS_PER_CHECKPOINT,
    TOKENS_PER_CHECKPOINT,
    memmap_tokens,
)
from concurrent.futures import ThreadPoolExecutor
import time


@dataclass(frozen=True)
class ContextResult:
    before: str
    match: str
    after: str


@dataclass
class QueryResult:
    shard: int
    sample_index: int
    token_offset: int
    sequence: list[int]
    context: ContextResult | None = None


def _get_file_index_to_shard() -> list[int]:
    """Creates mapping from file index to shard index due to lexicographical sorting by Infini-Gram."""
    filenames = [f"tokenized.{i}" for i in range(95)]
    sorted_filenames = sorted(filenames)
    return [int(f.split(".")[1]) for f in sorted_filenames]


FILE_INDEX_TO_SHARD = _get_file_index_to_shard()


class InfiniGramSearcher:
    def __init__(
        self,
        index_dir: str = "pile-data/index_dir",
        max_workers: int = 96,
    ):
        """
        Initializes the Infini-Gram searcher.

        Args:
            index_dir: The directory containing the Infini-Gram index.
            max_workers: Number of threads for parallel querying.
        """
        print(f"Initializing InfiniGramEngine with index_dir={index_dir}...")
        self.engine = InfiniGramEngine(
            index_dir=index_dir, eos_token_id=0, vocab_size=65535
        )
        print("Loading massive_tokens memmap...")
        self.massive_tokens = memmap_tokens(mode="r")
        self.samples_per_shard = STEPS_PER_CHECKPOINT * TOKENS_PER_CHECKPOINT
        self.max_workers = max_workers

    def query_sequence(self, seq: list[int] | np.ndarray) -> list[QueryResult]:
        """
        Queries a single sequence and applies fallback mechanism if offset is wrong.
        """
        if isinstance(seq, np.ndarray):
            seq_list = seq.tolist()
        else:
            seq_list = seq

        find_res = self.engine.find(input_ids=seq_list)
        results = []

        if not isinstance(find_res, dict) or "segment_by_shard" not in find_res:
            return results

        for s, (start, end) in enumerate(find_res["segment_by_shard"]):
            for rank in range(start, end):
                doc = self.engine.get_doc_by_rank(s=s, rank=rank)
                if not isinstance(doc, dict) or "doc_ix" not in doc:
                    continue

                doc_ix = doc["doc_ix"]
                needle_offset = doc["needle_offset"]

                file_idx = doc_ix // self.samples_per_shard
                sample_index = doc_ix % self.samples_per_shard

                shard = FILE_INDEX_TO_SHARD[file_idx]

                # Fallback mechanism
                actual_offset = needle_offset
                doc_tokens = self.massive_tokens[shard, sample_index]
                match_len = len(seq_list)
                seq_arr = np.array(seq_list, dtype=np.uint16)

                matched = False
                # Check offset
                if needle_offset + match_len <= len(doc_tokens) and np.array_equal(
                    doc_tokens[needle_offset : needle_offset + match_len],
                    seq_arr,
                ):
                    actual_offset = needle_offset
                    matched = True
                # Check offset - 1
                elif (
                    needle_offset > 0
                    and needle_offset - 1 + match_len <= len(doc_tokens)
                    and np.array_equal(
                        doc_tokens[needle_offset - 1 : needle_offset - 1 + match_len],
                        seq_arr,
                    )
                ):
                    actual_offset = needle_offset - 1
                    matched = True

                # Fallback: scan the whole document
                if not matched:
                    for i in range(len(doc_tokens) - match_len + 1):
                        if np.array_equal(doc_tokens[i : i + match_len], seq_arr):
                            actual_offset = i
                            matched = True
                            break

                results.append(
                    QueryResult(
                        shard=shard,
                        sample_index=sample_index,
                        token_offset=actual_offset,
                        sequence=seq_list,
                    )
                )
        return results

    def query_sequences(
        self,
        sequences: list[list[int] | np.ndarray],
        verbose: bool = True,
    ) -> list[QueryResult]:
        """
        Queries multiple sequences in parallel.
        """
        all_results = []

        def worker(seq):
            res = self.query_sequence(seq)
            return seq, res

        if verbose:
            print(
                f"Querying {len(sequences)} sequences in parallel with {self.max_workers} workers..."
            )
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results_iter = executor.map(worker, sequences)

            for seq, res in results_iter:
                if verbose:
                    print(f"  Sequence {seq} yielded {len(res)} results.")
                all_results.extend(res)

        if verbose:
            print(f"Parallel query completed in {time.time() - t0:.2f} seconds.")
        return all_results
