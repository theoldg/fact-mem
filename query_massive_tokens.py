from dataclasses import dataclass
import numpy as np
from infini_gram.engine import InfiniGramEngine
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

from build_massive_tokens import (
    STEPS_PER_CHECKPOINT,
    TOKENS_PER_CHECKPOINT,
    memmap_tokens,
)


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


@dataclass
class RawQueryResult:
    shard: int
    sample_index: int
    token_offset: int


def _get_file_index_to_shard() -> list[int]:
    """Creates mapping from file index to shard index due to lexicographical sorting by Infini-Gram."""
    filenames = [f"tokenized.{i}" for i in range(95)]
    sorted_filenames = sorted(filenames)
    return [int(f.split(".")[1]) for f in sorted_filenames]


FILE_INDEX_TO_SHARD = _get_file_index_to_shard()


def deduplicate_results(results: list[QueryResult]) -> list[QueryResult]:
    """Deduplicates QueryResult objects by (shard, sample_index, token_offset)."""
    seen = set()
    unique_results = []
    for r in results:
        key = (r.shard, r.sample_index, r.token_offset)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    return unique_results


class InfiniGramSearcher:
    def __init__(
        self,
        index_dir: str = "pile-data/index_dir",
        max_workers: int = 96,
        verbose: bool = False,
    ):
        """
        Initializes the Infini-Gram searcher.

        Args:
            index_dir: The directory containing the Infini-Gram index.
            max_workers: Number of threads for parallel querying.
        """
        self.verbose = verbose
        if self.verbose:
            print(f"Initializing InfiniGramEngine with index_dir={index_dir}...")
        self.engine = InfiniGramEngine(
            index_dir=index_dir, eos_token_id=0, vocab_size=65535
        )
        if self.verbose:
            print("Loading massive_tokens memmap...")
        self.massive_tokens = memmap_tokens(mode="r")
        self.samples_per_shard = STEPS_PER_CHECKPOINT * TOKENS_PER_CHECKPOINT
        self.max_workers = max_workers

    def query_sequence_raw(
        self, seq: list[int] | np.ndarray, max_disp_len: int | None = 0
    ) -> list[RawQueryResult]:
        """
        Queries a single sequence and returns raw results (shard, sample_index, token_offset)
        without looking up sequences or applying fallback by default.
        """
        find_res = self.engine.find(input_ids=list(seq))
        results = []

        if not isinstance(find_res, dict) or "segment_by_shard" not in find_res:
            return results

        shard_rank = [
            (shard, rank)
            for shard, (start, end) in enumerate(find_res["segment_by_shard"])
            for rank in range(start, end)
        ]

        # Use get_docs_by_ranks. max_disp_len=0 avoids looking up sequences.
        docs = self.engine.get_docs_by_ranks(shard_rank, max_disp_len=max_disp_len)

        if not isinstance(docs, list):
            return results

        for (shard_idx, rank), doc in zip(shard_rank, docs):
            if not isinstance(doc, dict) or "doc_ix" not in doc:
                continue

            doc_ix = doc["doc_ix"]
            needle_offset = doc.get("needle_offset", 0)
            file_idx = doc_ix // self.samples_per_shard
            sample_index = doc_ix % self.samples_per_shard
            shard = FILE_INDEX_TO_SHARD[file_idx]
            results.append(
                RawQueryResult(
                    shard=shard,
                    sample_index=sample_index,
                    token_offset=needle_offset,
                )
            )

        return results

    def post_process_results(
        self, raw_results: list[RawQueryResult], seq: list[int] | np.ndarray
    ) -> list[QueryResult]:
        """
        Post-processes raw results by verifying offsets and populating QueryResult.
        """
        results = []
        match_len = len(seq)
        seq_arr = np.array(seq, dtype=np.uint16)

        for r in tqdm(raw_results, "Post-processing results"):
            doc_tokens = self.massive_tokens[r.shard, r.sample_index]

            actual_offset = r.token_offset

            # Check offset
            if r.token_offset + match_len <= len(doc_tokens) and np.array_equal(
                doc_tokens[r.token_offset : r.token_offset + match_len],
                seq_arr,
            ):
                actual_offset = r.token_offset
            # Check offset - 1
            elif (
                r.token_offset > 0
                and r.token_offset - 1 + match_len <= len(doc_tokens)
                and np.array_equal(
                    doc_tokens[r.token_offset - 1 : r.token_offset - 1 + match_len],
                    seq_arr,
                )
            ):
                actual_offset = r.token_offset - 1
            else:
                # Fallback: scan the whole document
                for i in range(len(doc_tokens) - match_len + 1):
                    if np.array_equal(doc_tokens[i : i + match_len], seq_arr):
                        actual_offset = i
                        break
                else:
                    continue

            results.append(
                QueryResult(
                    shard=r.shard,
                    sample_index=r.sample_index,
                    token_offset=actual_offset,
                    sequence=list(seq),
                )
            )

        return deduplicate_results(results)

    def query_sequences(
        self,
        sequences: list[list[int] | np.ndarray],
        verbose: bool = False,
    ) -> list[QueryResult]:
        """
        Queries multiple sequences in parallel.
        """
        all_results = []

        def worker(seq: list[int] | np.ndarray):
            seq = list(seq)
            # Pass max_disp_len=None to get the default behavior (which usually finds correct offset)
            raw_results = self.query_sequence_raw(seq, max_disp_len=None)
            return seq, self.post_process_results(raw_results, seq)

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

        return deduplicate_results(all_results)
