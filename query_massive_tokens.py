from dataclasses import dataclass
from typing import List
import numpy as np
from infini_gram.engine import InfiniGramEngine
from build_massive_tokens import STEPS_PER_CHECKPOINT, TOKENS_PER_CHECKPOINT


@dataclass(frozen=True)
class QueryResult:
    shard: int
    sample_index: int
    token_offset: int


def _get_file_index_to_shard() -> List[int]:
    """Creates mapping from file index to shard index due to lexicographical sorting by Infini-Gram."""
    filenames = [f"tokenized.{i}" for i in range(95)]
    sorted_filenames = sorted(filenames)
    return [int(f.split(".")[1]) for f in sorted_filenames]


FILE_INDEX_TO_SHARD = _get_file_index_to_shard()


def query_sequence(
    seq: np.ndarray, index_dir: str = "pile-data/index_dir"
) -> List[QueryResult]:
    """
    Queries the dataset for a certain token sequence using find() and get_doc_by_rank().

    Args:
        seq: A uint16 array or list of tokens.
        index_dir: The directory containing the Infini-Gram index.

    Returns:
        A list of QueryResult dataclasses.
    """
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()

    engine = InfiniGramEngine(index_dir=index_dir, eos_token_id=0, vocab_size=65535)

    find_res = engine.find(input_ids=seq)

    results = []
    if isinstance(find_res, dict) and "segment_by_shard" in find_res:
        samples_per_shard = STEPS_PER_CHECKPOINT * TOKENS_PER_CHECKPOINT
        for s, (start, end) in enumerate(find_res["segment_by_shard"]):
            for rank in range(start, end):
                doc = engine.get_doc_by_rank(s=s, rank=rank)
                if isinstance(doc, dict) and "doc_ix" in doc:
                    doc_ix = doc["doc_ix"]
                    needle_offset = doc["needle_offset"]

                    file_idx = doc_ix // samples_per_shard
                    sample_index = doc_ix % samples_per_shard

                    shard = FILE_INDEX_TO_SHARD[file_idx]

                    results.append(
                        QueryResult(
                            shard=shard,
                            sample_index=sample_index,
                            token_offset=needle_offset,
                        )
                    )
    return results
