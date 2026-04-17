from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from tqdm import tqdm
from build_massive_tokens import memmap_tokens, MACRO_STEPS

# Constants from build_massive_tokens.py
STEPS_PER_CHECKPOINT = 1000
TOKENS_PER_CHECKPOINT = 1024
SEQ_LENGTH = 2049

SAVE_DIR = Path("pile-tokenized/infini_gram_index_massive")


def process_shard(shard_idx):
    """Processes a single macro step as a shard."""
    # Re-open memmap in each process to be safe with multiprocessing
    tokens = memmap_tokens(mode="r")

    ds_path = SAVE_DIR / f"tokenized.{shard_idx}"
    od_path = SAVE_DIR / f"offset.{shard_idx}"

    # Slice for one macro step
    shard_tokens = tokens[shard_idx]
    num_docs = shard_tokens.shape[0]
    seq_len = shard_tokens.shape[1]

    expected_ds_size = num_docs * (seq_len + 1) * 2
    expected_od_size = num_docs * 8

    # Resumable check
    if ds_path.exists() and od_path.exists():
        if (
            ds_path.stat().st_size == expected_ds_size
            and od_path.stat().st_size == expected_od_size
        ):
            return f"Shard {shard_idx} skipped (already complete)"

    # Create new array with space for separator
    combined = np.empty((num_docs, seq_len + 1), dtype=np.uint16)
    combined[:, 0] = 0xFFFF  # Set separator
    combined[:, 1:] = shard_tokens  # Copy tokens

    # Flatten and write
    combined = combined.ravel()
    combined.tofile(ds_path)

    # Create offset file
    doc_len_bytes = (seq_len + 1) * 2
    offsets = np.arange(num_docs, dtype=np.uint64) * doc_len_bytes
    offsets.tofile(od_path)

    return f"Shard {shard_idx} completed"


def convert_massive_tokens_parallel():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {MACRO_STEPS} shards in parallel...")
    # We have 200 GB RAM. Each process takes ~8.4 GB peak.
    # 200 / 8.4 = 23. Let's use 20 workers to be safe.
    with ProcessPoolExecutor(max_workers=20) as executor:
        # Use tqdm to show progress
        results = list(
            tqdm(
                executor.map(process_shard, range(MACRO_STEPS)),
                total=MACRO_STEPS,
            )
        )

    for r in results:
        if "skipped" not in r:
            print(r)

    print("Done!")


if __name__ == "__main__":
    convert_massive_tokens_parallel()
