from pathlib import Path
import numpy as np
from tqdm import tqdm
import textwrap
from concurrent.futures import ThreadPoolExecutor
from build_massive_tokens import memmap_tokens, MACRO_STEPS, DATA_DIR


def process_shard(shard_idx: int, save_dir: Path) -> int:
    """
    Processes a single shard by reading tokens from the memmap and writing
    them to the infini-gram tokenized format using efficient numpy operations.
    """
    massive_tokens = memmap_tokens(mode="r")

    ds_path = save_dir / f"tokenized.{shard_idx}"
    od_path = save_dir / f"offset.{shard_idx}"

    data = massive_tokens[shard_idx]

    # Prepend document separator (65535) to each document
    # data shape is (N, 2049), we need shape (N, 2050)
    seps = np.full((data.shape[0], 1), 65535, dtype=np.uint16)
    concatenated = np.hstack((seps, data))

    # Generate offsets: each document takes 2050 tokens * 2 bytes = 4100 bytes
    num_docs = data.shape[0]
    offsets = np.arange(num_docs, dtype=np.uint64) * 4100

    with open(ds_path, "wb") as f_ds, open(od_path, "wb") as f_od:
        f_ds.write(concatenated.tobytes())
        f_od.write(offsets.tobytes())

    return shard_idx


def main(n_workers: int = 20) -> None:
    """
    Main function to orchestrate the parallel processing of shards.
    """
    save_dir: Path = DATA_DIR / "index_dir"
    save_dir.mkdir(exist_ok=True)

    shards: int = MACRO_STEPS

    print(f"Creating {shards} shards using {n_workers} workers.")

    def worker(idx: int) -> int:
        return process_shard(idx, save_dir)

    with ThreadPoolExecutor(n_workers) as executor:
        results = executor.map(worker, range(shards))
        for _ in tqdm(results, total=shards, desc="Processing Shards"):
            pass

    print("Done creating files for infini-gram.")
    print("Now you can run:")
    print(
        textwrap.dedent(
            f"""python -m infini_gram.indexing \
            --data_dir dummy \
            --save_dir {save_dir.absolute()} \
            --shards {shards} \
            --mem 200 \
            --token_dtype u16"""
        )
    )


if __name__ == "__main__":
    main()
