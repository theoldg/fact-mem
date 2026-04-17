import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from build_massive_tokens import memmap_tokens, MACRO_STEPS, DATA_DIR


def process_shard(
    shard_idx, save_dir, massive_tokens, steps_per_shard, total_macro_steps, shards
):
    # print(f"Processing shard {shard_idx}...")
    start_step = shard_idx * steps_per_shard
    end_step = (
        (shard_idx + 1) * steps_per_shard
        if shard_idx < shards - 1
        else total_macro_steps
    )

    ds_path = save_dir / f"tokenized.{shard_idx}"
    od_path = save_dir / f"offset.{shard_idx}"

    with open(ds_path, "wb") as f_ds, open(od_path, "wb") as f_od:
        ods = 0
        for step in range(start_step, end_step):
            data = massive_tokens[step]
            for doc_idx in range(data.shape[0]):
                doc = data[doc_idx]
                content = b"\xff\xff" + doc.tobytes()
                f_ds.write(content)
                f_od.write(np.array([ods], dtype=np.uint64).tobytes())
                ods += len(content)
    return shard_idx


def main():
    save_dir = DATA_DIR / "index_dir"
    save_dir.mkdir(exist_ok=True)

    massive_tokens = memmap_tokens(mode="r")

    # The user wants 1 step per shard
    steps_per_shard = 1
    shards = MACRO_STEPS // steps_per_shard

    print(
        f"Creating {shards} shards with {steps_per_shard} step(s) each using 20 workers."
    )

    # Use 20 parallel workers
    with ThreadPoolExecutor(20) as executor:
        futures = []
        for shard_idx in range(shards):
            futures.append(
                executor.submit(
                    process_shard,
                    shard_idx,
                    save_dir,
                    massive_tokens,
                    steps_per_shard,
                    MACRO_STEPS,
                    shards,
                )
            )

        # Wait for all to complete with a progress bar
        for future in tqdm(futures, desc="Processing Shards"):
            future.result()

    print("Done creating files for infini-gram.")
    print("Now you can run:")
    print(
        f"python -m infini_gram.indexing --data_dir dummy --save_dir {save_dir.absolute()} --shards {shards} --mem 200 --token_dtype u16"
    )


if __name__ == "__main__":
    main()
