import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
import textwrap


STEPS_PER_CHECKPOINT = 1000
TOKENS_PER_CHECKPOINT = 1024
SEQ_LENGTH = 2049
MACRO_STEPS = 95


DATA_DIR = Path("pile-data/")


def memmap_tokens(mode: str) -> np.memmap:
    return np.memmap(
        DATA_DIR / "massive_tokens",
        dtype=np.uint16,
        mode=mode,
        shape=(
            MACRO_STEPS,
            STEPS_PER_CHECKPOINT * TOKENS_PER_CHECKPOINT,
            SEQ_LENGTH,
        ),
    )


if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise RuntimeError(
            textwrap.dedent(f"""
            {DATA_DIR} not found. Download the dataset using:
            uvx hf download \
                pietrolesci/pile-deduped-pythia-preshuffled \
                --repo-type dataset \
                --include "data/*" \
                --local-dir ./{DATA_DIR}
        """)
        )

    MASSIVE_TOKENS = memmap_tokens(mode="w+")

    LOCK = Lock()

    def process_macro_step(i):
        path = DATA_DIR / f"data/train-{i * 1000:0>6}.parquet"
        df = pd.read_parquet(path)
        tokens = np.stack(df.token_ids.values)
        with LOCK:
            MASSIVE_TOKENS[i - 1] = tokens

    with ThreadPoolExecutor(6) as ex:
        list(tqdm(ex.map(process_macro_step, range(1, 96))))
