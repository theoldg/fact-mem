import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

if not Path("pile-tokenized").exists():
    raise RuntimeError(
        "pile-tokenized not found. Download the dataset using\n"
        """uvx hf download pietrolesci/pile-deduped-pythia-preshuffled --repo-type dataset --include "data/*" --local-dir ./pile-tokenized"""
    )


steps = 1000
batch = 1024
seq = 2049
macro_steps = 95

MASSIVE_TOKENS = np.memmap(
    "pile-tokenized/massive_tokens.npy",
    dtype=np.uint16,
    mode="w+",
    shape=(macro_steps, steps * batch, seq),
)

LOCK = Lock()


def process_macro_step(i):
    path = f"pile-tokenized/data/train-{i * 1000:0>6}.parquet"
    df = pd.read_parquet(path)
    tokens = np.stack(df.token_ids.values)
    with LOCK:
        MASSIVE_TOKENS[i - 1] = tokens


with ThreadPoolExecutor(6) as ex:
    list(tqdm(ex.map(process_macro_step, range(1, 96))))
