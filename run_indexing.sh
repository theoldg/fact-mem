uv run python -m infini_gram.indexing \
    --data_dir /usr/local/google/home/lamort/Documents/fact-mem/pile-tokenized \
    --save_dir /usr/local/google/home/lamort/Documents/fact-mem/pile-tokenized/infini_gram_index_massive \
    --token_dtype u16 \
    --mem 200 \
    --shards 95 \
    --ulimit 131072