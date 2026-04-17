uv run -m infini_gram.indexing \
    --data_dir . \
    --save_dir $(realpath pile-data/index_dir) \
    --shards 95 \
    --mem 200 \
    --token_dtype u16 \
    --ulimit 131072
