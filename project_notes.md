# Fact Memorization in OLMo

## Objective
Study fact memorization in LLMs during training using the OLMo model family and the Dolma dataset.

## Goal
Identify when a piece of information first appeared in the training data and which checkpoint first saw it.
**Constraint:** Must use the entire dataset to avoid contamination and find the true first point of learning.

## Status
- **Tool Refactored:** `fact_search_api.py` supports anchors, regex filtering, and full retrieval.
- **Verified:** Tested with anchors `["Paris", "paris"]` and regex `"capital"`. Found 5 matches in Shard 0 at estimated step ~33541 (Checkpoint 33000). Snippets confirmed the presence of both terms.

## Uncertainties & Caveats
> [!WARNING]
> **Batch Size:** The step estimation assumes a global batch size of 4M tokens. This must be verified for the specific model version used.

## Current Task
Verify memorization in the model checkpoints.
1. Identify how to download/access specific OLMo checkpoints.
2. Run the model at those checkpoints to see if it knows the fact.

## Research & Next Steps
- Find the model ID or repository for OLMo checkpoints on Hugging Face.
- Write a script to load a specific checkpoint and query the model.
