# Fact Memorization in OLMo

## Objective
Study fact memorization in LLMs during training using the OLMo model family and the Dolma dataset.

## Goal
Identify when a piece of information first appeared in the training data and which checkpoint first saw it.
**Constraint:** Must use the entire dataset to avoid contamination and find the true first point of learning.

## Status
- **Tool Built:** `fact_search_api.py` searches the full Dolma dataset via the Infini-gram API and estimates the training step/checkpoint (assuming OLMo 7B parameters).
- **Verified:** Tested with "the capital of France is Paris" -> estimated step ~9756 (checkpoint 9000).

## Uncertainties & Caveats
> [!WARNING]
> **Batch Size:** The step estimation assumes a global batch size of 4M tokens (2048 instances * 2048 tokens). While this was the configuration for the original OLMo 7B pre-training, we must verify the exact batch size used for the specific model and data version we end up using. An incorrect batch size will shift the estimated training steps significantly.

## Current Task
Verify memorization in the model checkpoints.
1. Identify how to download/access specific OLMo checkpoints (e.g., step 9000 and surrounding steps).
2. Run the model at those checkpoints to see if it knows the fact.

## Research & Next Steps
- Find the model ID or repository for OLMo checkpoints on Hugging Face.
- Write a script to load a specific checkpoint and query the model.
- **Verify the exact training hyper-parameters (batch size, sequence length) for the chosen model version.**
