# Fact Memorization in OLMo

## Objective
Study fact memorization in LLMs during training using the OLMo model family and the Dolma dataset.

## Goal
Identify when a piece of information first appeared in the training data and which checkpoint first saw it.

## Current Task
Build a tool/snippet/function to:
1. Search for text in the Dolma training corpus.
2. Identify the first checkpoint index that was trained on this text.

## Research & Next Steps
- Understand how to access and search the Dolma dataset.
- Understand the mapping between Dolma documents and OLMo training steps/checkpoints.
- Investigate if tools like `infini-gram` or `OLMoTrace` can be used or replicated.
