from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM

MODEL_VARIANTS: list[str] = [
    "70m",
    "160m",
    "410m",
    "1.4b",
    "2.8b",
    "6.9b",
    "12b",
]

# Valid revisions are 0, 1000, 2000, ..., 143000 for all models.
VALID_REVISIONS: set[int] = {step for step in range(0, 144000, 1000)}


# For the deduped models, the first step which started seeing repeated data.
# https://github.com/EleutherAI/pythia/issues/144
FIRST_STEP_OF_SECOND_EPOCH = 95_000

BATCH_SIZE = 1024

# Sequences with seq_idx equal or greater than this should be considered already seen.
FIRST_REPEATED_SEQ_IDX = FIRST_STEP_OF_SECOND_EPOCH * BATCH_SIZE

CHECKPOINT_INTERVAL = 1000


@dataclass
class TextCompletionStats:
    total_surprisal: float
    avg_surprisal: float
    perplexity: float
    token_surprisals: list[float]
    tokens: list[str]


class NextTokenDistribution:
    def __init__(self, logits: torch.Tensor, tokenizer: AutoTokenizer):
        self.logits = logits  # Shape: (vocab_size,)
        self.tokenizer = tokenizer

    def top_n(self, n: int) -> list[dict]:
        probs = F.softmax(self.logits, dim=-1)
        top_logits, top_indices = torch.topk(self.logits, n)
        top_probs = probs[top_indices]
        
        results = []
        for logit, prob, idx in zip(top_logits, top_probs, top_indices):
            results.append({
                "token": self.tokenizer.decode([idx.item()]),
                "logit": logit.item(),
                "prob": prob.item(),
                "id": idx.item()
            })
        return results


class PythiaModel:
    model: GPTNeoXForCausalLM
    tokenizer: AutoTokenizer

    def __init__(self, model: GPTNeoXForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_variant_and_revision(
        cls,
        variant: str,
        revision: int,
        cache_dir: str | Path = '.cache/pythia',
    ) -> Self:
        cache_dir = Path(cache_dir)
        if variant not in MODEL_VARIANTS:
            raise ValueError(f'Invalid model variant: {variant}')
        if revision not in VALID_REVISIONS:
            raise ValueError(
                f'Invalid revision: {revision}. '
                '(Should be a number from 0 to 143000 divisible by 1000).'
            )
        model = GPTNeoXForCausalLM.from_pretrained(
            f'EleutherAI/pythia-{variant}-deduped',
            revision=f'step{revision}',
            cache_dir=cache_dir / variant / str(revision),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            f'EleutherAI/pythia-{variant}-deduped',
            revision=f'step{revision}',
            cache_dir=cache_dir / variant / str(revision),
        )
        # See huggingface.co/EleutherAI/pythia-6.9b/raw/main/tokenizer.json
        tokenizer.pad_token = '<|padding|>'
        return cls(model=model, tokenizer=tokenizer)

    def _get_logits(self, text: str, return_offsets_mapping: bool = False):
        encoding = self.tokenizer(text, return_offsets_mapping=return_offsets_mapping)
        input_ids = torch.tensor([encoding.input_ids]).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
        return outputs.logits, input_ids, encoding

    def next_token_distribution(self, prefix: str) -> NextTokenDistribution:
        """Returns the distribution over the next token for a given prefix."""
        logits, _, _ = self._get_logits(prefix)
        # We want the logits for the last token
        last_token_logits = logits[0, -1, :]
        return NextTokenDistribution(logits=last_token_logits, tokenizer=self.tokenizer)

    def text_completion_stats(self, prefix: str, suffix: str) -> TextCompletionStats:
        """
        Measures how well the model knows a fact by computing the surprisal
        and perplexity of the suffix given the prefix.
        """
        full_text = prefix + suffix

        logits, input_ids, encoding = self._get_logits(full_text, return_offsets_mapping=True)
        offset_mapping = encoding.offset_mapping

        prefix_len_chars = len(prefix)
        suffix_start_idx = None
        for i, (start, end) in enumerate(offset_mapping):
            if start < prefix_len_chars < end:
                raise ValueError(
                    f"Prefix and suffix are incompatible with token boundaries. "
                    f"Token {i} spans from {start} to {end}, straddling the boundary at {prefix_len_chars}."
                )
            if start >= prefix_len_chars:
                suffix_start_idx = i
                break

        if suffix_start_idx is None:
            raise ValueError("Could not find suffix tokens. Is suffix empty?")

        if suffix_start_idx == 0:
            raise ValueError("Suffix starts at index 0. Prefix cannot be empty.")

        target_ids = input_ids[0, suffix_start_idx:]
        relevant_logits = logits[0, suffix_start_idx - 1 : -1]
        assert len(target_ids) == len(relevant_logits)

        log_probs = F.log_softmax(relevant_logits, dim=-1)
        target_log_probs = log_probs[torch.arange(target_ids.shape[0]), target_ids]

        surprisal = -target_log_probs

        total_surprisal = surprisal.sum().item()
        avg_surprisal = surprisal.mean().item()
        perplexity = torch.exp(surprisal.mean()).item()

        return TextCompletionStats(
            total_surprisal=total_surprisal,
            avg_surprisal=avg_surprisal,
            perplexity=perplexity,
            token_surprisals=surprisal.tolist(),
            tokens=self.tokenizer.convert_ids_to_tokens(target_ids),
        )

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generates text completions for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
