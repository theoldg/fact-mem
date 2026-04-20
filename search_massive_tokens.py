import fire
import numpy as np
import re
from tqdm import tqdm

from token_superset_bpe import BPETokenSupersetSearcher
from query_massive_tokens import InfiniGramSearcher, QueryResult, ContextResult


class MassiveTokenSearcher:
    def __init__(
        self,
        index_dir: str = "pile-data/index_dir",
        model_name: str = "EleutherAI/pythia-70m",
        max_workers: int = 96,
        verbose: bool = False,
    ):
        """
        Initializes the searcher with a BPE searcher and an Infini-Gram searcher.

        Args:
            index_dir: The directory containing the Infini-Gram index.
            model_name: The name of the model for the tokenizer.
            max_workers: Number of threads for parallel querying.
            verbose: Whether to print progress messages.
        """
        self.verbose = verbose
        if self.verbose:
            print("Initializing BPETokenSupersetSearcher...")
        self.bpe_searcher = BPETokenSupersetSearcher(
            model_name=model_name, verbose=self.verbose
        )
        self.tokenizer = self.bpe_searcher.tokenizer

        self.infinigram_searcher = InfiniGramSearcher(
            index_dir=index_dir, max_workers=max_workers, verbose=self.verbose
        )

    def _get_context(self, res: QueryResult, context_len: int) -> ContextResult:
        """Extracts and decodes context around the match in a QueryResult."""
        seq = res.sequence
        shard = res.shard
        sample_idx = res.sample_index
        offset = res.token_offset

        doc_tokens = self.infinigram_searcher.massive_tokens[shard, sample_idx]
        match_len = len(seq)

        # Fallback check
        actual_offset = offset
        if offset + match_len <= len(doc_tokens) and np.array_equal(
            doc_tokens[offset : offset + match_len], seq
        ):
            actual_offset = offset
        elif (
            offset > 0
            and offset - 1 + match_len <= len(doc_tokens)
            and np.array_equal(doc_tokens[offset - 1 : offset - 1 + match_len], seq)
        ):
            actual_offset = offset - 1
        else:
            print(
                f"Warning: Sequence {seq} not found at offset {offset} or {offset - 1} in shard {shard}, sample {sample_idx}"
            )
            actual_offset = offset

        start = max(0, actual_offset - context_len)
        end = min(len(doc_tokens), actual_offset + match_len + context_len)

        before_match = doc_tokens[start:actual_offset]
        match_tokens = doc_tokens[actual_offset : actual_offset + match_len]
        after_match = doc_tokens[actual_offset + match_len : end]

        decoded_before = self.tokenizer.decode(before_match)
        decoded_match = self.tokenizer.decode(match_tokens)
        decoded_after = self.tokenizer.decode(after_match)

        return ContextResult(
            before=decoded_before, match=decoded_match, after=decoded_after
        )

    def search(
        self,
        query: str,
        regex: str | None = None,
        context_len: int = 50,
    ) -> list[QueryResult]:
        """
        Finds all token representations for a query and returns all matches in the dataset.

        Args:
            query: The string to search for.
            regex: Optional regex pattern to filter token sequences.
            context_len: Number of tokens to show before and after the match.

        Returns:
            A list of dicts containing the matching sequence and the result location.
        """
        if regex is None:
            if self.verbose:
                print(r"Using default regex: \b{query}\b")
            regex = r"\b" + query + r"\b"

        sequences = self.bpe_searcher.search(query, regex_pattern=regex)
        if self.verbose:
            print(f"Found {len(sequences)} token sequences for query {query!r}")

        results = self.infinigram_searcher.query_sequences(
            sequences,
            verbose=self.verbose,
            post_process=context_len > 0,
        )

        if context_len > 0:
            for result in tqdm(
                results, disable=not self.verbose, desc="Extracting context"
            ):
                context = self._get_context(result, context_len=context_len)
                full_context = context.before + context.match + context.after
                if re.search(regex, full_context):
                    result.context = context

        results.sort(key=lambda x: (x.shard, x.sample_index))
        return results

    def display_single_result(self, res: QueryResult, context_len: int = 50) -> str:
        """Returns a string representing a single search result in context."""
        context = res.context
        if context is None:
            context = self._get_context(res, context_len)

        lines = [
            f"Shard: {res.shard}, Sample: {res.sample_index}, Offset: {res.token_offset}",
            f"Context: ...{context.before}**{context.match}**{context.after}...",
            "-" * 40,
        ]
        return "\n".join(lines)
