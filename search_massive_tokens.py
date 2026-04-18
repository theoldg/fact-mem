import html

import IPython.display
import fire
import numpy as np

from token_superset_bpe import BPETokenSupersetSearcher
from query_massive_tokens import InfiniGramSearcher, QueryResult


class MassiveTokenSearcher:
    def __init__(
        self,
        index_dir: str = "pile-data/index_dir",
        model_name: str = "EleutherAI/pythia-70m",
        max_workers: int = 96,
        verbose: bool = True,
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
        self.bpe_searcher = BPETokenSupersetSearcher(model_name=model_name)
        self.tokenizer = self.bpe_searcher.tokenizer

        self.infinigram_searcher = InfiniGramSearcher(
            index_dir=index_dir, max_workers=max_workers
        )

    def search(self, query: str, regex: str | None = None) -> list[QueryResult]:
        """
        Finds all token representations for a query and returns all matches in the dataset.

        Args:
            query: The string to search for.
            regex: Optional regex pattern to filter token sequences.

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

        return self.infinigram_searcher.query_sequences(sequences, verbose=self.verbose)

    def _get_context(self, res: QueryResult, context_len: int) -> tuple[str, str, str]:
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

        return decoded_before, decoded_match, decoded_after

    def display_single_result(self, res: QueryResult, context_len: int = 50) -> str:
        """Returns a string representing a single search result in context."""
        decoded_before, decoded_match, decoded_after = self._get_context(
            res, context_len
        )

        lines = [
            f"Shard: {res.shard}, Sample: {res.sample_index}, Offset: {res.token_offset}",
            f"Context: ...{decoded_before}**{decoded_match}**{decoded_after}...",
            "-" * 40,
        ]
        return "\n".join(lines)

    def visualize_result_html(self, res: QueryResult, context_len: int = 50):
        """Creates an HTML visualization of a single search result for Jupyter Notebooks."""
        decoded_before, decoded_match, decoded_after = self._get_context(
            res, context_len
        )

        # Escape HTML special characters
        decoded_before = html.escape(decoded_before)
        decoded_match = html.escape(decoded_match)
        decoded_after = html.escape(decoded_after)

        html_str = f"""
        <div style="
            border: 1px solid #e0e0e0; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 8px; 
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            background-color: #ffffff;
        ">
            <table style="
                width: 100%; 
                border-collapse: collapse; 
                margin-bottom: 15px;
                font-size: 13px;
            ">
                <thead>
                    <tr style="background-color: #f7f9fc; color: #5c6b73;">
                        <th style="padding: 8px; border-bottom: 1px solid #e0e0e0; text-align: center; font-weight: 600;">Shard</th>
                        <th style="padding: 8px; border-bottom: 1px solid #e0e0e0; text-align: center; font-weight: 600;">Sample Index</th>
                        <th style="padding: 8px; border-bottom: 1px solid #e0e0e0; text-align: center; font-weight: 600;">Token Offset</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="color: #2c3e50;">
                        <td style="padding: 8px; text-align: center;">{res.shard}</td>
                        <td style="padding: 8px; text-align: center;">{res.sample_index}</td>
                        <td style="padding: 8px; text-align: center;">{res.token_offset}</td>
                    </tr>
                </tbody>
            </table>
            <div style="
                font-size: 15px; 
                line-height: 1.6; 
                color: #333;
                padding: 10px;
                background-color: #fafafa;
                border-radius: 4px;
            ">
                <span style="color: #999;">...</span>{decoded_before}<span style="color: #e74c3c; font-weight: 700; background-color: #fdedec; padding: 2px 4px; border-radius: 3px;">{decoded_match}</span>{decoded_after}<span style="color: #999;">...</span>
            </div>
        </div>
        """
        return IPython.display.HTML(html_str)

    def display(
        self,
        query: str,
        context_len: int = 50,
        limit: int = 10,
        regex: str | None = None,
    ):
        """
        Searches for a query and displays the results in context.

        Args:
            query: The string to search for.
            context_len: Number of tokens to show before and after the match.
            limit: Maximum number of results to display.
            regex: Optional regex pattern to filter token sequences.
        """
        results = self.search(query, regex=regex)

        if not results:
            print("No results found.")
            return

        print(f"\nTotal matches found across all sequences: {len(results)}")
        print(f"Displaying top {min(limit, len(results))} results for query: {query!r}")
        print("=" * 80)

        for res in results[:limit]:
            print(self.display_result(res, context_len=context_len))


if __name__ == "__main__":
    fire.Fire(MassiveTokenSearcher)
