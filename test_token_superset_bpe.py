import unittest
from token_superset_bpe import BPETokenSupersetSearcher


class TestBPETokenSupersetSearcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.searcher = BPETokenSupersetSearcher()

    def test_edge_cases(self):
        test_strings = [
            "hello",
            " hello",
            "hello ",
            " hello ",
            "hello.",
            ".hello",
            " hello.",
            ".hello ",
            " hel.lo ",
            "subword",
            "ell",
            "lo ",
            " hel",
            "hello world",
        ]
        for target in test_strings:
            with self.subTest(target=target):
                results = self.searcher.search(target)
                self.assertGreater(
                    len(results),
                    0,
                    f"No results found for target: {target!r}",
                )
                for seq in results:
                    decoded = self.searcher.tokenizer.decode(seq)
                    self.assertIn(
                        target,
                        decoded,
                        f"Sequence {seq} decoded to {decoded!r}, which does not contain {target!r}",
                    )
                    self._assert_minimal(seq, target)
                    self._assert_valid_sequence(seq)

    def test_specific_sequence_inclusion(self):
        target = "hello"
        results = self.searcher.search(target)
        
        # tokenize(".hello ")
        seq_dot_hello_space = self.searcher.tokenizer.encode(".hello ", add_special_tokens=False)
        
        # Check if it is in results
        is_in_results = list(seq_dot_hello_space) in results
        
        # We expect it to NOT be in results because it is not minimal
        self.assertFalse(
            is_in_results,
            f"Sequence for '.hello ' {seq_dot_hello_space} was found in results for 'hello', but it should not be minimal.",
        )
        
        # tokenize("hello") should be in results
        seq_hello = self.searcher.tokenizer.encode("hello", add_special_tokens=False)
        self.assertIn(list(seq_hello), results, f"Sequence for 'hello' {seq_hello} should be in results for 'hello'")

    def test_specific_sequence_is_superset_of_minimal(self):
        cases = [
            ("hello", ".hello "),
            ("subword", ".subword "),
            (" hello", " hello ."),
            ("hello.", " hello. "),
        ]
        
        for target, extended in cases:
            with self.subTest(target=target, extended=extended):
                results = self.searcher.search(target)
                self.assertGreater(len(results), 0, f"No results for {target}")
                
                seq_extended = self.searcher.tokenizer.encode(extended, add_special_tokens=False)
                
                found_subsequence = False
                for res in results:
                    if self._is_subsequence(res, seq_extended):
                        found_subsequence = True
                        break
                        
                self.assertTrue(
                    found_subsequence,
                    f"Expected at least one result for {target!r} to be a subsequence of {extended!r} (tokens: {seq_extended})",
                )

    def _is_subsequence(self, sub, main):
        n = len(sub)
        for i in range(len(main) - n + 1):
            if main[i:i+n] == sub:
                return True
        return False

    def _assert_minimal(self, seq, target):
        # Verify that removing any token makes it not cover target
        # We use the same logic as the searcher for consistency,
        # or we can use the decoded string check.
        # Let's use the decoded string check as it's the high-level requirement.
        for i in range(len(seq)):
            sub_seq = seq[:i] + seq[i + 1 :]
            sub_str = self.searcher.tokenizer.decode(sub_seq)
            self.assertNotIn(
                target, sub_str, f"Sequence {seq} is not minimal for {target}"
            )

    def _assert_valid_sequence(self, seq):
        # Verify round-trip
        decoded = self.searcher.tokenizer.decode(seq)
        re_encoded = self.searcher.tokenizer.encode(decoded, add_special_tokens=False)
        self.assertEqual(
            seq,
            re_encoded,
            f"Sequence {seq} is not valid (re-encoded to {re_encoded})",
        )


if __name__ == "__main__":
    unittest.main()
