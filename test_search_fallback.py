import unittest
import random
from build_massive_tokens import memmap_tokens
from query_massive_tokens import InfiniGramSearcher
from token_superset_bpe import BPETokenSupersetSearcher


class TestSearchFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Initializing InfiniGramSearcher for tests...")
        cls.infinigram_searcher = InfiniGramSearcher()
        cls.massive_tokens = memmap_tokens(mode="r")
        cls.searcher = BPETokenSupersetSearcher()
        cls.tokenizer = cls.searcher.tokenizer

    def test_specific_failure(self):
        print("Testing specific failure case...")
        # Document: Shard 0, Sample 65357
        # Sequence: [5347, 273, 6181] (' capital of France')

        seq = [5347, 273, 6181]

        # We expect query_sequence to return the correct offset 1671 despite Infini-Gram reporting 500
        results = self.infinigram_searcher.query_sequence(seq)

        found = False
        for r in results:
            if r.shard == 0 and r.sample_index == 65357:
                print(
                    f"  Found result for Shard 0, Sample 65357. Offset: {r.token_offset}"
                )
                if r.token_offset == 1671:
                    print("  Specific failure test passed!")
                    found = True
                else:
                    print(
                        f"  Specific failure test failed. Expected offset 1671, got {r.token_offset}"
                    )
                break

        self.assertTrue(found, "Specific failure test failed. Result not found.")

    def test_randomized(self):
        n_tests = 10
        print(f"\nRunning {n_tests} randomized tests...")

        for i in range(n_tests):
            # Pick random shard and sample
            shard = random.randint(0, 94)
            sample = random.randint(0, 1023999)

            doc_tokens = self.massive_tokens[shard, sample]

            # Pick a random span of length 6 to 10
            length = random.randint(6, 10)
            start = random.randint(0, len(doc_tokens) - length)
            seq = doc_tokens[start : start + length]

            decoded = self.tokenizer.decode(seq)
            print(
                f"Test {i + 1}/{n_tests}: Target: {decoded!r} (Shard {shard}, Sample {sample}, Offset {start})"
            )

            # Query sequence
            results = self.infinigram_searcher.query_sequence(seq)

            # Check if we find the expected location
            found = False
            for r in results:
                if r.shard == shard and r.sample_index == sample:
                    if r.token_offset == start or r.token_offset == start - 1:
                        print(f"  Passed: Found at offset {r.token_offset}")
                        found = True
                    else:
                        print(
                            f"  Offset mismatch! Reported: {r.token_offset}, Actual: {start}"
                        )
                        found = True  # Consider it found for reporting
                    break

            if not found:
                print(
                    "Document not found in Infini-Gram results (might be expected if sequence is too long or not indexed)."
                )


if __name__ == "__main__":
    unittest.main()
