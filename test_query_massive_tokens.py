import unittest
from dataclasses import dataclass
from query_massive_tokens import query_sequence
from build_massive_tokens import memmap_tokens


@dataclass
class TestCase:
    shard: int
    sample: int
    start: int
    end: int


class TestQueryMassiveTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.massive_tokens = memmap_tokens(mode="r")

    def test_query_sequences(self):
        # Define test cases using dataclass
        test_cases = [
            TestCase(shard=0, sample=0, start=10, end=20),
            TestCase(shard=1, sample=5, start=20, end=30),
            TestCase(shard=2, sample=10, start=50, end=60),
            TestCase(shard=10, sample=0, start=10, end=20),
            TestCase(shard=20, sample=5, start=20, end=30),
        ]

        for case in test_cases:
            with self.subTest(
                shard=case.shard, sample=case.sample, start=case.start, end=case.end
            ):
                seq = self.massive_tokens[
                    case.shard, case.sample, case.start : case.end
                ]
                results = query_sequence(seq)

                found = False
                for res in results:
                    if (
                        res.shard == case.shard
                        and res.sample_index == case.sample
                        and res.token_offset == case.start
                    ):
                        found = True
                        break
                self.assertTrue(
                    found,
                    f"Sequence not found in shard {case.shard}, sample {case.sample}, offset {case.start}. Results: {results}",
                )


if __name__ == "__main__":
    unittest.main()
