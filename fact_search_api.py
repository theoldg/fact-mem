from dataclasses import dataclass
import urllib.request
import urllib.error
import json
import re
from typing import List, Dict, Optional


@dataclass
class Occurrence:
    shard: int
    rank: int
    step: float
    checkpoint: int
    context: str
    matched_anchor: str


class DolmaSearcher:
    def __init__(self, index="v4_dolma-v1_7_llama"):
        self.index = index
        self.url = "https://api.infini-gram.io/"
        self.batch_size = 4000000  # Default for OLMo 7B

    def search(self, query_text: str) -> Optional[Dict]:
        payload = {"index": self.index, "query_type": "find", "query": query_text}
        return self._make_request(payload)

    def get_doc(self, shard: int, rank: int, query_text: str) -> Optional[Dict]:
        payload = {
            "index": self.index,
            "query_type": "get_doc_by_rank",
            "s": shard,
            "rank": rank,
            "query": query_text,
        }
        return self._make_request(payload)

    def _make_request(self, payload: Dict) -> Optional[Dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as response:
                res_data = response.read().decode("utf-8")
                return json.loads(res_data)
        except urllib.error.HTTPError:
            return None
        except Exception:
            return None

    def estimate_step(self, token_index: int) -> float:
        return token_index / self.batch_size


def search_and_filter(
    anchors: List[str],
    regex_pattern: str,
    searcher: DolmaSearcher,
    max_results=10,
    max_candidates=500,  # Increased limit
) -> List[Occurrence]:
    results = []
    seen_ranks = set()  # To avoid duplicate results across anchors

    regex = re.compile(regex_pattern, re.IGNORECASE)
    candidates_checked = 0
    hit_limit = False

    for anchor in anchors:
        print(f"Searching for anchor: '{anchor}'...")
        res = searcher.search(anchor)
        if not res or "segment_by_shard" not in res:
            continue

        for shard_idx, ranks in enumerate(res["segment_by_shard"]):
            if not ranks:
                continue

            ranges = ranks
            if isinstance(ranks[0], int):
                ranges = [ranks]

            for r in ranges:
                start_rank = r[0]
                end_rank = r[1]

                for rank in range(start_rank, end_rank + 1):
                    print(f"Debug: Checking shard {shard_idx}, rank {rank}...")
                    if len(results) >= max_results:
                        break
                    if candidates_checked >= max_candidates:
                        hit_limit = True
                        break

                    if rank in seen_ranks:
                        continue
                    seen_ranks.add(rank)

                    candidates_checked += 1
                    doc_res = searcher.get_doc(shard_idx, rank, anchor)
                    if doc_res and "spans" in doc_res:
                        text = " ".join([span[0] for span in doc_res["spans"]])
                        if candidates_checked == 1:
                            print(f"Debug: First candidate text snippet: {text[:100]}")

                        if regex.search(text):
                            step = searcher.estimate_step(rank)
                            checkpoint = int(step / 1000) * 1000

                            results.append(
                                Occurrence(
                                    shard=shard_idx,
                                    rank=rank,
                                    step=step,
                                    checkpoint=checkpoint,
                                    context=text,
                                    matched_anchor=anchor,
                                )
                            )

                if len(results) >= max_results or hit_limit:
                    break
            if len(results) >= max_results or hit_limit:
                break

    if hit_limit:
        print(
            f"Notice: Reached max candidates limit ({max_candidates}). Results may be incomplete."
        )

    results.sort(key=lambda x: x.step)
    return results


def interactive_session(anchors: List[str], regex_pattern: str, max_results=10):
    searcher = DolmaSearcher()
    results = search_and_filter(anchors, regex_pattern, searcher, max_results)

    print(f"\nFound {len(results)} matching occurrences (filtered):\n")

    regex = re.compile(regex_pattern, re.IGNORECASE)

    for i, res in enumerate(results):
        print(f"--- Result {i + 1} ---")
        print(f"Shard: {res.shard}, Rank: {res.rank}")
        print(f"Estimated Step: {res.step:.2f} (Checkpoint: {res.checkpoint})")
        print(f"Matched Anchor: '{res.matched_anchor}'")

        # Show context around anchor
        query_pos = res.context.lower().find(res.matched_anchor.lower())
        if query_pos != -1:
            start_pos = max(0, query_pos - 50)
            end_pos = min(len(res.context), query_pos + 50)
            snippet = res.context[start_pos:end_pos]
            print(f"Anchor Context: ...{snippet}...")

        # Show context around regex match
        match = regex.search(res.context)
        if match:
            match_start = match.start()
            start_pos = max(0, match_start - 50)
            end_pos = min(len(res.context), match_start + 50)
            snippet = res.context[start_pos:end_pos]
            print(f"Regex Context:  ...{snippet}...")

        print("-" * 40)


if __name__ == "__main__":
    # Example usage
    interactive_session(
        anchors=["Paris", "paris"], regex_pattern="capital", max_results=5
    )
