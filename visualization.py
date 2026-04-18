from query_massive_tokens import QueryResult
from search_massive_tokens import MassiveTokenSearcher
import html
import IPython


def plot_shard_histogram(results: list[QueryResult]):
    """
    Plots a histogram of results per shard.
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    shards = [r.shard for r in results]
    counts = Counter(shards)

    # Show all shards from 0 to 94
    x = list(range(95))
    y = [counts.get(s, 0) for s in x]

    plt.figure(figsize=(15, 5))
    plt.bar(x, y, color="#3498db", edgecolor="white")
    plt.xlabel("Shard", fontsize=12)
    plt.ylabel("Match Count", fontsize=12)
    plt.title(f"Distribution of Matches across Shards (Total: {len(results)})", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Improve x-ticks readability
    plt.xticks(range(0, 95, 5))

    plt.tight_layout()
    plt.show()


def visualize_result_html(
    res: QueryResult,
    searcher: MassiveTokenSearcher,
    context_len: int = 50,
):
    """
    Creates an HTML visualization of a single search result for Jupyter Notebooks.
    """
    context = searcher.get_context(res, context_len)

    # Escape HTML special characters
    decoded_before = html.escape(context.before)
    decoded_match = html.escape(context.match)
    decoded_after = html.escape(context.after)

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
