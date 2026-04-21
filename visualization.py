import html
import IPython
from collections import Counter
import matplotlib.pyplot as plt

from query_massive_tokens import QueryResult


def plot_shard_histogram(results: list[QueryResult]):
    """
    Plots a histogram of results per shard and returns the figure object.
    """
    shards = [r.shard for r in results]
    counts = Counter(shards)

    x = list(range(95))
    y = [counts.get(s, 0) for s in x]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x, y, color="#3498db", edgecolor="white")

    ax.set_xlabel("Shard", fontsize=8)
    ax.set_ylabel("Match Count", fontsize=8)
    ax.set_title(
        f"Distribution of Matches across Shards (Total: {len(results)})",
        fontsize=8,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xticks(range(0, 95, 5))

    fig.tight_layout()
    plt.close(fig)

    return fig


def plot_stacked_shard_histogram(results_map: dict[str, list["QueryResult"]]):
    """
    Plots a single histogram where shards are stacked by category with
    explicit y-axis headroom to prevent clipping.
    """
    x = list(range(95))
    fig, ax = plt.subplots(figsize=(8, 5))

    bottoms = [0] * 95
    total_matches = 0
    colors = plt.cm.tab10.colors

    for i, (label, results) in enumerate(results_map.items()):
        counts = Counter(r.shard for r in results)
        y = [counts.get(s, 0) for s in x]

        ax.bar(
            x,
            y,
            bottom=bottoms,
            label=label,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )

        # Update bottoms for the next category
        bottoms = [b + v for b, v in zip(bottoms, y)]
        total_matches += len(results)

    # --- THE FIX: SET EXPLICIT HEADROOM ---
    max_height = max(bottoms) if bottoms else 0
    if max_height > 0:
        # Adds 15% padding to the top so bars don't touch the title/border
        ax.set_ylim(0, max_height * 1.15)

    ax.set_xlabel("Shard", fontsize=9)
    ax.set_ylabel("Match Count", fontsize=9)
    ax.set_title(
        f"Distribution of Matches across Shards (Total: {total_matches})",
        fontsize=10,
        fontweight="bold",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xticks(range(0, 95, 5))

    # Place legend inside the plot, but the extra ylim ensures it won't
    # overlap the highest bars as easily
    ax.legend(fontsize=8, frameon=True, loc="upper right")

    fig.tight_layout()
    plt.close(fig)

    return fig


def visualize_result_html(
    res: QueryResult,
    context_len: int = 50,
):
    """
    Creates an HTML visualization of a single search result for Jupyter Notebooks.
    """
    context = res.context
    if context is None:
        raise ValueError("No context found for result!")

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
