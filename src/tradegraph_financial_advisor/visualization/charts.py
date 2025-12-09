from pathlib import Path

import plotly.graph_objects as go


def create_portfolio_allocation_chart(
    recommendations, output_path="results/portfolio_allocation.png"
):
    """
    Creates a pie chart showing portfolio allocation

    Args:
        recommendations: List of recommendation dicts from analysis results
        output_path: Where to save the HTML file
    """
    if not recommendations:
        raise ValueError("No recommendations supplied for allocation chart")

    symbols = [rec.get("symbol", "?") for rec in recommendations]
    allocations = []
    for rec in recommendations:
        allocation_value = rec.get("recommended_allocation")
        if allocation_value is None:
            allocation_value = rec.get("allocation_percentage")
        if allocation_value is None:
            # fall back to max_position_size / portfolio size if present
            portfolio_size = rec.get("portfolio_size") or 1
            max_position = rec.get("max_position_size")
            allocation_value = (
                (max_position / portfolio_size) if max_position and portfolio_size else 0
            )
        allocations.append(float(allocation_value) * 100)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=symbols,
                values=allocations,
                hole=0.3,
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )

    fig.update_layout(title="Portfolio Allocation Recommendation", showlegend=True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_image(str(path))
    except ValueError as exc:
        raise RuntimeError(
            "Plotly static image export requires the kaleido package."
        ) from exc

    print(f"Chart saved to: {path}")

    return str(path)
