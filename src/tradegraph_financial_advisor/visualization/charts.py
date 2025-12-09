import plotly.graph_objects as go


def create_portfolio_allocation_chart(
    recommendations, output_path="portfolio_allocation.html"
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

    # Save to HTML file
    fig.write_html(output_path)
    print(f"Chart saved to: {output_path}")

    return output_path
