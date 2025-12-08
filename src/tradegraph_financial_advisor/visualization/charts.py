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
    symbols = [rec["symbol"] for rec in recommendations]
    allocations = [rec["allocation_percentage"] * 100 for rec in recommendations]

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
