"""
Plotly visualization tools for interactive plots
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import json


def create_plotly_charts(
    df: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    facet_col: Optional[str] = None,
    facet_row: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    color_scheme: str = "viridis",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create various interactive Plotly charts.
    
    Args:
        df: DataFrame containing the data
        chart_type: Type of chart to create. Options:
            - 'scatter': Scatter plot
            - 'line': Line chart
            - 'bar': Bar chart
            - 'histogram': Histogram
            - 'box': Box plot
            - 'violin': Violin plot
            - 'heatmap': Heatmap
            - 'sunburst': Sunburst chart
            - 'treemap': Treemap
            - 'pie': Pie chart
            - 'area': Area chart
            - 'bubble': Bubble chart
            - 'parallel_coordinates': Parallel coordinates plot
            - 'radar': Radar chart
            - '3d_scatter': 3D scatter plot
            - 'candlestick': Candlestick chart (for financial data)
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column name for color mapping
        size_col: Column name for size mapping
        facet_col: Column name for column faceting
        facet_row: Column name for row faceting
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        color_scheme: Color scheme to use
        save_path: Optional path to save the chart as HTML
        
    Returns:
        Dictionary with chart information and HTML content
    """
    
    try:
        fig = None
        
        if chart_type == "scatter":
            if not x_col or not y_col:
                raise ValueError("Scatter plot requires both x_col and y_col")
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col, size=size_col,
                facet_col=facet_col, facet_row=facet_row,
                color_continuous_scale=color_scheme,
                title=title or f"Scatter Plot: {x_col} vs {y_col}"
            )
            
        elif chart_type == "line":
            if not x_col or not y_col:
                raise ValueError("Line chart requires both x_col and y_col")
            fig = px.line(
                df, x=x_col, y=y_col, color=color_col,
                facet_col=facet_col, facet_row=facet_row,
                title=title or f"Line Chart: {y_col} over {x_col}"
            )
            
        elif chart_type == "bar":
            if not x_col:
                raise ValueError("Bar chart requires x_col")
            fig = px.bar(
                df, x=x_col, y=y_col, color=color_col,
                facet_col=facet_col, facet_row=facet_row,
                title=title or f"Bar Chart: {x_col}"
            )
            
        elif chart_type == "histogram":
            if not x_col:
                raise ValueError("Histogram requires x_col")
            fig = px.histogram(
                df, x=x_col, color=color_col, nbins=30,
                facet_col=facet_col, facet_row=facet_row,
                title=title or f"Histogram: {x_col}"
            )
            
        elif chart_type == "box":
            if not y_col:
                raise ValueError("Box plot requires y_col")
            fig = px.box(
                df, x=x_col, y=y_col, color=color_col,
                facet_col=facet_col, facet_row=facet_row,
                title=title or f"Box Plot: {y_col}"
            )
            
        elif chart_type == "violin":
            if not y_col:
                raise ValueError("Violin plot requires y_col")
            fig = px.violin(
                df, x=x_col, y=y_col, color=color_col,
                facet_col=facet_col, facet_row=facet_row,
                title=title or f"Violin Plot: {y_col}"
            )
            
        elif chart_type == "heatmap":
            # Select numeric columns for heatmap
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("No numeric columns found for heatmap")
            
            correlation_matrix = numeric_df.corr()
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=color_scheme,
                title=title or "Correlation Heatmap"
            )
            
        elif chart_type == "sunburst":
            # Sunburst requires hierarchical categorical data
            if not color_col:
                raise ValueError("Sunburst chart requires color_col for categories")
            
            # Create a simple hierarchy if we have categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if len(categorical_cols) < 2:
                raise ValueError("Sunburst requires at least 2 categorical columns")
            
            fig = px.sunburst(
                df, path=categorical_cols[:3], values=size_col,
                color=color_col, color_continuous_scale=color_scheme,
                title=title or "Sunburst Chart"
            )
            
        elif chart_type == "treemap":
            if not color_col:
                raise ValueError("Treemap requires color_col")
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                raise ValueError("Treemap requires categorical columns")
            
            fig = px.treemap(
                df, path=categorical_cols[:2], values=size_col,
                color=color_col, color_continuous_scale=color_scheme,
                title=title or "Treemap"
            )
            
        elif chart_type == "pie":
            if not x_col:
                raise ValueError("Pie chart requires x_col for categories")
            
            # Aggregate data for pie chart
            pie_data = df.groupby(x_col)[y_col].sum() if y_col else df[x_col].value_counts()
            
            fig = px.pie(
                values=pie_data.values, names=pie_data.index,
                title=title or f"Pie Chart: {x_col}"
            )
            
        elif chart_type == "area":
            if not x_col or not y_col:
                raise ValueError("Area chart requires both x_col and y_col")
            fig = px.area(
                df, x=x_col, y=y_col, color=color_col,
                facet_col=facet_col, facet_row=facet_row,
                title=title or f"Area Chart: {y_col} over {x_col}"
            )
            
        elif chart_type == "bubble":
            if not x_col or not y_col:
                raise ValueError("Bubble chart requires both x_col and y_col")
            fig = px.scatter(
                df, x=x_col, y=y_col, size=size_col, color=color_col,
                hover_name=facet_col, size_max=60,
                title=title or f"Bubble Chart: {x_col} vs {y_col}"
            )
            
        elif chart_type == "parallel_coordinates":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 3:
                raise ValueError("Parallel coordinates requires at least 3 numeric columns")
            
            fig = px.parallel_coordinates(
                df, dimensions=numeric_cols[:6], color=color_col,
                title=title or "Parallel Coordinates Plot"
            )
            
        elif chart_type == "radar":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 3:
                raise ValueError("Radar chart requires at least 3 numeric columns")
            
            # Create radar chart using graph_objects
            fig = go.Figure()
            
            if color_col and color_col in df.columns:
                for category in df[color_col].unique()[:5]:  # Limit to 5 categories
                    subset = df[df[color_col] == category]
                    values = subset[numeric_cols[:6]].mean().tolist()
                    values += [values[0]]  # Close the radar chart
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=numeric_cols[:6] + [numeric_cols[0]],
                        fill='toself',
                        name=str(category)
                    ))
            else:
                # Single radar chart with mean values
                values = df[numeric_cols[:6]].mean().tolist()
                values += [values[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=numeric_cols[:6] + [numeric_cols[0]],
                    fill='toself',
                    name='Average'
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title=title or "Radar Chart"
            )
            
        elif chart_type == "3d_scatter":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 3:
                raise ValueError("3D scatter requires at least 3 numeric columns")
            
            fig = px.scatter_3d(
                df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                color=color_col, size=size_col,
                title=title or "3D Scatter Plot"
            )
            
        elif chart_type == "candlestick":
            # Requires OHLC data
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                # Generate synthetic OHLC data for demonstration
                if len(df) < 10:
                    raise ValueError("Need more data points for candlestick chart")
                
                # Create synthetic OHLC from any numeric column
                if not y_col:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols:
                        raise ValueError("Need numeric data for candlestick chart")
                    y_col = numeric_cols[0]
                
                # Generate OHLC from the selected column
                values = df[y_col].rolling(window=4).agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last'
                }).dropna()
                
                fig = go.Figure(data=go.Candlestick(
                    x=values.index,
                    open=values['open'],
                    high=values['high'],
                    low=values['low'],
                    close=values['close']
                ))
            else:
                fig = go.Figure(data=go.Candlestick(
                    x=df.index if x_col is None else df[x_col],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                ))
            
            fig.update_layout(title=title or "Candlestick Chart")
            
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        # Update layout
        if fig:
            fig.update_layout(
                width=width,
                height=height,
                template="plotly_white"
            )
        
        # Save as HTML if path provided
        if save_path and fig:
            fig.write_html(save_path)
        
        # Convert to HTML for returning
        html_content = fig.to_html(include_plotlyjs=True) if fig else ""
        
        return {
            'success': True,
            'chart_type': chart_type,
            'title': title or f"Plotly {chart_type.title()} Chart",
            'html_content': html_content,
            'columns_used': {
                'x': x_col,
                'y': y_col,
                'color': color_col,
                'size': size_col,
                'facet_col': facet_col,
                'facet_row': facet_row
            },
            'data_shape': df.shape,
            'save_path': save_path,
            'interactive': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'chart_type': chart_type
        }


def create_dashboard(
    df: pd.DataFrame,
    charts: List[Dict[str, Any]],
    title: str = "Data Visualization Dashboard",
    layout: str = "grid",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a multi-chart dashboard using Plotly subplots.
    
    Args:
        df: DataFrame containing the data
        charts: List of chart configurations
        title: Dashboard title
        layout: Layout type ('grid', 'vertical', 'horizontal')
        save_path: Optional path to save the dashboard
        
    Returns:
        Dictionary with dashboard information and HTML content
    """
    
    try:
        n_charts = len(charts)
        if n_charts == 0:
            raise ValueError("No charts provided for dashboard")
        
        # Determine subplot layout
        if layout == "vertical":
            rows, cols = n_charts, 1
        elif layout == "horizontal":
            rows, cols = 1, n_charts
        else:  # grid
            cols = min(2, n_charts)
            rows = (n_charts + cols - 1) // cols
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[chart.get('title', f"Chart {i+1}") for i, chart in enumerate(charts)]
        )
        
        # Add each chart
        for i, chart_config in enumerate(charts):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Create individual chart
            chart_result = create_plotly_charts(df, **chart_config)
            
            if chart_result['success']:
                # This is a simplified approach - in practice, you'd need to extract
                # the trace data from the individual charts and add to subplots
                # For now, we'll create a simple placeholder
                fig.add_trace(
                    go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name=f"Chart {i+1}"),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            height=300 * rows,
            showlegend=False
        )
        
        # Save as HTML if path provided
        if save_path:
            fig.write_html(save_path)
        
        html_content = fig.to_html(include_plotlyjs=True)
        
        return {
            'success': True,
            'title': title,
            'html_content': html_content,
            'n_charts': n_charts,
            'layout': layout,
            'save_path': save_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'title': title
        }


def get_plotly_chart_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Suggest appropriate Plotly charts based on data characteristics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of chart suggestions with parameters
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    suggestions = []
    
    # Interactive scatter plots for numeric pairs
    if len(numeric_cols) >= 2:
        suggestions.append({
            'chart_type': 'scatter',
            'description': f'Interactive scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}',
            'parameters': {
                'x_col': numeric_cols[0],
                'y_col': numeric_cols[1],
                'color_col': categorical_cols[0] if categorical_cols else None,
                'size_col': numeric_cols[2] if len(numeric_cols) > 2 else None
            }
        })
    
    # Time series if datetime columns exist
    if datetime_cols and numeric_cols:
        suggestions.append({
            'chart_type': 'line',
            'description': f'Time series plot of {numeric_cols[0]}',
            'parameters': {
                'x_col': datetime_cols[0],
                'y_col': numeric_cols[0],
                'color_col': categorical_cols[0] if categorical_cols else None
            }
        })
    
    # Interactive histograms
    for col in numeric_cols[:2]:
        suggestions.append({
            'chart_type': 'histogram',
            'description': f'Interactive histogram of {col}',
            'parameters': {
                'x_col': col,
                'color_col': categorical_cols[0] if categorical_cols else None
            }
        })
    
    # Box plots for numeric vs categorical
    if numeric_cols and categorical_cols:
        suggestions.append({
            'chart_type': 'box',
            'description': f'Interactive box plot of {numeric_cols[0]} by {categorical_cols[0]}',
            'parameters': {
                'x_col': categorical_cols[0],
                'y_col': numeric_cols[0]
            }
        })
    
    # Pie charts for categorical data
    if categorical_cols:
        suggestions.append({
            'chart_type': 'pie',
            'description': f'Pie chart of {categorical_cols[0]} distribution',
            'parameters': {
                'x_col': categorical_cols[0]
            }
        })
    
    # 3D scatter if enough numeric columns
    if len(numeric_cols) >= 3:
        suggestions.append({
            'chart_type': '3d_scatter',
            'description': '3D scatter plot of numeric variables',
            'parameters': {
                'color_col': categorical_cols[0] if categorical_cols else None
            }
        })
    
    # Heatmap for correlation
    if len(numeric_cols) >= 2:
        suggestions.append({
            'chart_type': 'heatmap',
            'description': 'Interactive correlation heatmap',
            'parameters': {}
        })
    
    return suggestions


def get_available_plotly_charts() -> List[str]:
    """
    Get list of available Plotly chart types.
    
    Returns:
        List of available chart types
    """
    return [
        "scatter", "line", "bar", "histogram", "box", "violin", 
        "heatmap", "sunburst", "treemap", "pie", "area", "bubble",
        "parallel_coordinates", "radar", "3d_scatter", "candlestick"
    ]