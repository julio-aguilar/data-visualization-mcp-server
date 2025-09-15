"""
FastMCP Data Visualization Server

A FastMCP server that provides data visualization tools using Seaborn and Plotly.
Includes tools for creating statistical plots, interactive charts, and fetching sample data.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from fastmcp.utilities.types import Image
from mcp.types import ImageContent
from fastmcp import FastMCP

# Import our custom tools
from data_visualization_mcp_server.tools.data_sources import (
    get_sample_data,
    fetch_api_data,
    get_available_datasets,
    get_available_apis,
    get_data_info,
)
from data_visualization_mcp_server.tools.seaborn_tools import (
    create_seaborn_plots,
    get_seaborn_plot_suggestions,
    get_available_seaborn_plots,
)
from data_visualization_mcp_server.tools.plotly_tools import (
    create_plotly_charts,
    get_plotly_chart_suggestions,
    get_available_plotly_charts,
    create_dashboard,
)

# Initialize the FastMCP server
mcp = FastMCP(name="Data Visualization Server")

# Global variable to store the current dataset
current_dataset: Optional[pd.DataFrame] = None


@mcp.tool
def load_sample_dataset(dataset_name: str = "tips") -> Dict[str, Any]:
    """
    Load a sample dataset for visualization.
    
    Args:
        dataset_name: Name of the dataset to load. Use get_available_datasets() to see options.
    
    Returns:
        Dictionary with dataset information and first few rows
    """
    global current_dataset
    
    try:
        current_dataset = get_sample_data(dataset_name)
        
        return {
            'success': True,
            'dataset_name': dataset_name,
            'shape': current_dataset.shape,
            'columns': current_dataset.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in current_dataset.dtypes.to_dict().items()},
            'sample_data': current_dataset.head(5).to_dict('records'),
            'info': get_data_info(current_dataset)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'dataset_name': dataset_name
        }


@mcp.tool
def load_api_data(api_type: str = "jsonplaceholder") -> Dict[str, Any]:
    """
    Load data from a public API for visualization.
    
    Args:
        api_type: Type of API to fetch from. Use get_available_apis() to see options.
    
    Returns:
        Dictionary with API data information and first few rows
    """
    global current_dataset
    
    try:
        current_dataset = fetch_api_data(api_type)
        
        return {
            'success': True,
            'api_type': api_type,
            'shape': current_dataset.shape,
            'columns': current_dataset.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in current_dataset.dtypes.to_dict().items()},
            'sample_data': current_dataset.head(5).to_dict('records'),
            'info': get_data_info(current_dataset)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'api_type': api_type
        }


@mcp.tool
def list_available_datasets() -> List[str]:
    """
    Get list of available sample datasets.
    
    Returns:
        List of available dataset names
    """
    return get_available_datasets()


@mcp.tool
def list_available_apis() -> List[str]:
    """
    Get list of available API data sources.
    
    Returns:
        List of available API types
    """
    return get_available_apis()


@mcp.tool
def get_current_dataset_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded dataset.
    
    Returns:
        Dictionary with dataset information
    """
    if current_dataset is None:
        return {
            'success': False,
            'error': 'No dataset loaded. Use load_sample_dataset() or load_api_data() first.'
        }
    
    return {
        'success': True,
        'shape': current_dataset.shape,
        'columns': current_dataset.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in current_dataset.dtypes.to_dict().items()},
        'missing_values': {col: int(count) for col, count in current_dataset.isnull().sum().to_dict().items()},
        'sample_data': current_dataset.head(3).to_dict('records'),
        'info': get_data_info(current_dataset)
    }


@mcp.tool
def create_seaborn_plot(
    plot_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    hue_col: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a statistical plot using Seaborn.
    
    Args:
        plot_type: Type of plot (scatter, line, bar, histogram, box, violin, heatmap, etc.)
        x_col: Column name for x-axis
        y_col: Column name for y-axis  
        hue_col: Column name for color grouping
        title: Plot title
        save_path: Optional path to save the plot
    
    Returns:
        Dictionary with plot information and base64 encoded image
    """
    if current_dataset is None:
        return {
            'success': False,
            'error': 'No dataset loaded. Use load_sample_dataset() or load_api_data() first.'
        }
    
    return create_seaborn_plots(
        df=current_dataset,
        plot_type=plot_type,
        x_col=x_col,
        y_col=y_col,
        hue_col=hue_col,
        title=title,
        save_path=save_path
    )


@mcp.tool
def create_plotly_chart(
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an interactive chart using Plotly.
    
    Args:
        chart_type: Type of chart (scatter, line, bar, histogram, box, pie, heatmap, etc.)
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column name for color mapping
        size_col: Column name for size mapping
        title: Chart title
        save_path: Optional path to save the chart as HTML
    
    Returns:
        Dictionary with chart information and HTML content
    """
    if current_dataset is None:
        return {
            'success': False,
            'error': 'No dataset loaded. Use load_sample_dataset() or load_api_data() first.'
        }
    
    return create_plotly_charts(
        df=current_dataset,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        size_col=size_col,
        title=title,
        save_path=save_path
    )


@mcp.tool
def get_seaborn_suggestions() -> List[Dict[str, Any]]:
    """
    Get suggested Seaborn plots based on the current dataset.
    
    Returns:
        List of plot suggestions with parameters
    """
    if current_dataset is None:
        return [{
            'error': 'No dataset loaded. Use load_sample_dataset() or load_api_data() first.'
        }]
    
    return get_seaborn_plot_suggestions(current_dataset)


@mcp.tool  
def get_plotly_suggestions() -> List[Dict[str, Any]]:
    """
    Get suggested Plotly charts based on the current dataset.
    
    Returns:
        List of chart suggestions with parameters
    """
    if current_dataset is None:
        return [{
            'error': 'No dataset loaded. Use load_sample_dataset() or load_api_data() first.'
        }]
    
    return get_plotly_chart_suggestions(current_dataset)


@mcp.tool
def list_seaborn_plot_types() -> List[str]:
    """
    Get list of available Seaborn plot types.
    
    Returns:
        List of available plot types
    """
    return get_available_seaborn_plots()


@mcp.tool
def list_plotly_chart_types() -> List[str]:
    """
    Get list of available Plotly chart types.
    
    Returns:
        List of available chart types
    """
    return get_available_plotly_charts()


@mcp.tool
def create_visualization_dashboard(
    charts: List[Dict[str, Any]],
    title: str = "Data Dashboard",
    layout: str = "grid",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a multi-chart dashboard using Plotly.
    
    Args:
        charts: List of chart configurations, each containing chart_type and parameters
        title: Dashboard title
        layout: Layout type ('grid', 'vertical', 'horizontal')
        save_path: Optional path to save the dashboard
    
    Returns:
        Dictionary with dashboard information and HTML content
    """
    if current_dataset is None:
        return {
            'success': False,
            'error': 'No dataset loaded. Use load_sample_dataset() or load_api_data() first.'
        }
    
    return create_dashboard(
        df=current_dataset,
        charts=charts,
        title=title,
        layout=layout,
        save_path=save_path
    )


@mcp.tool
def export_current_dataset(
    file_path: str,
    file_format: str = "csv"
) -> Dict[str, Any]:
    """
    Export the current dataset to a file.
    
    Args:
        file_path: Path where to save the file
        file_format: Format to save in (csv, json, excel, parquet)
    
    Returns:
        Dictionary with export status
    """
    if current_dataset is None:
        return {
            'success': False,
            'error': 'No dataset loaded. Use load_sample_dataset() or load_api_data() first.'
        }
    
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format.lower() == "csv":
            current_dataset.to_csv(file_path, index=False)
        elif file_format.lower() == "json":
            current_dataset.to_json(file_path, orient='records', indent=2)
        elif file_format.lower() in ["excel", "xlsx"]:
            current_dataset.to_excel(file_path, index=False)
        elif file_format.lower() == "parquet":
            current_dataset.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return {
            'success': True,
            'file_path': str(file_path),
            'file_format': file_format,
            'rows_exported': len(current_dataset)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': file_path
        }


@mcp.tool
def get_server_info() -> Dict[str, Any]:
    """
    Get information about the data visualization server and its capabilities.
    
    Returns:
        Dictionary with server information
    """
    return {
        'server_name': 'Data Visualization Server',
        'description': 'FastMCP server for creating statistical and interactive visualizations',
        'capabilities': {
            'data_sources': {
                'sample_datasets': get_available_datasets(),
                'api_sources': get_available_apis()
            },
            'visualization_libraries': {
                'seaborn': {
                    'description': 'Statistical data visualization',
                    'plot_types': get_available_seaborn_plots()
                },
                'plotly': {
                    'description': 'Interactive data visualization',
                    'chart_types': get_available_plotly_charts()
                }
            },
            'features': [
                'Load sample datasets',
                'Fetch data from APIs',
                'Create statistical plots with Seaborn',
                'Create interactive charts with Plotly',
                'Get plot suggestions based on data',
                'Export datasets',
                'Create multi-chart dashboards'
            ]
        },
        'current_dataset_loaded': current_dataset is not None,
        'current_dataset_shape': current_dataset.shape if current_dataset is not None else None
    }


# Resources for configuration and help
@mcp.resource("config://server")  
def get_server_config() -> Dict[str, Any]:
    """Get server configuration information."""
    return {
        'name': 'Data Visualization Server',
        'version': '1.0.0',
        'supported_formats': ['png', 'html', 'svg'],
        'max_data_points': 10000,
        'default_figure_size': (10, 6),
        'available_color_schemes': ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    }


@mcp.resource("help://usage")
def get_usage_help() -> str:
    """Get usage instructions for the visualization server."""
    return """
    Data Visualization Server Usage:
    
    1. Load Data:
       - load_sample_dataset(dataset_name) - Load built-in datasets like 'tips', 'iris', etc.
       - load_api_data(api_type) - Load data from APIs like 'jsonplaceholder'
    
    2. Explore Data:
       - get_current_dataset_info() - Get info about loaded dataset
       - list_available_datasets() - See available sample datasets
       - list_available_apis() - See available API sources
    
    3. Create Visualizations:
       - create_seaborn_plot(plot_type, x_col, y_col, ...) - Statistical plots
       - create_plotly_chart(chart_type, x_col, y_col, ...) - Interactive charts
    
    4. Get Suggestions:
       - get_seaborn_suggestions() - Get plot suggestions for current dataset
       - get_plotly_suggestions() - Get chart suggestions for current dataset
    
    5. Advanced Features:
       - create_visualization_dashboard(charts) - Multi-chart dashboards
       - export_current_dataset(file_path) - Export data to file
    
    Example Workflow:
    1. load_sample_dataset("tips")
    2. get_current_dataset_info()
    3. get_seaborn_suggestions()
    4. create_seaborn_plot("scatter", "total_bill", "tip", "time")
    """


def main():
    """Main function to run the server."""
    print("🚀 Starting Data Visualization Server...")
    print("📊 Features: Seaborn statistical plots + Plotly interactive charts")
    print("📈 Sample datasets and API data sources included")
    print("🔧 Use get_server_info() to see all capabilities")
    
    # Run the server
    mcp.run(host="0.0.0.0", port=8000, transport="http")


if __name__ == "__main__":
    main()