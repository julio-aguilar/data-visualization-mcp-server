"""
Simple test to verify the server setup
"""

import pytest
import pandas as pd
from data_visualization_mcp_server.tools.data_sources import get_sample_data, get_available_datasets
from data_visualization_mcp_server.tools.seaborn_tools import create_seaborn_plots, get_available_seaborn_plots
from data_visualization_mcp_server.tools.plotly_tools import create_plotly_charts, get_available_plotly_charts


def test_data_sources():
    """Test data loading functionality."""
    # Test sample data loading
    datasets = get_available_datasets()
    assert len(datasets) > 0
    assert "tips" in datasets
    
    # Load tips dataset
    df = get_sample_data("tips")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "total_bill" in df.columns


def test_seaborn_tools():
    """Test Seaborn plotting functionality."""
    # Get available plot types
    plot_types = get_available_seaborn_plots()
    assert len(plot_types) > 0
    assert "scatter" in plot_types
    
    # Create a simple plot
    df = get_sample_data("tips")
    result = create_seaborn_plots(
        df=df,
        plot_type="scatter",
        x_col="total_bill",
        y_col="tip"
    )
    
    assert result["success"] is True
    assert "image_base64" in result
    assert result["plot_type"] == "scatter"


def test_plotly_tools():
    """Test Plotly charting functionality."""
    # Get available chart types
    chart_types = get_available_plotly_charts()
    assert len(chart_types) > 0
    assert "scatter" in chart_types
    
    # Create a simple chart
    df = get_sample_data("tips")
    result = create_plotly_charts(
        df=df,
        chart_type="scatter",
        x_col="total_bill",
        y_col="tip"
    )
    
    assert result["success"] is True
    assert "html_content" in result
    assert result["chart_type"] == "scatter"
    assert result["interactive"] is True


def test_error_handling():
    """Test error handling for invalid inputs."""
    df = get_sample_data("tips")
    
    # Test invalid plot type
    result = create_seaborn_plots(
        df=df,
        plot_type="invalid_plot"
    )
    assert result["success"] is False
    assert "error" in result
    
    # Test missing required columns
    result = create_plotly_charts(
        df=df,
        chart_type="scatter"  # Missing x_col and y_col
    )
    assert result["success"] is False
    assert "error" in result


if __name__ == "__main__":
    # Run tests manually if pytest not available
    print("🧪 Running basic tests...")
    
    try:
        test_data_sources()
        print("✅ Data sources test passed")
        
        test_seaborn_tools()
        print("✅ Seaborn tools test passed")
        
        test_plotly_tools()
        print("✅ Plotly tools test passed")
        
        test_error_handling()
        print("✅ Error handling test passed")
        
        print("\n🎉 All tests passed! The server setup is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("🔧 Check your dependencies and try running: uv sync")