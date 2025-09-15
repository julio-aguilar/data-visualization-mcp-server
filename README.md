# Data Visualization MCP Server 📊

A FastMCP server that provides powerful data visualization tools using Seaborn and Plotly. Create statistical plots, interactive charts, and dashboards with built-in sample datasets and API data sources.

## ✨ Features

- **📈 Statistical Plots**: Create publication-ready plots with Seaborn
- **🎯 Interactive Charts**: Build dynamic visualizations with Plotly  
- **📊 Sample Datasets**: Built-in datasets from Seaborn (tips, iris, flights, etc.)
- **🌐 API Data Sources**: Fetch real data from public APIs
- **🤖 Smart Suggestions**: Get plot recommendations based on your data
- **📋 Multi-Chart Dashboards**: Combine multiple visualizations
- **💾 Export Capabilities**: Save plots and data in various formats

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/julio-aguilar/data-visualization-mcp-server.git
cd data-visualization-mcp-server
```

2. **Install dependencies with uv:**
```bash
uv sync
```

3. **Run the server:**
```bash
uv run python server.py
```

The server will start on `http://localhost:8000` by default.

## 📖 Usage

### Basic Workflow

1. **Load a dataset:**
```python
# Load a sample dataset
load_sample_dataset("tips")

# Or fetch data from an API
load_api_data("jsonplaceholder")
```

2. **Explore your data:**
```python
# Get dataset information
get_current_dataset_info()

# See available datasets
list_available_datasets()
```

3. **Create visualizations:**
```python
# Statistical plot with Seaborn
create_seaborn_plot("scatter", x_col="total_bill", y_col="tip", hue_col="time")

# Interactive chart with Plotly
create_plotly_chart("histogram", x_col="total_bill", color_col="day")
```

4. **Get smart suggestions:**
```python
# Get plot recommendations
get_seaborn_suggestions()
get_plotly_suggestions()
```

### Available Tools

#### Data Management
- `load_sample_dataset(dataset_name)` - Load built-in datasets
- `load_api_data(api_type)` - Fetch data from APIs
- `get_current_dataset_info()` - Examine loaded data
- `export_current_dataset(file_path, format)` - Export data

#### Visualization  
- `create_seaborn_plot(plot_type, ...)` - Statistical plots
- `create_plotly_chart(chart_type, ...)` - Interactive charts
- `create_visualization_dashboard(charts)` - Multi-chart dashboards

#### Discovery
- `list_available_datasets()` - See sample datasets
- `list_available_apis()` - See API sources
- `get_seaborn_suggestions()` - Plot recommendations
- `get_plotly_suggestions()` - Chart recommendations
- `get_server_info()` - Server capabilities

## 📊 Supported Visualizations

### Seaborn (Statistical Plots)
- `scatter` - Scatter plots
- `line` - Line plots
- `bar` - Bar charts
- `histogram` - Histograms
- `box` - Box plots
- `violin` - Violin plots  
- `heatmap` - Correlation heatmaps
- `pair` - Pair plots
- `regression` - Regression plots
- `count` - Count plots

### Plotly (Interactive Charts)
- `scatter` - Interactive scatter plots
- `line` - Time series and line charts
- `bar` - Interactive bar charts
- `histogram` - Interactive histograms
- `box` - Interactive box plots
- `pie` - Pie charts
- `heatmap` - Interactive heatmaps
- `3d_scatter` - 3D scatter plots
- `sunburst` - Sunburst charts
- `treemap` - Treemaps
- `radar` - Radar charts

## 🗂️ Sample Datasets

Built-in datasets available via `load_sample_dataset()`:

- `tips` - Restaurant tip data
- `iris` - Iris flower measurements
- `flights` - Airline passenger data
- `car_crashes` - Car crash statistics  
- `titanic` - Titanic passenger data
- `mpg` - Car fuel efficiency
- `diamonds` - Diamond characteristics
- `custom` - Generated synthetic data

## 🌐 API Data Sources

Available via `load_api_data()`:

- `jsonplaceholder` - Sample posts and user data
- `httpbin` - HTTP testing data  
- `random_user` - Random user demographics

## 🔧 Example Usage

### Complete Example

```python
from fastmcp import Client
import asyncio

async def example_workflow():
    client = Client("http://localhost:8000")
    
    async with client:
        # Load sample data
        result = await client.call_tool("load_sample_dataset", {"dataset_name": "tips"})
        print("Dataset loaded:", result)
        
        # Get data info
        info = await client.call_tool("get_current_dataset_info")
        print("Columns:", info['columns'])
        
        # Create a scatter plot
        plot = await client.call_tool("create_seaborn_plot", {
            "plot_type": "scatter",
            "x_col": "total_bill", 
            "y_col": "tip",
            "hue_col": "time"
        })
        
        # Create interactive chart
        chart = await client.call_tool("create_plotly_chart", {
            "chart_type": "box",
            "x_col": "day",
            "y_col": "total_bill"
        })

# Run the example
asyncio.run(example_workflow())
```

### Creating Dashboards

```python
# Define multiple charts
charts = [
    {"chart_type": "histogram", "x_col": "total_bill"},
    {"chart_type": "scatter", "x_col": "total_bill", "y_col": "tip"},
    {"chart_type": "bar", "x_col": "day", "y_col": "total_bill"}
]

# Create dashboard
dashboard = await client.call_tool("create_visualization_dashboard", {
    "charts": charts,
    "title": "Restaurant Data Dashboard",
    "layout": "grid"
})
```

## 🏗️ Project Structure

```
data-visualization-mcp-server/
├── pyproject.toml          # Project configuration with uv
├── server.py               # Main FastMCP server
├── tools/                  # Visualization tools
│   ├── __init__.py
│   ├── data_sources.py     # Data loading utilities
│   ├── seaborn_tools.py    # Seaborn plotting functions
│   └── plotly_tools.py     # Plotly charting functions
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 🛠️ Development

### Running in Development Mode

```bash
# Install development dependencies
uv sync --dev

# Run with auto-reload
uv run python server.py

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .
```

### Adding New Visualizations

1. Add your plotting function to the appropriate tool module
2. Register it as a tool in `server.py` using `@mcp.tool`
3. Update the suggestions functions if needed
4. Add tests and documentation

## 📝 API Reference

### Tool Signatures

All tools return dictionaries with consistent structure:

```python
{
    'success': bool,           # Whether operation succeeded
    'data': Any,              # Main result data
    'error': str,             # Error message if failed
    'metadata': dict          # Additional information
}
```

### Configuration

The server uses sensible defaults but can be configured:

- Default port: 8000
- Default transport: HTTP
- Figure size: (10, 6)
- Max data points: 10,000

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastMCP](https://github.com/jlowin/fastmcp) - Python framework for MCP servers
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization
- [Plotly](https://plotly.com/python/) - Interactive plotting library
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/julio-aguilar/data-visualization-mcp-server/issues) page
2. Create a new issue with detailed information
3. Join the discussion in existing issues

---

**Happy Visualizing!** 📊✨

A powerful Model Context Protocol (MCP) server that enables AI assistants to create stunning data visualizations. Built with FastMCP framework and leveraging the best of both Plotly's interactive charts and Seaborn's statistical plotting capabilities.

## Features

- 📊 **Interactive Visualizations** - Create dynamic charts with Plotly
- 📈 **Statistical Plots** - Generate publication-ready statistical visualizations with Seaborn  
- 🚀 **FastMCP Integration** - High-performance MCP server implementation
- 🔧 **Easy Integration** - Seamlessly works with any MCP-compatible AI assistant
- 📱 **Web-Ready** - Export visualizations for web applications and dashboards

## Supported Chart Types

- Interactive scatter plots, line charts, bar charts, heatmaps
- Statistical distributions, regression plots, correlation matrices
- Time series visualizations, box plots, violin plots
- 3D plots, geographic maps, and custom dashboards

Built with Python, FastMCP, Plotly, and Seaborn.
