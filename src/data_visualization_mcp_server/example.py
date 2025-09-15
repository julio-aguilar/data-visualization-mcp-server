"""
Example usage of the Data Visualization MCP Server
"""

import os
import asyncio
import time
from typing import Optional
from fastmcp import Client
import httpx


def _resolve_server_url() -> str:
    """Resolve the server URL from env or defaults.

    The server binds to 0.0.0.0 internally but clients should use a routable address.
    Precedence:
      1. ENV VAR: DATA_VIZ_SERVER_URL
      2. localhost default
    """
    return os.getenv("DATA_VIZ_SERVER_URL", "http://localhost:8000/mcp")


async def _connect_client(url: str, retries: int = 3, delay: float = 0.75) -> Optional[Client]:
    """Attempt to create and enter a Client context with simple retries."""
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        client = Client(url)
        try:
            await client.__aenter__()
            if attempt > 1:
                print(f"🔁 Connected on retry attempt {attempt}.")
            return client
        except Exception as e:  # broad catch to surface after retries
            last_error = e
            # Close partially opened client
            try:
                await client.__aexit__(type(e), e, e.__traceback__)
            except Exception:
                pass
            print(f"⚠️  Connection attempt {attempt} failed: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
    print("❌ Could not connect to server after retries.")
    if last_error:
        # Provide targeted hints
        if isinstance(last_error, httpx.ConnectError):
            print("🔍 Troubleshooting tips:\n  • Ensure the server is running (see instructions below).\n  • Avoid using 0.0.0.0 as the client URL; use localhost or 127.0.0.1.\n  • Check if another process is already using port 8000.\n  • On Windows, sometimes a firewall prompt needs confirmation.")
        else:
            print(f"🔍 Last error type: {type(last_error).__name__}")
    return None


async def demonstrate_server():
    """Demonstrate the key features of the visualization server."""

    server_url = _resolve_server_url()
    print(f"🚀 Connecting to Data Visualization Server at: {server_url}")

    client = await _connect_client(server_url)
    if client is None:
        print("💡 Start the server in another terminal with:\n   uv run python -m data_visualization_mcp_server.server\n   (or) uv run python server.py")
        return

    # Use async context manually managed in _connect_client
    try:
        try:
            # Get server information
            print("\n📋 Getting server info...")
            server_info = await client.call_tool("get_server_info")
            print(f"Server: {server_info['data']['server_name']}")
            print(f"Features: {len(server_info['data']['capabilities']['features'])} available")
            
            # Load a sample dataset
            print("\n📊 Loading sample dataset...")
            dataset_result = await client.call_tool("load_sample_dataset", {"dataset_name": "tips"})
            if dataset_result['success']:
                print(f"✅ Loaded dataset with shape: {dataset_result['data']['shape']}")
                print(f"Columns: {dataset_result['data']['columns']}")
            
            # Get dataset information
            print("\n🔍 Analyzing dataset...")
            info_result = await client.call_tool("get_current_dataset_info")
            if info_result['success']:
                data_info = info_result['data']
                print(f"📈 Numeric columns: {data_info['info']['numeric_columns']}")
                print(f"📋 Categorical columns: {data_info['info']['categorical_columns']}")
            
            # Get visualization suggestions
            print("\n💡 Getting visualization suggestions...")
            seaborn_suggestions = await client.call_tool("get_seaborn_suggestions")
            plotly_suggestions = await client.call_tool("get_plotly_suggestions")
            
            print(f"📊 Seaborn suggestions: {len(seaborn_suggestions['data'])} plots")
            for i, suggestion in enumerate(seaborn_suggestions['data'][:3]):
                print(f"  {i+1}. {suggestion['description']}")
            
            print(f"🎯 Plotly suggestions: {len(plotly_suggestions['data'])} charts")
            for i, suggestion in enumerate(plotly_suggestions['data'][:3]):
                print(f"  {i+1}. {suggestion['description']}")
            
            # Create a Seaborn plot
            print("\n📈 Creating Seaborn scatter plot...")
            seaborn_plot = await client.call_tool("create_seaborn_plot", {
                "plot_type": "scatter",
                "x_col": "total_bill",
                "y_col": "tip",
                "hue_col": "time",
                "title": "Restaurant Tips Analysis"
            })
            
            if seaborn_plot['success']:
                print("✅ Created Seaborn plot successfully")
                print(f"📊 Plot type: {seaborn_plot['data']['plot_type']}")
                print(f"📝 Columns used: {seaborn_plot['data']['columns_used']}")
            else:
                print(f"❌ Seaborn plot failed: {seaborn_plot['error']}")
            
            # Create a Plotly chart
            print("\n🎯 Creating interactive Plotly chart...")
            plotly_chart = await client.call_tool("create_plotly_chart", {
                "chart_type": "box",
                "x_col": "day",
                "y_col": "total_bill",
                "color_col": "time",
                "title": "Bill Distribution by Day"
            })
            
            if plotly_chart['success']:
                print("✅ Created Plotly chart successfully")
                print(f"📊 Chart type: {plotly_chart['data']['chart_type']}")
                print(f"🎨 Interactive: {plotly_chart['data']['interactive']}")
            else:
                print(f"❌ Plotly chart failed: {plotly_chart['error']}")
            
            # Try loading API data
            print("\n🌐 Loading API data...")
            api_result = await client.call_tool("load_api_data", {"api_type": "jsonplaceholder"})
            if api_result['success']:
                print(f"✅ Loaded API data with shape: {api_result['data']['shape']}")
                
                # Create a chart with API data
                print("\n📊 Creating chart with API data...")
                api_chart = await client.call_tool("create_plotly_chart", {
                    "chart_type": "histogram",
                    "x_col": "title_length",
                    "title": "Distribution of Post Title Lengths"
                })
                
                if api_chart['success']:
                    print("✅ Created API data chart successfully")
            else:
                print(f"⚠️ API data loading failed: {api_result['error']}")
            
            # List available options
            print("\n📋 Available options:")
            datasets = await client.call_tool("list_available_datasets")
            apis = await client.call_tool("list_available_apis")
            seaborn_types = await client.call_tool("list_seaborn_plot_types")
            plotly_types = await client.call_tool("list_plotly_chart_types")
            
            print(f"📊 Sample datasets: {len(datasets['data'])} available")
            print(f"🌐 API sources: {len(apis['data'])} available")
            print(f"📈 Seaborn plots: {len(seaborn_types['data'])} types")
            print(f"🎯 Plotly charts: {len(plotly_types['data'])} types")
            
            print("\n🎉 Demo completed successfully!")
            print("💡 Try creating your own visualizations by modifying the parameters above.")
            
        except Exception as e:
            print(f"❌ Error during demonstration: {e}")
            print("🔧 Verify the server is healthy and responding. You can restart it and re-run this demo.")
    finally:
        # Ensure proper exit of client context
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass


def main():
    """Run the demonstration."""
    print("🎨 Data Visualization MCP Server Demo")
    print("=" * 50)
    print("This demo will showcase the server's capabilities.")
    print("Make sure to start the server first with: uv run python -m data_visualization_mcp_server.server")
    print("(You can override the demo target URL via DATA_VIZ_SERVER_URL env var.)")
    print()
    
    asyncio.run(demonstrate_server())


if __name__ == "__main__":
    main()