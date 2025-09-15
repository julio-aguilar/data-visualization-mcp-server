"""
Seaborn visualization tools for statistical plots
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import io
import base64
from pathlib import Path


def create_seaborn_plots(
    df: pd.DataFrame,
    plot_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    hue_col: Optional[str] = None,
    style_col: Optional[str] = None,
    size_col: Optional[str] = None,
    col_col: Optional[str] = None,
    row_col: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    palette: str = "viridis"
) -> Dict[str, Any]:
    """
    Create various Seaborn statistical plots.
    
    Args:
        df: DataFrame containing the data
        plot_type: Type of plot to create. Options:
            - 'scatter': Scatter plot
            - 'line': Line plot  
            - 'bar': Bar plot
            - 'histogram': Histogram
            - 'box': Box plot
            - 'violin': Violin plot
            - 'heatmap': Correlation heatmap
            - 'pair': Pair plot
            - 'distribution': Distribution plot
            - 'regression': Regression plot
            - 'count': Count plot
            - 'strip': Strip plot
            - 'swarm': Swarm plot
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        hue_col: Column name for color grouping
        style_col: Column name for style grouping
        size_col: Column name for size mapping
        col_col: Column name for column faceting
        row_col: Column name for row faceting
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size as (width, height)
        palette: Color palette to use
        
    Returns:
        Dictionary with plot information and base64 encoded image
    """
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = figsize
    
    # Create the plot based on type
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        if plot_type == "scatter":
            if not x_col or not y_col:
                raise ValueError("Scatter plot requires both x_col and y_col")
            sns.scatterplot(
                data=df, x=x_col, y=y_col, hue=hue_col, 
                style=style_col, size=size_col, palette=palette, ax=ax
            )
            
        elif plot_type == "line":
            if not x_col or not y_col:
                raise ValueError("Line plot requires both x_col and y_col")
            sns.lineplot(
                data=df, x=x_col, y=y_col, hue=hue_col, 
                style=style_col, palette=palette, ax=ax
            )
            
        elif plot_type == "bar":
            if not x_col:
                raise ValueError("Bar plot requires x_col")
            sns.barplot(
                data=df, x=x_col, y=y_col, hue=hue_col, 
                palette=palette, ax=ax
            )
            
        elif plot_type == "histogram":
            if not x_col:
                raise ValueError("Histogram requires x_col")
            sns.histplot(
                data=df, x=x_col, hue=hue_col, 
                bins=30, palette=palette, ax=ax
            )
            
        elif plot_type == "box":
            if not y_col:
                raise ValueError("Box plot requires y_col")
            sns.boxplot(
                data=df, x=x_col, y=y_col, hue=hue_col, 
                palette=palette, ax=ax
            )
            
        elif plot_type == "violin":
            if not y_col:
                raise ValueError("Violin plot requires y_col")
            sns.violinplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                palette=palette, ax=ax
            )
            
        elif plot_type == "heatmap":
            # Select only numeric columns for correlation
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("No numeric columns found for heatmap")
            
            correlation_matrix = numeric_df.corr()
            sns.heatmap(
                correlation_matrix, annot=True, cmap=palette, 
                center=0, square=True, ax=ax
            )
            
        elif plot_type == "pair":
            # Close the current figure since pairplot creates its own
            plt.close(fig)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for pair plot")
            
            g = sns.pairplot(
                df, vars=numeric_cols[:4], hue=hue_col, 
                palette=palette, height=3
            )
            fig = g.fig
            
        elif plot_type == "distribution":
            if not x_col:
                raise ValueError("Distribution plot requires x_col")
            sns.histplot(
                data=df, x=x_col, hue=hue_col, kde=True,
                palette=palette, ax=ax
            )
            
        elif plot_type == "regression":
            if not x_col or not y_col:
                raise ValueError("Regression plot requires both x_col and y_col")
            sns.regplot(
                data=df, x=x_col, y=y_col, ax=ax
            )
            if hue_col:
                # Add color grouping
                for name, group in df.groupby(hue_col):
                    ax.scatter(group[x_col], group[y_col], label=name, alpha=0.7)
                ax.legend()
                
        elif plot_type == "count":
            if not x_col:
                raise ValueError("Count plot requires x_col")
            sns.countplot(
                data=df, x=x_col, hue=hue_col, 
                palette=palette, ax=ax
            )
            
        elif plot_type == "strip":
            if not y_col:
                raise ValueError("Strip plot requires y_col")
            sns.stripplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                palette=palette, ax=ax
            )
            
        elif plot_type == "swarm":
            if not y_col:
                raise ValueError("Swarm plot requires y_col")
            sns.swarmplot(
                data=df, x=x_col, y=y_col, hue=hue_col,
                palette=palette, ax=ax
            )
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        # Set title
        if title:
            if plot_type == "pair":
                fig.suptitle(title, y=1.02)
            else:
                ax.set_title(title)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to base64 for returning
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Close the figure to free memory
        plt.close(fig)
        
        return {
            'success': True,
            'plot_type': plot_type,
            'title': title or f"Seaborn {plot_type.title()} Plot",
            'image_base64': image_base64,
            'columns_used': {
                'x': x_col,
                'y': y_col, 
                'hue': hue_col,
                'style': style_col,
                'size': size_col
            },
            'data_shape': df.shape,
            'save_path': save_path
        }
        
    except Exception as e:
        plt.close(fig)  # Make sure to close figure on error
        return {
            'success': False,
            'error': str(e),
            'plot_type': plot_type
        }


def get_seaborn_plot_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Suggest appropriate Seaborn plots based on data characteristics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of plot suggestions with parameters
    """
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    suggestions = []
    
    # Correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 2:
        suggestions.append({
            'plot_type': 'heatmap',
            'description': 'Correlation heatmap of numeric variables',
            'parameters': {}
        })
        
        # Pair plot for exploring relationships
        suggestions.append({
            'plot_type': 'pair',
            'description': 'Pair plot showing relationships between numeric variables',
            'parameters': {
                'hue_col': categorical_cols[0] if categorical_cols else None
            }
        })
    
    # Distribution plots for numeric columns
    for col in numeric_cols[:2]:  # Limit to first 2
        suggestions.append({
            'plot_type': 'histogram',
            'description': f'Distribution of {col}',
            'parameters': {
                'x_col': col,
                'hue_col': categorical_cols[0] if categorical_cols else None
            }
        })
    
    # Box plots for numeric vs categorical
    if numeric_cols and categorical_cols:
        suggestions.append({
            'plot_type': 'box',
            'description': f'Box plot of {numeric_cols[0]} by {categorical_cols[0]}',
            'parameters': {
                'x_col': categorical_cols[0],
                'y_col': numeric_cols[0]
            }
        })
    
    # Count plots for categorical variables
    for col in categorical_cols[:2]:  # Limit to first 2
        suggestions.append({
            'plot_type': 'count',
            'description': f'Count of {col} categories',
            'parameters': {
                'x_col': col
            }
        })
    
    # Scatter plots for numeric pairs
    if len(numeric_cols) >= 2:
        suggestions.append({
            'plot_type': 'scatter',
            'description': f'Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}',
            'parameters': {
                'x_col': numeric_cols[0],
                'y_col': numeric_cols[1],
                'hue_col': categorical_cols[0] if categorical_cols else None
            }
        })
    
    return suggestions


def get_available_seaborn_plots() -> List[str]:
    """
    Get list of available Seaborn plot types.
    
    Returns:
        List of available plot types
    """
    return [
        "scatter", "line", "bar", "histogram", "box", "violin", 
        "heatmap", "pair", "distribution", "regression", "count", 
        "strip", "swarm"
    ]