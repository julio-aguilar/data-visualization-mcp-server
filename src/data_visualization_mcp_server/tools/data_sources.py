"""
Data sources for visualization tools
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, Optional, Any, List
import seaborn as sns


def get_sample_data(dataset_name: str = "tips") -> pd.DataFrame:
    """
    Get sample datasets for visualization.
    
    Args:
        dataset_name: Name of the dataset to load. Options include:
            - 'tips': Restaurant tips dataset
            - 'iris': Iris flower dataset  
            - 'flights': Flight passenger data
            - 'car_crashes': Car crash statistics
            - 'titanic': Titanic passenger data
            - 'custom': Generate custom synthetic data
    
    Returns:
        DataFrame containing the requested sample data
    """
    
    if dataset_name == "custom":
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'value': np.random.normal(50, 15, n_samples),
            'size': np.random.uniform(10, 100, n_samples),
            'date': pd.date_range('2023-01-01', periods=n_samples, freq='D')
        }
        
        df = pd.DataFrame(data)
        df['value'] = df['value'].clip(0, 100)  # Ensure positive values
        
        return df
    
    try:
        # Use seaborn's built-in datasets
        return sns.load_dataset(dataset_name)
    except Exception as e:
        # Fallback to tips dataset if requested dataset is not available
        print(f"Dataset '{dataset_name}' not available, falling back to 'tips': {e}")
        return sns.load_dataset("tips")


def fetch_api_data(api_type: str = "jsonplaceholder") -> pd.DataFrame:
    """
    Fetch sample data from public APIs for visualization.
    
    Args:
        api_type: Type of API to fetch from. Options:
            - 'jsonplaceholder': Sample posts/users data
            - 'httpbin': HTTP testing data
            - 'random_user': Random user generator
    
    Returns:
        DataFrame containing the API data
    """
    
    try:
        if api_type == "jsonplaceholder":
            # Fetch posts data
            posts_response = requests.get("https://jsonplaceholder.typicode.com/posts", timeout=10)
            posts_data = posts_response.json()
            
            # Fetch users data
            users_response = requests.get("https://jsonplaceholder.typicode.com/users", timeout=10)
            users_data = users_response.json()
            
            # Create DataFrames
            posts_df = pd.DataFrame(posts_data)
            users_df = pd.DataFrame(users_data)
            
            # Merge posts with user info
            merged_df = posts_df.merge(
                users_df[['id', 'name', 'username', 'email', 'company']], 
                left_on='userId', 
                right_on='id', 
                suffixes=('_post', '_user')
            )
            
            # Add some derived metrics
            merged_df['title_length'] = merged_df['title'].str.len()
            merged_df['body_length'] = merged_df['body'].str.len()
            merged_df['company_name'] = merged_df['company'].apply(
                lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
            )
            
            return merged_df[['userId', 'id_post', 'title', 'title_length', 'body_length', 
                           'name', 'username', 'email', 'company_name']]
            
        elif api_type == "httpbin":
            # Generate some sample HTTP-related data
            response = requests.get("https://httpbin.org/json", timeout=10)
            base_data = response.json()
            
            # Create synthetic HTTP metrics
            np.random.seed(42)
            n_samples = 50
            
            methods = ['GET', 'POST', 'PUT', 'DELETE']
            status_codes = [200, 201, 400, 404, 500]
            
            data = {
                'method': np.random.choice(methods, n_samples),
                'status_code': np.random.choice(status_codes, n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
                'response_time': np.random.lognormal(4, 1, n_samples),  # Log-normal for realistic response times
                'size_bytes': np.random.exponential(1000, n_samples),
                'endpoint': [f"/api/v1/resource/{i}" for i in range(n_samples)]
            }
            
            return pd.DataFrame(data)
            
        elif api_type == "random_user":
            # Fetch random users for demographics visualization
            response = requests.get("https://randomuser.me/api/?results=100", timeout=10)
            data = response.json()
            
            users = []
            for user in data['results']:
                users.append({
                    'gender': user['gender'],
                    'age': user['dob']['age'],
                    'country': user['location']['country'],
                    'city': user['location']['city'], 
                    'email': user['email'],
                    'phone': user['phone'],
                    'registered_year': int(user['registered']['date'][:4])
                })
            
            return pd.DataFrame(users)
            
        else:
            raise ValueError(f"Unknown API type: {api_type}")
            
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        # Return fallback synthetic data
        return get_sample_data("custom")
    except Exception as e:
        print(f"Error processing API data: {e}")
        # Return fallback synthetic data
        return get_sample_data("custom")


def get_available_datasets() -> List[str]:
    """
    Get list of available dataset names.
    
    Returns:
        List of available dataset names
    """
    return [
        "tips", "iris", "flights", "car_crashes", "titanic", 
        "mpg", "diamonds", "fmri", "custom"
    ]


def get_available_apis() -> List[str]:
    """
    Get list of available API data sources.
    
    Returns:
        List of available API types
    """
    return ["jsonplaceholder", "httpbin", "random_user"]


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about a DataFrame for better visualization choices.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'datetime_columns': datetime_cols,
        'missing_values': {col: int(count) for col, count in df.isnull().sum().to_dict().items()},
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
    }