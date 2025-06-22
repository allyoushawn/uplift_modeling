import pandas as pd
import numpy as np
from pathlib import Path

def analyze_criteo_data(file_path):
    """
    Parse the Criteo CSV file and display column information and data types.
    
    Args:
        file_path (str): Path to the CSV file
    """
    print("Loading Criteo dataset...")
    print("=" * 50)
    
    try:
        # Read the CSV file
        # Using chunksize for large files to get a sample first
        print("Reading file in chunks to analyze structure...")
        
        # First, let's read just the header to see column names
        header_df = pd.read_csv(file_path, nrows=0)
        print(f"Number of columns: {len(header_df.columns)}")
        print(f"Column names: {list(header_df.columns)}")
        print()
        
        # Read a sample of the data to analyze data types
        print("Analyzing data types from sample...")
        # Shuffle and sample 100,000 rows for analysis
        sample_df = pd.read_csv(file_path).sample(n=100000, random_state=42)
        
        # Display column information
        print("Column Information:")
        print("-" * 80)
        print(f"{'Column Name':<30} {'Data Type':<15} {'Non-Null Count':<15} {'Sample Values'}")
        print("-" * 80)
        
        for col in sample_df.columns:
            dtype = str(sample_df[col].dtype)
            non_null_count = sample_df[col].count()
            
            # Get sample values (first few non-null values)
            sample_values = sample_df[col].dropna().head(3).tolist()
            sample_str = str(sample_values)[:50] + "..." if len(str(sample_values)) > 50 else str(sample_values)
            
            print(f"{col:<30} {dtype:<15} {non_null_count:<15} {sample_str}")
        
        print()
        
        # Basic statistics
        print("Basic Statistics:")
        print("-" * 50)
        print(f"Total rows in sample: {len(sample_df)}")
        print(f"Memory usage: {sample_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # New: Distribution analysis
        print("\nDistribution Analysis (mean, P10, P25, P50, P75, P90):")
        print("-" * 80)
        percentiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        for col in sample_df.columns:
            if np.issubdtype(sample_df[col].dtype, np.number):
                mean = sample_df[col].mean()
                p10, p25, p50, p75, p90 = sample_df[col].quantile(percentiles).values
                print(f"{col:<30} mean={mean:.4f}  P10={p10:.4f}  P25={p25:.4f}  P50={p50:.4f}  P75={p75:.4f}  P90={p90:.4f}")
            else:
                print(f"{col:<30} (non-numeric, skipped)")
        print()
        
        # Check for missing values
        print("\nMissing Values Analysis:")
        print("-" * 50)
        missing_data = sample_df.isnull().sum()
        missing_percentage = (missing_data / len(sample_df)) * 100
        
        for col in sample_df.columns:
            if missing_data[col] > 0:
                print(f"{col}: {missing_data[col]} missing values ({missing_percentage[col]:.2f}%)")
            else:
                print(f"{col}: No missing values")
        
        # New: Grouped analysis by treatment
        print("\nGrouped Analysis by Treatment (conversion, visit, exposure):")
        print("-" * 80)
        group_cols = ['conversion', 'visit', 'exposure']
        grouped = sample_df.groupby('treatment')[group_cols].mean()
        print("Mean values for each group:")
        print(grouped)
        print()
        print("Relative Improvement (treatment=1 vs treatment=0) [%]:")
        rel_improvement = ((grouped.loc[1] - grouped.loc[0]) / grouped.loc[0]) * 100
        print(rel_improvement)
        print()
        
        # Try to get total row count (this might take a while for large files)
        print("\nEstimating total file size...")
        try:
            # Count lines in file (rough estimate)
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"Estimated total rows: {line_count - 1}")  # Subtract 1 for header
        except Exception as e:
            print(f"Could not estimate total rows: {e}")
        
        return sample_df
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    """Main function to run the analysis."""
    file_path = "criteo-uplift-v2.1.csv"
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found!")
        return
    
    # Analyze the data
    df = analyze_criteo_data(file_path)
    
    if df is not None:
        print("\n" + "=" * 50)
        print("Analysis complete!")
        print("=" * 50)

if __name__ == "__main__":
    main() 