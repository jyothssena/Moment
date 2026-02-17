'''
import sys
sys.path.append("/Users/jyothssena/Moment/") 

from data_pipeline.scripts.data_acquisition import DataAcquisition

acquisition = DataAcquisition(config_path="data_pipeline/config/config.yaml")
metadata = acquisition.run()
dataframes_dict = acquisition.get_dataframes()

for filename, df in dataframes_dict.items():
    print(f"\n📄 {filename}:")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   First few rows:")
    print(df.head(3))
    print()
'''
"""
Test script to run data acquisition without Airflow DAG
Each file is kept as a separate DataFrame
Run this file directly: python test_acquisition.py
"""
import sys
sys.path.append("/Users/jyothssena/Moment/") 
import logging
from data_pipeline.scripts.data_acquisition import DataAcquisition

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_acquisition():
    """Test the data acquisition function."""
    
    print("\n" + "="*60)
    print("Testing GCP Data Acquisition (Individual DataFrames)")
    print("="*60 + "\n")
    
    try:
        # Path to your config file
        config_path = "data_pipeline/config/config.yaml"
        
        print(f"📂 Loading config from: {config_path}\n")
        
        # Initialize acquisition
        acquisition = DataAcquisition(config_path=config_path)
        
        # Run acquisition
        print("🚀 Starting data acquisition...\n")
        metadata = acquisition.run()
        
        # Get dictionary of DataFrames
        dataframes_dict = acquisition.get_dataframes()
        
        # Display results
        print("\n" + "="*60)
        print("✅ ACQUISITION SUCCESSFUL!")
        print("="*60 + "\n")
        
        print("📊 Metadata:")
        print(f"  - Timestamp: {metadata['timestamp']}")
        print(f"  - Number of files: {metadata['num_files']}")
        print(f"  - Total rows: {metadata['total_rows']}")
        print(f"  - Source bucket: {metadata['source_bucket']}")
        print(f"  - Prefix: {metadata['prefix']}")
        print(f"  - Files: {metadata['files']}")
        
        print("\n" + "="*60)
        print("📋 Individual DataFrames:")
        print("="*60 + "\n")
        
        # Show details for each DataFrame
        for filename, df in dataframes_dict.items():
            print(f"📄 File: {filename}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            print(f"   Missing values: {df.isnull().sum().sum()}")
            print(f"\n   First 3 rows:")
            print(df.head(3).to_string(index=False))
            print("\n" + "-"*60 + "\n")
        
        print("\n💡 Usage Examples:")
        print("-"*60)
        print("\n# Get all DataFrames:")
        print("dataframes_dict = acquisition.get_dataframes()")
        print("\n# Get specific DataFrame:")
        print(f"df = acquisition.get_dataframe('{metadata['files'][0]}')")
        print("\n# Iterate through all DataFrames:")
        print("for filename, df in dataframes_dict.items():")
        print("    print(f'{filename}: {len(df)} rows')")
        
        return dataframes_dict
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR OCCURRED!")
        print("="*60)
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your config.yaml file exists")
        print("2. Verify GCP credentials are set")
        print("3. Confirm bucket name and prefix are correct")
        print("4. Make sure you have read permissions")
        raise


def test_specific_file():
    """Test accessing a specific file."""
    
    print("\n" + "="*60)
    print("Testing Access to Specific File")
    print("="*60 + "\n")
    
    config_path = "data_pipeline/config/config.yaml"
    
    acquisition = DataAcquisition(config_path=config_path)
    metadata = acquisition.run()
    
    # Get list of available files
    available_files = metadata['files']
    print(f"Available files: {available_files}\n")
    
    # Access first file
    first_file = available_files[0]
    df = acquisition.get_dataframe(first_file)
    
    print(f"✅ Accessed '{first_file}':")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n{df.head()}")
    
    return df


def test_process_each_file():
    """Example: Process each file separately."""
    
    print("\n" + "="*60)
    print("Example: Processing Each File Separately")
    print("="*60 + "\n")
    
    config_path = "data_pipeline/config/config.yaml"
    
    acquisition = DataAcquisition(config_path=config_path)
    acquisition.run()
    
    dataframes_dict = acquisition.get_dataframes()
    
    # Process each file individually
    for filename, df in dataframes_dict.items():
        print(f"\nProcessing {filename}...")
        
        # Example processing
        print(f"  Original rows: {len(df)}")
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        print(f"  After removing duplicates: {len(df_clean)}")
        
        # Remove missing values
        df_clean = df_clean.dropna()
        print(f"  After removing missing: {len(df_clean)}")
        
        # You can save each processed file separately
        # df_clean.to_csv(f'processed_{filename}', index=False)
        
    print("\n✅ All files processed!")


if __name__ == "__main__":
    """
    Run this script directly:
    $ python test_acquisition.py
    """
    
    print("\n" + "="*60)
    print("GCP Data Acquisition Test Script")
    print("="*60)
    
    print("\nAvailable test modes:")
    print("1. Full test (show all DataFrames)")
    print("2. Test specific file access")
    print("3. Test processing each file separately")
    print("4. Exit")
    
    choice = input("\nSelect mode (1/2/3/4): ")
    
    if choice == "1":
        # Run standard test
        dataframes_dict = test_acquisition()
        
        # Optional: Save results
        save = input("\n💾 Save all DataFrames to CSV? (yes/no): ")
        if save.lower() == 'yes':
            for filename, df in dataframes_dict.items():
                output_file = f"test_{filename}"
                df.to_csv(output_file, index=False)
                print(f"✅ Saved to {output_file}")
    
    elif choice == "2":
        # Test specific file access
        df = test_specific_file()
    
    elif choice == "3":
        # Test processing each file
        test_process_each_file()
    
    else:
        print("Exiting...")