import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

def split_data(processed_data_path, output_dir):
    """
    Reads the processed data and splits it into train, validation, and test sets.
    The split is 80% train, 10% validation, 10% test.
    Saves them as CSV files in the processed data directory.

    Args:
        processed_data_path (str): Path to the input processed_data.csv file.
        output_dir (str): Directory to save the split files.
    """
    logging.info(f"Reading processed data from {processed_data_path}...")
    try:
        df = pd.read_parquet(processed_data_path)
    except FileNotFoundError:
        logging.error(f"Error: Could not find {processed_data_path}.")
        logging.error("Please ensure you have run the 'ml/preprocess.py' script first.")
        return

    # Ensure the dataframe is not empty
    if df.empty:
        logging.warning("The processed data file is empty. Aborting split.")
        return

    # First split: 80% train, 20% temp (for val/test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']  # Stratify to maintain label distribution
    )

    # Second split: 10% validation, 10% test from the 20% temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label'] # Stratify again
    )

    # Define output paths
    train_path = os.path.join(output_dir, 'train.parquet')
    val_path = os.path.join(output_dir, 'validation.parquet')
    test_path = os.path.join(output_dir, 'test.parquet')

    # Save the files
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logging.info("Data splitting complete.")
    logging.info(f"Training set shape:   {train_df.shape}")
    logging.info(f"Validation set shape: {val_df.shape}")
    logging.info(f"Test set shape:       {test_df.shape}")
    logging.info(f"Files saved to {output_dir}")

def main():
    """Main function to configure logging and run the data split."""
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file_path = os.path.join(LOGS_DIR, 'prepare_data.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    processed_file = os.path.join(PROCESSED_DATA_DIR, 'processed_data.parquet')
    split_data(processed_file, PROCESSED_DATA_DIR)

if __name__ == '__main__':
    main()