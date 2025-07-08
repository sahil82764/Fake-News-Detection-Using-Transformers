import pandas as pd
import os
import logging

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
KAGGLE_DIR = os.path.join(RAW_DATA_DIR, 'kaggle_fake_news')
LIAR_DIR = os.path.join(RAW_DATA_DIR, 'liar')

def _data_loader_wrapper(dataset_name, error_message):
    """A decorator to handle common data loading boilerplate like logging and error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"Processing {dataset_name} dataset...")
            try:
                df = func(*args, **kwargs)
                if not df.empty:
                    logging.info(f"{dataset_name} dataset processed. Shape: {df.shape}")
                return df
            except FileNotFoundError:
                logging.exception(error_message)
                return pd.DataFrame()
        return wrapper
    return decorator

@_data_loader_wrapper("Kaggle", "Error loading Kaggle files. Please ensure Fake.csv and True.csv are in data/raw/kaggle_fake_news/")
def load_and_process_kaggle():
    """
    Loads and processes the Kaggle Fake and Real News dataset.
    - Combines title and text.
    - Assigns binary labels.
    """
    fake_df = pd.read_csv(os.path.join(KAGGLE_DIR, 'Fake.csv'))
    true_df = pd.read_csv(os.path.join(KAGGLE_DIR, 'True.csv'))

    # Assign labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Combine title and text
    fake_df['text'] = fake_df['title'] + '. ' + fake_df['text']
    true_df['text'] = true_df['title'] + '. ' + true_df['text']

    # Combine into one dataframe
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Select and rename columns for consistency
    return combined_df[['text', 'label']]

@_data_loader_wrapper("LIAR", "Error loading LIAR files. Please ensure train.tsv, test.tsv, and valid.tsv are in data/raw/liar/")
def load_and_process_liar():
    """
    Loads and processes the LIAR dataset.
    - Maps multi-class labels to binary.
    - Discards ambiguous 'half-true' labels.
    """
    # Column names for the LIAR dataset
    liar_columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
        'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]
    
    # Load all parts of the dataset
    train_df = pd.read_csv(os.path.join(LIAR_DIR, 'train.tsv'), sep='\t', names=liar_columns)
    test_df = pd.read_csv(os.path.join(LIAR_DIR, 'test.tsv'), sep='\t', names=liar_columns)
    valid_df = pd.read_csv(os.path.join(LIAR_DIR, 'valid.tsv'), sep='\t', names=liar_columns)

    combined_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)

    # Define the label mapping
    label_mapping = {
        'true': 1,
        'mostly-true': 1,
        'false': 0,
        'barely-true': 0,
        'pants-on-fire': 0
    }

    # Filter out 'half-true' and apply mapping
    combined_df = combined_df[combined_df['label'] != 'half-true']
    combined_df['label'] = combined_df['label'].map(label_mapping)
    
    # Rename 'statement' to 'text' for consistency
    combined_df.rename(columns={'statement': 'text'}, inplace=True)
    
    # Select final columns
    return combined_df[['text', 'label']]

def main():
    """Main function to run all preprocessing steps."""
    # Create directories for logs and processed data
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Configure logging to write to a file and to the console
    log_file_path = os.path.join(LOGS_DIR, 'preprocess.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
            logging.StreamHandler()
        ]
    )

    logging.info("Starting data harmonization...")

    kaggle_data = load_and_process_kaggle()
    liar_data = load_and_process_liar()

    if kaggle_data.empty and liar_data.empty:
        logging.warning("No data processed. Exiting.")
        return

    # Combine the two processed datasets
    final_data = pd.concat([kaggle_data, liar_data], ignore_index=True)
    logging.info(f"Combined dataset shape before cleaning: {final_data.shape}")

    # Basic cleaning
    final_data.dropna(subset=['text', 'label'], inplace=True)
    final_data.drop_duplicates(subset=['text'], inplace=True)
    
    # Ensure label is integer type
    final_data['label'] = final_data['label'].astype(int)

    # Shuffle the dataset
    final_data = final_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.parquet')
    final_data.to_parquet(output_path, index=False)
    
    logging.info(f"Final cleaned dataset shape: {final_data.shape}")
    logging.info(f"Label distribution (0=Fake, 1=Real):\n{final_data['label'].value_counts(normalize=True)}")
    logging.info(f"Successfully saved processed data to {output_path} (Parquet format)")


if __name__ == '__main__':
    # To run this script, navigate to the 'ml' directory
    # and execute 'python preprocess.py' in your terminal.
    main()
