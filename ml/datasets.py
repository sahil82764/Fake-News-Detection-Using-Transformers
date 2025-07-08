import torch
from torch.utils.data import Dataset
import pandas as pd

class FakeNewsDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and tokenizing fake news data.
    This class is memory-efficient as it tokenizes data on-the-fly.
    """
    def __init__(self, data_path: str, tokenizer, max_length: int):
        """
        Args:
            data_path (str): Path to the csv file (e.g., train.csv).
            tokenizer: A Hugging Face tokenizer (e.g., DistilBertTokenizer).
            max_length (int): The maximum sequence length for tokenization.
        """
        try:
            self.data = pd.read_parquet(data_path)
        except FileNotFoundError:
            # Provide a more helpful error message
            raise FileNotFoundError(
                f"Data file not found at {data_path}. "
                "Please ensure you have run the preprocessing and data preparation scripts."
            )

        # --- Data Cleaning ---
        self.data = self.data.dropna(subset=['text']).reset_index(drop=True)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches a single data sample and tokenizes it.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            A dictionary containing:
            - 'input_ids': The token IDs.
            - 'attention_mask': The attention mask.
            - 'labels': The label of the sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get text and label from the dataframe
        text = str(self.data.loc[idx, 'text'])
        label = int(self.data.loc[idx, 'label'])

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,   # Pad & truncate all sentences
            padding='max_length',         # Pad to max_length
            truncation=True,              # Truncate to max_length
            return_attention_mask=True,   # Return attention mask
            return_tensors='pt',          # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
