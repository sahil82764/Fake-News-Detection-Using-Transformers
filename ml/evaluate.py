import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm.auto import tqdm
import os
import numpy as np
import logging

# Import configurations and the custom Dataset class
import config
from datasets import FakeNewsDataset # Assuming datasets.py is in the same directory

def evaluate_on_test_set(model, data_loader, device):
    """
    Evaluates the final model on the test set and returns predictions and labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Testing", leave=True)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def main():
    """
    Main function to load the trained model and evaluate it on the test set.
    """
    # --- Setup Logging ---
    log_dir = os.path.join(config.PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'evaluate.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Using device: {config.DEVICE}")

    # --- 1. Load Trained Model and Tokenizer ---
    logging.info(f"Loading trained model from: {config.MODEL_OUTPUT_DIR}")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_OUTPUT_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_OUTPUT_DIR)
        model.to(config.DEVICE)
    except OSError:
        logging.error(f"Could not find a trained model in {config.MODEL_OUTPUT_DIR}.")
        logging.error("Please run the training script (train.py) first.")
        return

    # --- 2. Load Test Dataset ---
    logging.info("Loading test dataset...")
    test_data_path = os.path.join(config.DATA_DIR, 'test.parquet')
    try:
        test_dataset = FakeNewsDataset(
            data_path=test_data_path,
            tokenizer=tokenizer,
            max_length=config.MAX_LENGTH
        )
    except FileNotFoundError:
        logging.error(f"Test data file not found at {test_data_path}")
        return

    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 3. Evaluate and Report Metrics ---
    predictions, true_labels = evaluate_on_test_set(model, test_loader, config.DEVICE)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    accuracy = accuracy_score(true_labels, predictions)

    logging.info("\n--- Final Model Evaluation on Test Set ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    
    # --- 4. Display Detailed Classification Report ---
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=['Fake (0)', 'Real (1)']
    )
    logging.info("\nClassification Report:\n")
    # Print the report directly to the console for better formatting
    print(report)
    
    # Also log it to the file
    for line in report.split('\n'):
        logging.info(line)


if __name__ == '__main__':
    main()
