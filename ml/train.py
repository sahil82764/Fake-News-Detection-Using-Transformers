import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
import os
import numpy as np
import logging

# Import configurations and the custom Dataset class
import config
from datasets import FakeNewsDataset

def compute_metrics(preds, labels):
    """Computes and returns accuracy, precision, recall, and F1-score."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_epoch(model, data_loader, optimizer, device, scheduler, scaler):
    """Performs one full training pass over the data."""
    model.train()
    total_loss = 0
    
    # Using tqdm for a progress bar
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for i, batch in enumerate(progress_bar):
        # Move batch to the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Use autocast for mixed-precision training (FP16)
        # This reduces memory usage and can speed up training on compatible GPUs
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        # Normalize loss for gradient accumulation
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation: update weights every N steps
        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            # Update the scale for next iteration
            scaler.update()
            # Update learning rate
            scheduler.step()
            # Clear gradients
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(data_loader.dataset)

def evaluate_model(model, data_loader, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations for evaluation
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    return avg_loss, metrics

def main():
    """Main function to run the training and evaluation pipeline."""
    # --- Setup Logging ---
    log_dir = os.path.join(config.PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'train.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Using device: {config.DEVICE}")

    # --- 1. Load Tokenizer and Model ---
    logging.info(f"Loading tokenizer and model: {config.MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
    model.to(config.DEVICE)

    # --- 2. Load Datasets and DataLoaders ---
    logging.info("Loading datasets...")
    train_dataset = FakeNewsDataset(
        data_path=os.path.join(config.DATA_DIR, 'train.parquet'),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    val_dataset = FakeNewsDataset(
        data_path=os.path.join(config.DATA_DIR, 'validation.parquet'),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )

    # Set num_workers=0 for better compatibility, especially on Windows
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=0)

    # --- 3. Setup Optimizer, Scheduler, and Scaler ---
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = (len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    # Gradient scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(config.DEVICE == "cuda"))

    # --- 4. Training Loop ---
    best_val_f1 = 0
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    for epoch in range(config.EPOCHS):
        logging.info(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, config.DEVICE, scheduler, scaler)
        logging.info(f"Average Training Loss: {train_loss:.4f}")
        
        val_loss, val_metrics = evaluate_model(model, val_loader, config.DEVICE)
        logging.info(f"Validation Loss: {val_loss:.4f}")
        logging.info(f"Validation Metrics: {val_metrics}")

        # --- 5. Save the best model (checkpointing) ---
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            logging.info(f"New best F1 score: {best_val_f1:.4f}. Saving model...")
            model.save_pretrained(config.MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)
            # Save a config file with the model for easy loading later
            torch.save(model.config, os.path.join(config.MODEL_OUTPUT_DIR, 'config.bin'))

    logging.info("\nTraining complete!")
    logging.info(f"Best validation F1 score: {best_val_f1:.4f}")
    logging.info(f"Model saved to {config.MODEL_OUTPUT_DIR}")

if __name__ == '__main__':
    main()
