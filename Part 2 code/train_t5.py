import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
# import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)  # Fixed from 1e-1
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f't5_{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    # Fixed naming convention
    experiment_name = args.experiment_name
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{experiment_name}_dev.pkl')
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            # wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            print(f"  ✓ New best Record F1: {best_f1:.4f}")
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if args.patience_epochs > 0 and epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping after {epoch+1} epochs (patience={args.patience_epochs})")
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    '''
    Train for one epoch.
    
    Returns:
        Average loss per token
    '''
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Forward pass
        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        # Compute loss (only on non-padding tokens)
        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        # Track loss
        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens

        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluation loop during training.
    
    Computes:
        - Cross-entropy loss
        - Generated SQL queries
        - Metrics: SQL EM, Record EM, Record F1
        - Error rate
    
    Returns:
        eval_loss, record_f1, record_em, sql_em, error_rate
    '''
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    all_pred_sql = []

    # Get tokenizer
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # Generation config
    # Note: SQL queries can be very long (avg 98 tokens, max 258 tokens)
    gen_config = GenerationConfig(
        max_length=512,  # Use max_length instead of max_new_tokens for longer sequences
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=PAD_IDX,
    )

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            encoder_input, encoder_mask, decoder_input, decoder_targets, _ = batch

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # Compute loss (teacher forcing)
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )
            logits = outputs["logits"]

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=gen_config,
            )
            
            # Decode to strings
            pred_sql_batch = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            pred_sql_batch = [s.strip() for s in pred_sql_batch]
            all_pred_sql.extend(pred_sql_batch)

    # Average loss
    eval_loss = total_loss / max(total_tokens, 1)

    # Save predictions and compute records
    save_queries_and_records(all_pred_sql, model_sql_path, model_record_path)

    # Compute metrics
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    # Error rate
    num_errors = sum(1 for msg in model_error_msgs if msg is not None and msg != "")
    error_rate = num_errors / max(len(model_error_msgs), 1)

    return eval_loss, record_f1, record_em, sql_em, error_rate

        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Run inference on test set.
    
    Generates SQL queries and saves them along with database records.
    '''
    model.eval()
    all_pred_sql = []

    # Get tokenizer
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # Generation config (same as eval)
    gen_config = GenerationConfig(
        max_length=512,  # Use max_length for longer SQL sequences
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=PAD_IDX,
    )

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            encoder_input = batch[0].to(DEVICE)
            encoder_mask = batch[1].to(DEVICE)

            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                generation_config=gen_config,
            )
            
            pred_sql_batch = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            pred_sql_batch = [s.strip() for s in pred_sql_batch]
            all_pred_sql.extend(pred_sql_batch)

    # Save queries and records
    save_queries_and_records(all_pred_sql, model_sql_path, model_record_path)
    print(f"✓ Saved test predictions to {model_sql_path}")


def main():
    # Get arguments
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    # Load data and model
    print("Loading data...")
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    
    print("Initializing model...")
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    print(f"\nTraining configuration:")
    print(f"  Model type: {'Fine-tune' if args.finetune else 'From scratch'}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.max_n_epochs}")
    print(f"  Patience: {args.patience_epochs}")
    print(f"  Device: {DEVICE}\n")

    # Train (skip if max_n_epochs == 0)
    if args.max_n_epochs > 0:
        print("Starting training...")
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        print("\nTraining complete!")
        
        # Load best model
        print("Loading best model...")
        model = load_model_from_checkpoint(args, best=True)
    else:
        print("Skipping training (max_n_epochs=0)")
        args.checkpoint_dir = os.path.join('checkpoints', f't5_{"ft" if args.finetune else "scr"}_experiments', args.experiment_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        save_model(args.checkpoint_dir, model, best=True)
    
    model.eval()
    
    # Evaluate on dev set
    print("\nEvaluating on dev set...")
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{experiment_name}_dev.pkl')
    
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    
    print(f"\n{'='*70}")
    print(f"DEV SET RESULTS")
    print(f"{'='*70}")
    print(f"Loss: {dev_loss:.4f}")
    print(f"Record F1: {dev_record_f1:.4f}")
    print(f"Record EM: {dev_record_em:.4f}")
    print(f"SQL EM: {dev_sql_em:.4f}")
    print(f"Error rate: {dev_error_rate*100:.2f}%")
    print(f"{'='*70}\n")

    # Run inference on test set
    print("Running inference on test set...")
    model_sql_path = os.path.join('results', f't5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    
    print(f"\n✓ All done!")
    print(f"✓ Dev results saved to: {model_sql_path.replace('test', 'dev')}")
    print(f"✓ Test results saved to: {model_sql_path}")

if __name__ == "__main__":
    main()