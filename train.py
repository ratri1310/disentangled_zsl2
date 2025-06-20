#!/usr/bin/env python3

import os
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import pandas as pd

from data_preprocessing import load_and_preprocess_data, BiomedicalDataset
from main_model import DisentangledBiomedicalClassifier, ModelTrainer
from zero_shot_classifier import ZeroShotEvaluator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Disentangled Biomedical Classifier')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data CSV file')
    parser.add_argument('--mesh_path', type=str, required=True,
                       help='Path to MeSH definitions CSV file')
    parser.add_argument('--seen_classes_file', type=str, required=True,
                       help='File containing seen MeSH class codes')
    parser.add_argument('--unseen_classes_file', type=str, required=True,
                       help='File containing unseen MeSH class codes')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                       help='Pre-trained model name')
    parser.add_argument('--d1', type=int, default=512, help='First hidden dimension')
    parser.add_argument('--d2', type=int, default=256, help='Second hidden dimension')
    parser.add_argument('--d3', type=int, default=256, help='Third hidden dimension')
    parser.add_argument('--d4', type=int, default=128, help='Final content dimension')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent variance dimension')
    
    # Loss function arguments
    parser.add_argument('--lambda_elbo', type=float, default=1.0, help='ELBO loss weight')
    parser.add_argument('--lambda_classification', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--lambda_contrastive', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--lambda_similarity', type=float, default=0.3, help='Similarity loss weight')
    parser.add_argument('--temperature', type=float, default=0.07, help='Contrastive learning temperature')
    parser.add_argument('--beta_vae', type=float, default=1.0, help='Beta-VAE parameter')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Validation arguments
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Model save interval (epochs)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--experiment_name', type=str, default='biomedical_disentanglement',
                       help='Experiment name for logging')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    return parser.parse_args()

def load_class_lists(seen_file: str, unseen_file: str):
    """Load seen and unseen class lists from files."""
    with open(seen_file, 'r') as f:
        seen_classes = [line.strip() for line in f if line.strip()]
    
    with open(unseen_file, 'r') as f:
        unseen_classes = [line.strip() for line in f if line.strip()]
    
    return seen_classes, unseen_classes

def setup_device(device_arg: str):
    """Setup device for training."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    return device

def create_data_loaders(args, seen_classes, unseen_classes):
    """Create training and validation data loaders."""
    print("Loading and preprocessing data...")
    
    # Load dataset
    dataset, mesh_data = load_and_preprocess_data(args.data_path, args.mesh_path)
    
    # Filter dataset to only include seen classes for training
    # This would require modifying the dataset to only include relevant labels
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    return train_loader, val_loader, mesh_data

def setup_model_and_optimizer(args, seen_classes, unseen_classes, device):
    """Setup model and optimizer."""
    print("Initializing model...")
    
    model = DisentangledBiomedicalClassifier(
        model_name=args.model_name,
        seen_mesh_codes=seen_classes,
        unseen_mesh_codes=unseen_classes,
        d1=args.d1, d2=args.d2, d3=args.d3, d4=args.d4,
        latent_dim=args.latent_dim,
        lambda_elbo=args.lambda_elbo,
        lambda_classification=args.lambda_classification,
        lambda_contrastive=args.lambda_contrastive,
        lambda_similarity=args.lambda_similarity,
        temperature=args.temperature,
        beta_vae=args.beta_vae
    )
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    
    return model, optimizer, scheduler

def train_model(args):
    """Main training function."""
    # Setup
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs', args.experiment_name))
    
    # Load class lists
    seen_classes, unseen_classes = load_class_lists(args.seen_classes_file, args.unseen_classes_file)
    print(f"Seen classes: {len(seen_classes)}, Unseen classes: {len(unseen_classes)}")
    
    # Create data loaders
    train_loader, val_loader, mesh_data = create_data_loaders(args, seen_classes, unseen_classes)
    
    # Setup model and optimizer
    model, optimizer, scheduler = setup_model_and_optimizer(args, seen_classes, unseen_classes, device)
    
    # Initialize MeSH anchors
    print("Initializing MeSH anchor embeddings...")
    model.initialize_mesh_anchors(mesh_data)
    
    # Setup trainer
    trainer = ModelTrainer(model, device)
    
    # Training loop
    best_val_loss = float('inf')
    best_f1 = 0.0
    global_step = 0
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Training
        train_metrics = trainer.train_epoch(train_loader, optimizer, epoch, args.log_interval)
        
        # Log training metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'train/{key}', value, epoch)
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        
        # Validation
        if (epoch + 1) % (args.eval_interval // len(train_loader)) == 0:
            print("Evaluating on validation set...")
            val_metrics = trainer.evaluate(val_loader, mode='seen')
            
            # Log validation metrics
            for key, value in val_metrics.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            
            print(f"Val Loss: {val_metrics['eval_loss']:.4f}")
            if 'f1_@0.5' in val_metrics:
                print(f"Val F1@0.5: {val_metrics['f1_@0.5']:.4f}")
            
            # Save best model
            if val_metrics['eval_loss'] < best_val_loss:
                best_val_loss = val_metrics['eval_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'args': vars(args)
                }, os.path.join(args.output_dir, 'best_model.pt'))
                print("Saved best model!")
            
            # Save model based on F1 score
            if 'f1_@0.5' in val_metrics and val_metrics['f1_@0.5'] > best_f1:
                best_f1 = val_metrics['f1_@0.5']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1_score': best_f1,
                    'args': vars(args)
                }, os.path.join(args.output_dir, 'best_f1_model.pt'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
        
        # Update scheduler
        scheduler.step()
        
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('train/learning_rate', current_lr, epoch)
    
    # Final model save
    model.save_model(os.path.join(args.output_dir, 'final_model.pt'))
    
    print("Training completed!")
    writer.close()

def zero_shot_evaluation(args):
    """Evaluate model on zero-shot task."""
    device = setup_device(args.device)
    
    # Load model
    model_path = os.path.join(args.output_dir, 'best_f1_model.pt')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load class lists
    seen_classes, unseen_classes = load_class_lists(args.seen_classes_file, args.unseen_classes_file)
    
    # Initialize model
    model = DisentangledBiomedicalClassifier(
        seen_mesh_codes=seen_classes,
        unseen_mesh_codes=unseen_classes,
        **{k: v for k, v in checkpoint['args'].items() if k in [
            'model_name', 'd1', 'd2', 'd3', 'd4', 'latent_dim',
            'lambda_elbo', 'lambda_classification', 'lambda_contrastive',
            'lambda_similarity', 'temperature', 'beta_vae'
        ]}
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load test data (would need unseen class test set)
    # This is a placeholder - you'd need to implement loading test data for unseen classes
    print("Zero-shot evaluation would be implemented here...")
    print("This requires test data with unseen class labels.")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Save configuration
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train model
    train_model(args)
    
    # Optionally run zero-shot evaluation
    print("\nTo run zero-shot evaluation, implement the test data loading...")
