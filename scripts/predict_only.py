# a script to predict the test set and save the confusion matrix and metrics when path to the trained model is provided

import os 
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

from milesial_unet.unet_model import UNet
from data_prep import prepare_test_loader
from train_cross_val import save_curves
from predict import save_predicted_files, save_cm_metrics

def parse_args():
    parser = ArgumentParser(description='SIAM for Semantic Image Segmentation')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the outputs directory')
    parser.add_argument('--input_type', type=str, default='s2', help='Type of input data: s2, siam_18, siam_33, siam_48, siam_96')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--process_level', type=str, default='l1c', help='Process level of the data: l1c or l2a')
    parser.add_argument('--learn_type', type=str, default='csl', help='Type of learning: cls or ssl')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes in the output task')
    parser.add_argument('--generate_curves', type=bool, default=False, help='Generate training curves')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare output directory
    big_outputs_path = os.path.join(os.path.split(os.path.split(args.out_path)[0])[0], 'model_outputs/')
    if not os.path.exists(big_outputs_path):
        os.makedirs(big_outputs_path, exist_ok=True)

    # Prepare test loader
    test_loader = prepare_test_loader(input_dir=args.input_dir, process_level=args.process_level, 
                                      learn_type=args.learn_type, input_type=args.input_type, batch_size=args.batch_size)
    print(f"Test data: {len(test_loader.dataset)} samples")
    print(f"Number of batches in test: {len(test_loader)}")

    # Load the trained model
    if args.input_type == 's2':
        num_channels = 10       # Sentinel-2 spectral bands
    elif 'siam' in args.input_type:
        num_channels = 3        # RGB channels

    model = UNet(n_channels=num_channels, n_classes=args.num_classes)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    criterion = torch.nn.CrossEntropyLoss()  # Initialize loss function

    # Load metrics history for training curves
    train_loss_history=  ckpt['train_loss_history']
    train_accuracy_history= ckpt['train_accuracy_history']
    val_loss_history = ckpt['val_loss_history'] 
    val_accuracy_history= ckpt['val_accuracy_history']  
    folds_train_loss_history =ckpt['folds_train_loss_history']                
    folds_train_accuracy_history= ckpt['folds_train_accuracy_history']
    folds_val_loss_history=ckpt['folds_val_loss_history']
    folds_val_accuracy_history=ckpt['folds_val_accuracy_history']

    # Generate training curves
    if args.generate_curves:
        save_curves(epoch_metrics = (train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history), 
                    folds_metrics = (folds_train_loss_history, folds_train_accuracy_history, folds_val_loss_history, folds_val_accuracy_history), 
                    out_path = args.out_path)
        print("Training curves saved!")

    model.to(device)
    model.eval()

    # Predict the test set
    avg_loss, overall_accuracy, cm, class_names = save_predicted_files(model=model, data=test_loader, batch_size=args.batch_size, 
                                                                       n_classes=args.num_classes, criterion=criterion, 
                                                                       device=device, out_path=big_outputs_path)
    
    # Save confusion matrix and class-wise metrics
    save_cm_metrics(avg_loss, overall_accuracy, cm, class_names, out_path=args.out_path)
    print("Predictions saved!")

if __name__ == '__main__':
    main()