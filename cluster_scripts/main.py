import argparse
import torch
import os

from data_prep import prepare_loaders
from train import train_model, save_training_curves
from milesial_unet.unet_model import UNet

def parse_args():
    parser = argparse.ArgumentParser(description='SIAM for Semantic Image Segmentation')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the outputs directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--process_level', type=str, default='l1c', help='Process level of the data: l1c or l2a')
    parser.add_argument('--learn_type', type=str, default='csl', help='Type of learning: cls or ssl')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--num_channels', type=int, default=13, help='Number of channels in the dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the output task')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    #parser.add_argument('--use_checkpoint', action='store_true', help='Flag to resume training from a checkpoint')
    #parser.add_argument('--model', type=str, default='unet_modified', help='Model to use for training')
    #parser.add_argument('--loss_func', type=str, default='DiceLoss', help='Loss function for training')
    #parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training')
    #parser.add_argument('--scheduler', type=str, default=None, help='Scheduler for learning rate adjustment')
    #parser.add_argument('--gamma', type=float, default=0.95, help='Gamma for ExponentialLR scheduler')
    return parser.parse_args()

def main():
    args = parse_args()
    BATCH_SIZE = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare input diretory and data loaders
    input_dir = os.path.join(args.input_dir,args.process_level) # currently l1c,l2c directories within data/small/
    train_loader, val_loader = prepare_loaders(input_dir=input_dir, 
                                               process_level = args.process_level, 
                                               learn_type = args.learn_type, batch_size= BATCH_SIZE)
    
    # Prepare output directory
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Initialize model, optimizer, scheduler and loss function
    model = UNet(in_channels=args.num_channels, out_channels=args.num_classes) # Initialize model    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # Initialize optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()  # Initialize loss function
    
    # Train model
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train_model(model = model, 
                                                        trin_data=train_loader, val_data = val_loader, optimizer = optimizer, 
                                                        scheduler = scheduler, criterion = criterion, device = device, 
                                                        patience = args.patience, out_path =  args.out_path, epochs = args.epochs)
    
    # Save training curves
    save_training_curves(train_loss_history= train_loss_history, train_accuracy_history= train_accuracy_history, 
                         val_loss_history = val_loss_history, val_accuracy_history = val_accuracy_history, out_path = args.out_path)

if __name__ == "__main__":
    main()