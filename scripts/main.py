import argparse
import torch
import os

from data_prep import generate_stratified_folds, prepare_test_loader
#from train import train_model, save_training_curves
from train_cross_val import train_model_cross_val, save_training_curves
from milesial_unet.unet_model import UNet
from predict import save_predicted_files, save_cm_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='SIAM for Semantic Image Segmentation')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the outputs directory')
    parser.add_argument('--input_type', type=str, default='s2', help='Type of input data: s2, siam_18, siam_33, siam_48, siam_96')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--process_level', type=str, default='l1c', help='Process level of the data: l1c or l2a')
    parser.add_argument('--learn_type', type=str, default='csl', help='Type of learning: cls or ssl')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    #parser.add_argument('--num_channels', type=int, default=10, help='Number of channels in the dataset')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes in the output task')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       #device = torch.device('cpu')
    print(f"Device: {device}")

    # Create output directory for storage heavy files. For smaller outputs, the default out_path is used.
    big_outputs_path = os.path.join(os.path.split(args.out_path)[0], 'model_outputs/')
    if not os.path.exists(big_outputs_path):
        os.makedirs(big_outputs_path, exist_ok=True)

    if args.input_type == 's2':
        num_channels = 10       # Sentinel-2 spectral bands
    elif 'siam' in args.input_type:
        num_channels = 3        # RGB channels
    
    # Initialize model, optimizer, scheduler and loss function
    model = UNet(n_channels=num_channels, n_classes=args.num_classes) # Initialize model    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # Initialize optimizer
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) # Initialize scheduler
    criterion = torch.nn.CrossEntropyLoss()  # Initialize loss function
    
    # Prepare fold generator
    base_args = {'input_dir':args.input_dir, 'process_level':args.process_level, 'learn_type':args.learn_type, 
                      'input_type':args.input_type, 'batch_size':args.batch_size}
    
   
    # Train model
    trained_model, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train_model_cross_val(model = model, 
                                                        generator_func=generate_stratified_folds, generator_args = base_args,
                                                        batch_size =args.batch_size , n_classes=args.num_classes ,optimizer = optimizer, 
                                                        scheduler = scheduler, criterion = criterion, device = device, 
                                                        patience = args.patience, out_path =  big_outputs_path, epochs = args.epochs)
    
    # Save training curves
    save_training_curves(train_loss_history= train_loss_history, train_accuracy_history= train_accuracy_history, 
                         val_loss_history = val_loss_history, val_accuracy_history = val_accuracy_history, out_path = args.out_path)

    #---- Testing script from below ----- 

    # Prepare test loader 
    test_loader = prepare_test_loader(**base_args)
    # Save predicted files
    avg_loss, overall_accuracy, cm, class_names =save_predicted_files(model = trained_model, 
                                        data= test_loader, device = device, out_path = big_outputs_path,
                                        batch_size = args.batch_size, n_classes = args.num_classes, criterion = criterion)
    
    # Save class-wise metrics
    save_cm_metrics(avg_loss = avg_loss, overall_accuracy = overall_accuracy, cm = cm, class_names = class_names, out_path = args.out_path)
    
    print("All done!")

if __name__ == "__main__":
    main()