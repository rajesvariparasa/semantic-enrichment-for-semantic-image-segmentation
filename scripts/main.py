import argparse
import torch
import os

#from milesial_unet.unet_model import UNet
import segmentation_models_pytorch as smp
from data_prep import generate_stratified_folds, prepare_test_loader
#from train import train_model, save_training_curves
from train_cross_val import fit_kfolds, save_training_curves
from predict import predict
from losses_metrics import load_loss, load_metric

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
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--encoder_name', type=str, default='resnet18', help='Encoder name for the model')
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       #device = torch.device('cpu')
    print(f"Device: {device}")

    # Create output directory for storage heavy files. For smaller outputs, the default out_path is used.
    big_outputs_path = os.path.join(os.path.split(os.path.split(args.out_path)[0])[0], 'model_outputs/')
    if not os.path.exists(big_outputs_path):
        os.makedirs(big_outputs_path, exist_ok=True)

    if 's2_siam' in args.input_type:
        num_channels = 13       # Sentinel-2 spectral bands + RGB channels
    elif args.input_type == 's2':
        num_channels = 10       # Sentinel-2 spectral bands
    elif args.input_type.split('_')[0] == 'siam':   
        num_channels = 3        # RGB channels
    else:
        raise ValueError("Invalid input type. Please choose from: s2, siam_18, siam_33, siam_48, siam_96 or s2_siam_18, s2_siam_33, s2_siam_48, s2_siam_96.")

    # Initialize model, optimizer, scheduler and loss function
    # models = [UNet(n_channels=num_channels, n_classes=args.num_classes),
    #           UNet(n_channels=num_channels, n_classes=args.num_classes),
    #           UNet(n_channels=num_channels, n_classes=args.num_classes)] # Initialize model -milesial unet
    
    csl_init_args = {'encoder_name':args.encoder_name, 'in_channels':num_channels, 'classes':args.num_classes, 
                     'encoder_weights':None, 'activation':None, 'add_reconstruction_head':False}
    models = [smp.Unet(**csl_init_args), smp.Unet(**csl_init_args), smp.Unet(**csl_init_args)] # Initialize model -smp unet

    optimizers = [torch.optim.Adam(models[0].parameters(), lr=args.lr, weight_decay=args.weight_decay),
                  torch.optim.Adam(models[1].parameters(), lr=args.lr, weight_decay=args.weight_decay),
                  torch.optim.Adam(models[2].parameters(), lr=args.lr, weight_decay=args.weight_decay)] # Initialize optimizer
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=args.gamma),
                  torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=args.gamma),
                  torch.optim.lr_scheduler.ExponentialLR(optimizers[2], gamma=args.gamma)] # Initialize scheduler
    #criterion = torch.nn.CrossEntropyLoss()  # Initialize loss function
    criterion = load_loss(loss_name='DiceLoss')  # Initialize loss function
    metrics = {'Accuracy': load_metric(metric_name='Accuracy'),
               'IoUScore': load_metric(metric_name = 'IoUScore'),
               'F1Score': load_metric(metric_name = 'F1Score')} # Initialize metrics

    # Prepare loader arguments - same for train folds and test set
    base_args = {'input_dir':args.input_dir, 'process_level':args.process_level, 'learn_type':args.learn_type, 
                      'input_type':args.input_type, 'batch_size':args.batch_size}
   
    generator = generate_stratified_folds(**base_args)

    # Train models
    model_args = {'batch_size': args.batch_size,'patience': args.patience,'criterion': criterion, 'metrics': metrics,
                  'device': device,'epochs':args.epochs,'out_paths': (big_outputs_path, args.out_path)}
    
    trained_models, best_epoch_nums,foldwise_histories, foldwise_best_epoch_metrics, cross_val_metrics = fit_kfolds(models=models, generator=generator,optimizers=optimizers,
                                                                                schedulers=schedulers, n_splits=3, csl_init_args=csl_init_args, **model_args)
    
    # Print results
    print("Best epoch metrics for each fold: ", foldwise_best_epoch_metrics, "\n")
    print("Cross-validation metrics (Averaged over folds): ", cross_val_metrics, "\n")

    # Save training curves
    save_training_curves(best_epoch_nums=best_epoch_nums, foldwise_histories=foldwise_histories, foldwise_best_epoch_metrics=foldwise_best_epoch_metrics,
                         cross_val_metrics=cross_val_metrics, out_path= args.out_path)
    
    #---- Testing script from below ----- 

    # Prepare test loader 
    test_loader = prepare_test_loader(**base_args)
    
    # Save prediction files and all DW calcs: confusion matrices, class metrics, losses and accuracies
    final_test_metrics =predict(trained_models = trained_models, data=test_loader, 
                                batch_size=args.batch_size, n_classes=args.num_classes, 
                                criterion=criterion,metrics=metrics ,device=device, out_paths=(big_outputs_path, args.out_path))

    print("The performance of the ensemble model on the test set is as follows:")
    print(final_test_metrics)
    print("All done!")

if __name__ == "__main__":
    main()