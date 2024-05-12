import argparse
import torch
import os

import segmentation_models_pytorch as smp
from data_prep import generate_stratified_folds, prepare_test_loader
#from train import train_model, save_training_curves
from train_cross_val import fit_kfolds, save_training_curves
from train_cross_val_ssl import fit_kfolds_ssl, save_training_curves_ssl
from predict import predict
from loss import load_loss


def parse_args():
    parser = argparse.ArgumentParser(description='SIAM for Semantic Image Segmentation')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--input_type', type=str, default='s2', help='Type of input data: s2, siam_18, siam_33, siam_48, siam_96')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--process_level', type=str, default='l1c', help='Process level of the data: l1c or l2a')
    #parser.add_argument('--learn_type', type=str, default='csl', help='Type of learning: cls or ssl')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    #parser.add_argument('--num_channels', type=int, default=10, help='Number of channels in the dataset')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes in the output task')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--omega', type=float, default=0.5, help='Weight for the two tasks in the pretext task')
    parser.add_argument('--gamma_ssl', type = float, default=0.8, help='Gamma for semi-supervised learning rate scheduler')
    parser.add_argument('--loss_ssl_1', type = str, default='DiceLoss', help='Loss function for task 1-segmentation in the pretext task')
    parser.add_argument('--loss_ssl_2', type = str, default='L1Loss', help='Loss function for task 2-reconstruction in the pretext task')
    parser.add_argument('--ssl_type', type=str, required=True, help='One of dual or single_segsiam or single_recon')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the outputs directory')

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

    # ------------------------------------Train pretext task---------------------------------------------
    # ------------------------------------------------------------------------------------------------------
    ssl_config_args = {'gamma_ssl':args.gamma_ssl, 'lr_ssl':0.0005, 'weight_decay_ssl':1e-7, 'epochs_ssl':2, 'patience_ssl':5} #only using gamma_ssl for now

    # Initialize SSL multitask-model UNet with resnet encoder
    ssl_init_args = {'encoder_name':'resnet18', 'in_channels':10, 'classes':49, 
                     'encoder_weights':None, 'activation':None, 'add_reconstruction_head':True}
    ssl_models = [smp.Unet(**ssl_init_args), smp.Unet(**ssl_init_args), smp.Unet(**ssl_init_args)] # Initialize model
    
    optimizers_ssl = [torch.optim.Adam(ssl_models[0].parameters(), lr=args.lr, weight_decay=args.weight_decay),
                        torch.optim.Adam(ssl_models[1].parameters(), lr=args.lr, weight_decay=args.weight_decay),
                        torch.optim.Adam(ssl_models[2].parameters(), lr=args.lr, weight_decay=args.weight_decay)] # Initialize optimizer
    schedulers_ssl = [torch.optim.lr_scheduler.ExponentialLR(optimizers_ssl[0], gamma=ssl_config_args['gamma_ssl']),
                        torch.optim.lr_scheduler.ExponentialLR(optimizers_ssl[1], gamma=ssl_config_args['gamma_ssl']),
                        torch.optim.lr_scheduler.ExponentialLR(optimizers_ssl[2], gamma=ssl_config_args['gamma_ssl'])] # Initialize scheduler
    criterion_ssl_1 = load_loss(loss_name=args.loss_ssl_1)  # segmentation loss
    criterion_ssl_2 = load_loss(loss_name=args.loss_ssl_2)  # reconstruciton loss

    #criterion_ssl_1 = nn.CrossEntorpyLoss()  # Initialize loss function
    #criterion_ssl_2 =  nn.L1Loss() # Initialize loss function
    # -------------------------------------------------------------------------------------------------------

    # Prepare loader arguments
    base_args_ssl = {'input_dir':args.input_dir, 'process_level':args.process_level, 'learn_type':'ssl',
                        'input_type':args.input_type, 'batch_size':args.batch_size}
    
    generator_ssl = generate_stratified_folds(**base_args_ssl)
    
    # Train model
    model_args_ssl = {'batch_size': args.batch_size,'patience': args.patience,'criterion_1': criterion_ssl_1, 'criterion_2': criterion_ssl_2, 
                        'ssl_type': args.ssl_type, 'omega': args.omega, 'device': device,'epochs':args.epochs,'out_paths': (big_outputs_path, args.out_path)}
    print(f"\nSSL Training Begins now...")
    trained_ssl_models, fold_histories_ssl, fold_metrics_ssl, cross_val_metrics_ssl = fit_kfolds_ssl(models=ssl_models, generator=generator_ssl,
                                                                                                    optimizers=optimizers_ssl, schedulers=schedulers_ssl, 
                                                                                                    n_splits=3, **model_args_ssl)
    # --------------------------------------------------------------------------------------------------------
    # Print results
    print("Best epoch metrics for each fold: ", fold_metrics_ssl)
    print("Cross-validation metrics (Averaged over folds): ", cross_val_metrics_ssl)

    # Save training curves
    save_training_curves_ssl(fold_histories=fold_histories_ssl, fold_metrics=fold_metrics_ssl,
                             cross_val_metrics=cross_val_metrics_ssl, out_path= args.out_path)
    
    # ------------------------------------Weights Transfer---------------------------------------------------
    # --------------------------------------------------------------------------------------------------------
    # average the encoder weights of the trained models
    avg_model_ssl = smp.Unet(**ssl_init_args)
    for avg_layer, layer_1, layer_2, layer_3 in zip(avg_model_ssl.encoder.parameters(), 
                                                    trained_ssl_models[0].encoder.parameters(), 
                                                    trained_ssl_models[1].encoder.parameters(), 
                                                    trained_ssl_models[2].encoder.parameters()):
        avg_layer.data = (layer_1.data + layer_2.data + layer_3.data) / 3
    
    # DELETING TRAINED MODELS TO FREE UP MEMORY
    del trained_ssl_models

    if args.input_type == 's2':
        num_channels = 10       # Sentinel-2 spectral bands
    elif 'siam' in args.input_type:
        num_channels = 3        # RGB channels
    
    # Transfer learning from SSL resnet encoder to new instance of UNet
    csl_init_args = {'encoder_name':'resnet18', 'in_channels':num_channels, 'classes':args.num_classes, 
                     'encoder_weights':None, 'activation':None, 'add_reconstruction_head':False}
    models = [smp.Unet(**csl_init_args), smp.Unet(**csl_init_args), smp.Unet(**csl_init_args)] # Initialize model

    models[0].encoder = avg_model_ssl.encoder
    models[1].encoder = avg_model_ssl.encoder
    models[2].encoder = avg_model_ssl.encoder
    
    # Freeze the encoder weights
    for param in models[0].encoder.parameters():
        param.requires_grad = False
    for param in models[1].encoder.parameters():
        param.requires_grad = False
    for param in models[2].encoder.parameters():
        param.requires_grad = False
    # -------------------------------------Train downstream segmentation task---------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------

    # Initialize model, optimizer, scheduler and loss function
    optimizers = [torch.optim.Adam(models[0].parameters(), lr=args.lr, weight_decay=args.weight_decay),
                  torch.optim.Adam(models[1].parameters(), lr=args.lr, weight_decay=args.weight_decay),
                  torch.optim.Adam(models[2].parameters(), lr=args.lr, weight_decay=args.weight_decay)]
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=args.gamma),
                  torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=args.gamma),
                  torch.optim.lr_scheduler.ExponentialLR(optimizers[2], gamma=args.gamma)] 
    #criterion = torch.nn.CrossEntropyLoss()  
    criterion = load_loss(loss_name=args.loss_ssl_1)  # Initialize loss function - keeping it same as segmentation task in pretext task

    # Prepare loader arguments - same for train folds and test set
    base_args = {'input_dir':args.input_dir, 'process_level':args.process_level, 'learn_type':'csl', 
                      'input_type':args.input_type, 'batch_size':args.batch_size}
    generator = generate_stratified_folds(**base_args)

    # Train models
    model_args = {'batch_size': args.batch_size,'patience': args.patience,'criterion': criterion, 
                  'device': device,'epochs':args.epochs,'out_paths': (big_outputs_path, args.out_path)}
    print(f"\nDownstream Segmentation Training Begins now...")
    trained_models, fold_histories, fold_metrics, cross_val_metrics = fit_kfolds(models=models, generator=generator,optimizers=optimizers,
                                                                                schedulers=schedulers, n_splits=3, **model_args)
    
    # Print results
    print("Best epoch metrics for each fold: ", fold_metrics)
    print("Cross-validation metrics (Averaged over folds): ", cross_val_metrics)

    # Save training curves
    save_training_curves(fold_histories=fold_histories, fold_metrics=fold_metrics,
                         cross_val_metrics=cross_val_metrics, out_path= args.out_path)
    
    #-------------------------------------Evaluate on test set---------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    # Prepare test loader 
    test_loader = prepare_test_loader(**base_args)
    
    # Save prediction files and all DW calcs: confusion matrices, class metrics, losses and accuracies
    final_test_metrics =predict(trained_models = trained_models, data=test_loader, 
                                batch_size=args.batch_size, n_classes=args.num_classes, 
                                criterion=criterion, device=device, out_paths=(big_outputs_path, args.out_path))

    print("The performance of the ensemble model on the test set is as follows:")
    print(final_test_metrics)
    print("All done!")

if __name__ == "__main__":
    main()