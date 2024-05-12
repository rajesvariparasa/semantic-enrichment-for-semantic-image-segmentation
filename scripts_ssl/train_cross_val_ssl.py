import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import segmentation_models_pytorch as smp


def load_best_model(path):
    ssl_init_args = {'encoder_name':'resnet18', 'in_channels':10, 'classes':49, 
                     'encoder_weights':None, 'activation':None, 'add_reconstruction_head':True}
    model = smp.Unet(**ssl_init_args)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

    
# ------- Training functions for semi-supervised learning ------- #
def train_epoch_ssl(model, data, optimizer, ssl_type, criterion_t1,criterion_t2, omega, device):
    model.train()
    running_loss = 0.0
    running_loss_t1 = 0.0
    running_loss_t2 = 0.0
    #correct_preds = 0
    num_batches = len(data)

    for _,batch in enumerate(tqdm(data, desc='Training', leave=False)): # for each batch
        #print(f"Batch {i}")
        features, labels,_ = batch
        
        siam_band = 3                                  # band number 0,1,2,3,4, = scl, siam 18,33,48,96 - decide based on baseline experiments
        features, labels = features.to(device), labels[:,siam_band,:,:].to(device)
        optimizer.zero_grad()

        if ssl_type == 'dual':                              
            outputs_t1, outputs_t2 = model(features)   #task 1 - siam prediction and task 2 - predict input bands back
            loss_t1 = criterion_t1(outputs_t1, labels) 
            loss_t2 = criterion_t2(outputs_t2, features) 
            loss = (omega* loss_t1) + (1-omega)*loss_t2
        
        elif ssl_type == 'single_segsiam':                                    
            outputs_t1, _ = model(features)      #task 1 - siam prediction
            loss_t1 = criterion_t1(outputs_t1, labels) 
            loss_t2 = torch.Tensor([0])
            loss = loss_t1                      
        
        elif ssl_type == 'single_recon':
            _, outputs_t2 = model(features)
            loss_t1 = torch.Tensor([0])
            loss_t2 = criterion_t2(outputs_t2, features) #task 2 - predict input bands back
            loss = loss_t2
        
        loss.backward()
        optimizer.step()

        # loss and number of correct predictions of the batch
        running_loss += loss.item() # loss per batch
        running_loss_t1 += loss_t1.item()
        running_loss_t2 += loss_t2.item()


        #preds= torch.argmax(outputs, dim=1)
        #correct_preds += torch.sum(preds == labels).item()
    
    avg_loss = running_loss / num_batches
    avg_loss_t1 = running_loss_t1 / num_batches
    avg_loss_t2 = running_loss_t2 / num_batches
    #overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, avg_loss_t1, avg_loss_t2 #, overall_accuracy 

def validate_epoch_ssl(model, data, ssl_type, criterion_t1, criterion_t2, omega, device):
    model.eval()
    running_loss = 0.0
    running_loss_t1 = 0.0
    running_loss_t2 = 0.0
    #correct_preds = 0
    num_batches = len(data)
 
    with torch.no_grad():
        for _,batch in enumerate(tqdm(data, desc='Validation', leave=False)):
            features, labels,_ = batch
            
            siam_band = 2                                  # band number 0,1,2,3,4, = scl, siam 18,33,48,96 - decide based on baseline experiments
            features, labels = features.to(device), labels[:,siam_band,:,:].to(device)

            if ssl_type == 'dual':                              
                outputs_t1, outputs_t2 = model(features)   #task 1 - siam prediction and task 2 - predict input bands back
                loss_t1 = criterion_t1(outputs_t1, labels) 
                loss_t2 = criterion_t2(outputs_t2, features) 
                loss = (omega* loss_t1) + (1-omega)*loss_t2
            
            elif ssl_type == 'single_segsiam':                                    
                outputs_t1, _ = model(features)      #task 1 - siam prediction
                loss_t1 = criterion_t1(outputs_t1, labels) 
                loss_t2 = torch.Tensor([0])
                loss = loss_t1                      
            
            elif ssl_type == 'single_recon':
                _, outputs_t2 = model(features)
                loss_t2 = criterion_t2(outputs_t2, features) #task 2 - predict input bands back
                loss_t1 = torch.Tensor([0])
                loss = loss_t2
                
            # loss and number of correct predictions of the batch
            running_loss += loss.item()
            running_loss_t1 += loss_t1.item()
            running_loss_t2 += loss_t2.item()
            # preds = torch.argmax(outputs, dim=1)
            # correct_preds += torch.sum(preds == labels).item()

    avg_loss = running_loss / num_batches
    avg_loss_t1 = running_loss_t1 / num_batches
    avg_loss_t2 = running_loss_t2 / num_batches
    # overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, avg_loss_t1,avg_loss_t2  #, overall_accuracy

def train_model_ssl(model, fold, train_data, val_data, ssl_type, batch_size, optimizer, scheduler, criterion_1,criterion_2, omega, device, patience,out_paths, epochs):
    start = time.time()
    model = model.to(device)

    min_val_loss = np.inf
    train_loss_history = []
    val_loss_history = []
    model_out_path = os.path.join(out_paths[0], f'fold_{fold}_best_model_ssl.pth')
    patience_counter = 0

    for epoch in range(epochs):

        epoch_args = {'model':model, 'ssl_type':ssl_type, 'criterion_t1':criterion_1, 
                      'criterion_t2':criterion_2, 'omega':omega, 'device':device}
        
        train_epoch_loss,train_epoch_loss_t1, train_epoch_loss_t2  = train_epoch_ssl(optimizer=optimizer, data=train_data, **epoch_args)
        val_epoch_loss,val_epoch_loss_t1, val_epoch_loss_t2 = validate_epoch_ssl(data=val_data, **epoch_args)

        print(f"\nEpoch {epoch+1}/{epochs} => Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f},\n Train Loss T1: {train_epoch_loss_t1:.4f}, Val Loss T1: {val_epoch_loss_t1:.4f}, Train Loss T2: {train_epoch_loss_t2:.4f}, Val Loss T2: {val_epoch_loss_t2:.4f}")
    
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)

        if scheduler is not None:
            scheduler.step()
        
        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            best_epoch_metrics= {'train_loss':train_epoch_loss,
                                 'val_loss':val_epoch_loss}
            
            ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'loss': min_val_loss
                }
            torch.save(ckpt, model_out_path)
            print("Model saved") # saves the last best model, overwrites the previous best one
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_duration = time.time() - start
    print(f"Training completed for fold {fold+1} in {train_duration//60:.0f}m {train_duration % 60:.0f}s\n")

    model_tracking = {'train_loss_history':train_loss_history, 
                      'val_loss_history':val_loss_history}
    
    trained_model = load_best_model(model_out_path)
    return trained_model, model_tracking, best_epoch_metrics, train_duration

def fit_kfolds_ssl(models,generator, optimizers, schedulers, n_splits, **model_args):
    start = time.time()
    sum_train_loss = 0.0
    sum_val_loss = 0.0

    fold_histories ={}
    fold_metrics = {}
    trained_models = []
    fold_durations = []
   
    for k, (train_data, val_data) in tqdm(enumerate(generator), desc='Folds', leave=False):

        
        model, model_tracking, best_epoch_metrics, fold_duration = train_model_ssl(model=models[k], optimizer=optimizers[k],
                                                                scheduler = schedulers[k], fold=k,
                                                                train_data=train_data, val_data=val_data,
                                                                **model_args)
        
        fold_durations.append(fold_duration)

        sum_train_loss += best_epoch_metrics['train_loss']
        sum_val_loss += best_epoch_metrics['val_loss']

        fold_metrics[f'fold_{k}'] = best_epoch_metrics
        fold_histories[f'fold_{k}'] = model_tracking
        trained_models.append(model)
    
    total_duration = time.time() - start
    print(f"Training completed for ALL all SSL folds in {total_duration//60:.0f}m {total_duration % 60:.0f}s \n")

    # save training durations for each fold and total in csv
    durations = fold_durations
    durations.append(total_duration)
    durations = [duration//60 for duration in durations]
    print("List durations" , durations)
    duration_names = ['fold_1', 'fold_2', 'fold_3', 'total']
    durations_df = pd.DataFrame(durations, index=duration_names, columns=['duration'])
    durations_df.to_csv(os.path.join(model_args['out_paths'][1], 'durations_ssl.csv'), index=True)

    cross_val_metrics = {'train_loss':sum_train_loss/n_splits,
                            'val_loss':sum_val_loss/n_splits}
    
    # fold_metrics are the best metrics for each fold
    # cross_val_metrics are the average metrics for all folds    
    return trained_models, fold_histories, fold_metrics, cross_val_metrics

def save_training_curves_ssl(fold_histories, fold_metrics, cross_val_metrics, out_path):
    # plot training and validation losses and accuracies of all three folds. One plot for losses and one for accuracies side by side
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    colors = ['b', 'g', 'r']
    for i, (fold, history) in enumerate(fold_histories.items()):
        axs.plot(history['train_loss_history'], label=f'Fold {i+1} Train', color=colors[i])
        axs.plot(history['val_loss_history'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')

    axs.set_title('Training and Validation Losses')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.legend()

    # on x axis, as only integers epoch values are shown. As many as possible
    plt.xticks(np.arange(0, max([len(history['train_loss_history']) for history in fold_histories.values()]), step = 5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'training_curves_ssl.png'))
    plt.close()

    #save fold_metrics in csv
    metrics = pd.DataFrame(fold_metrics)
    metrics.to_csv(os.path.join(out_path, 'train_fold_metrics_ssl.csv'), index=False)

    #save cross_val_metrics in csv
    metrics = pd.DataFrame(cross_val_metrics, index=[0])
    metrics.to_csv(os.path.join(out_path, 'train_cross_val_metrics_ssl.csv'), index=False)

