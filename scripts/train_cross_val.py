import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import segmentation_models_pytorch as smp

def load_best_model(model_out_path, **csl_init_args):
    model = smp.Unet(**csl_init_args) # Initialize model
    ckpt = torch.load(model_out_path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model
    
def train_epoch(model, data, batch_size, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    num_batches = len(data)
    for _,batch in enumerate(tqdm(data, desc='Training', leave=False)): # for each batch
        #print(f"Batch {i}")
        features, labels,_ = batch
        
        dw_band = 5 # 6th band is the DW band
        features, labels = features.to(device), labels[:,dw_band,:,:].to(device)
        optimizer.zero_grad()
        outputs = model(features)
        labels = labels.long()

        loss = criterion(outputs, labels) # loss
        loss.backward()
        optimizer.step()

        # loss and number of correct predictions of the batch
        running_loss += loss.item() # loss per batch
        preds= torch.argmax(outputs, dim=1)
        correct_preds += torch.sum(preds == labels).item()
    
    avg_loss = running_loss / num_batches
    overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, overall_accuracy 

def validate_epoch(model, data,batch_size, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    num_batches = len(data)
    with torch.no_grad():
        for _,batch in enumerate(tqdm(data, desc='Validation', leave=False)):
            features, labels,_ = batch

            dw_band = 5 # 6th band is the DW band
            features, labels = features.to(device), labels[:,dw_band,:,:].to(device)
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            labels = labels.long()
            loss = criterion(outputs, labels)
            
            # loss and number of correct predictions of the batch
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(preds == labels).item()

    avg_loss = running_loss / num_batches
    overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, overall_accuracy


def train_model(model, fold, train_data, val_data, batch_size, optimizer, scheduler, criterion, device, patience,out_paths, epochs, csl_init_args):
    start = time.time()
    model = model.to(device)

    min_val_loss = np.inf
    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    model_out_path = os.path.join(out_paths[0], f'fold_{fold}_best_model.pth')
    patience_counter = 0

    for epoch in range(epochs):
        train_epoch_loss, train_epoch_accuracy = train_epoch(model, train_data, batch_size, optimizer, criterion, device)
        val_epoch_loss, val_epoch_accuracy = validate_epoch(model, val_data, batch_size, criterion, device)
        print(f"\nEpoch {epoch+1}/{epochs} => Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f} , Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")
    
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        train_accuracy_history.append(train_epoch_accuracy)
        val_accuracy_history.append(val_epoch_accuracy)

        if scheduler is not None:
            scheduler.step()
        
        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            best_epoch_metrics= {'train_loss':train_epoch_loss, 'train_accuracy':train_epoch_accuracy,
                                 'val_loss':val_epoch_loss, 'val_accuracy':val_epoch_accuracy}
            
            ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'train_loss_history': train_loss_history,
                    'train_accuracy_history': train_accuracy_history,
                    'val_loss_history': val_loss_history,
                    'val_accuracy_history': val_accuracy_history,
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
    print(f"Training completed for fold {fold} in {train_duration//60:.0f}m {train_duration % 60:.0f}s")

    model_tracking = {'train_loss_history':train_loss_history, 'train_accuracy_history':train_accuracy_history,
                      'val_loss_history':val_loss_history, 'val_accuracy_history':val_accuracy_history}
    
    trained_model = load_best_model(model_out_path, **csl_init_args)
    return trained_model, model_tracking, best_epoch_metrics, train_duration

def fit_kfolds(models,generator, optimizers, schedulers, n_splits, csl_init_args, **model_args):
    start = time.time()
    sum_train_loss, sum_train_accuracy = 0.0, 0.0
    sum_val_loss, sum_val_accuracy = 0.0, 0.0

    fold_histories ={}
    fold_metrics = {}
    trained_models = []
    fold_durations = []
   
    for k, (train_data, val_data) in tqdm(enumerate(generator), desc='Folds', leave=False):

        
        model, model_tracking, best_epoch_metrics, fold_duration = train_model(model=models[k], optimizer=optimizers[k],
                                                                scheduler = schedulers[k], fold=k,
                                                                train_data=train_data, val_data=val_data, 
                                                                csl_init_args=csl_init_args, **model_args)
        
        fold_durations.append(fold_duration)

        sum_train_loss += best_epoch_metrics['train_loss']
        sum_train_accuracy += best_epoch_metrics['train_accuracy']
        sum_val_loss += best_epoch_metrics['val_loss']
        sum_val_accuracy += best_epoch_metrics['val_accuracy']

        fold_metrics[f'fold_{k}'] = best_epoch_metrics
        fold_histories[f'fold_{k}'] = model_tracking
        trained_models.append(model)
    
    total_duration = time.time() - start
    print(f"\nTraining completed for ALL all folds in {total_duration//60:.0f}m {total_duration % 60:.0f}s \n")

    # save training durations for each fold and total in csv
    durations = fold_durations
    durations.append(total_duration)
    #in minutes
    durations = [duration//60 for duration in durations]
    print("List durations" , durations)
    duration_names = ['fold_1', 'fold_2', 'fold_3', 'total']
    durations_df = pd.DataFrame(durations, index=duration_names, columns=['duration'])
    durations_df.to_csv(os.path.join(model_args['out_paths'][1], 'durations.csv'), index=True)

    cross_val_metrics = {'train_loss':sum_train_loss/n_splits, 'train_accuracy':sum_train_accuracy/n_splits,
                            'val_loss':sum_val_loss/n_splits, 'val_accuracy':sum_val_accuracy/n_splits}
    
    # fold_metrics are the best metrics for each fold
    # cross_val_metrics are the average metrics for all folds    
    return trained_models, fold_histories, fold_metrics, cross_val_metrics


#-------------Training Metrics and Curves-------------#

def save_training_curves(fold_histories, fold_metrics, cross_val_metrics, out_path):
    # plot training and validation losses and accuracies of all three folds. One plot for losses and one for accuracies side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['b', 'g', 'r']
    for i, (fold, history) in enumerate(fold_histories.items()):
        axs[0].plot(history['train_loss_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[0].plot(history['val_loss_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])
        axs[1].plot(history['train_accuracy_history'], label=  f"Fold {i+1} Train", color = colors[i])
        axs[1].plot(history['val_accuracy_history'], label=  f"Fold {i+1} Validation", linestyle='--', color = colors[i])

    axs[0].set_title('Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].set_title('Accuracies')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # on x axis, as only integers epoch values are shown. As many as possible
    plt.xticks(np.arange(0, max([len(history['train_loss_history']) for history in fold_histories.values()]), step = 5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'training_curves.png'))
    plt.close()

    #save fold_metrics in csv
    metrics = pd.DataFrame(fold_metrics)
    metrics.to_csv(os.path.join(out_path, 'train_fold_metrics.csv'), index=False)

    #save cross_val_metrics in csv
    metrics = pd.DataFrame(cross_val_metrics, index=[0])
    metrics.to_csv(os.path.join(out_path, 'train_cross_val_metrics.csv'), index=False)        
