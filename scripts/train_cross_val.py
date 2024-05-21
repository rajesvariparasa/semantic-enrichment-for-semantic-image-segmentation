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
    
def train_epoch(model, data, batch_size, optimizer, criterion, metrics, device):
    model.train()
    running_loss = 0.0
    # correct_preds = 0   
    num_batches = len(data)

    running_metrics = {
        'iou': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
    }    

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
        # correct_preds += torch.sum(preds == labels).item()

        tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=11)
        running_metrics['iou'] += metrics['IoUScore'](tp, fp, fn, tn).item()
        running_metrics['f1'] += metrics['F1Score'](tp, fp, fn, tn).item()
        running_metrics['accuracy'] += metrics['Accuracy'](tp, fp, fn, tn).item()

    avg_loss = running_loss / num_batches
    avg_running_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    #overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, avg_running_metrics 

def validate_epoch(model, data,batch_size, criterion, metrics, device):
    model.eval()
    running_loss = 0.0
    num_batches = len(data)

    running_metrics = {
        'iou': 0,
        'f1': 0,
        'accuracy': 0,
    }   

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
            # correct_preds += torch.sum(preds == labels).item()
            tp, fp, fn, tn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=11)
            running_metrics['iou'] += metrics['IoUScore'](tp, fp, fn, tn).item()
            running_metrics['f1'] += metrics['F1Score'](tp, fp, fn, tn).item()
            running_metrics['accuracy'] += metrics['Accuracy'](tp, fp, fn, tn).item()

    avg_loss = running_loss / num_batches
    #overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    avg_running_metrics = {k: v / num_batches for k, v in running_metrics.items()}
    return avg_loss, avg_running_metrics


def train_model(model, fold, train_data, val_data, batch_size, optimizer, scheduler, criterion, metrics, device, patience,out_paths, epochs, csl_init_args):
    start = time.time()
    model = model.to(device)

    min_val_loss = np.inf
    train_loss_history = []
    train_metrics = {'iou':[], 'f1':[], 'accuracy':[]}

    val_loss_history = []
    val_metrics = {'iou':[], 'f1':[], 'accuracy':[]}

    model_out_path = os.path.join(out_paths[0], f'fold_{fold}_best_model.pth')
    patience_counter = 0

    for epoch in range(epochs):
        train_epoch_loss, train_epoch_metrics = train_epoch(model=model, data=train_data, batch_size=batch_size, optimizer=optimizer, criterion=criterion, metrics = metrics, device=device)
        val_epoch_loss, val_epoch_metrics  = validate_epoch(model=model, data=val_data, batch_size=batch_size, criterion=criterion, metrics=metrics, device=device)
        #print(f"\nEpoch {epoch+1}/{epochs} => Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f} , Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")
        print(f"\nEpoch {epoch+1}/{epochs} => Train Loss: {train_epoch_loss:.4f}, Train IoU: {train_epoch_metrics['iou']:.4f}, " 
              f"Train F1: {train_epoch_metrics['f1']:.4f}, Train Accuracy: {train_epoch_metrics['accuracy']:.4f} , \nVal Loss: {val_epoch_loss:.4f}, "
              f"Val IoU: {val_epoch_metrics['iou']:.4f}, Val F1: {val_epoch_metrics['f1']:.4f}, Val Accuracy: {val_epoch_metrics['accuracy']:.4f}")

        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)

        for key in train_epoch_metrics.keys():
            train_metrics[key].append(train_epoch_metrics[key])
            val_metrics[key].append(val_epoch_metrics[key])

        if scheduler is not None:
            scheduler.step()
        
        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            best_epoch_metrics= {'train_loss':train_epoch_loss, 'train_accuracy':train_epoch_metrics['accuracy'], 'train_iou':train_epoch_metrics['iou'], 'train_f1':train_epoch_metrics['f1'],  # 'train_accuracy':train_epoch_accuracy,
                                 'val_loss':val_epoch_loss, 'val_accuracy':val_epoch_metrics['accuracy'], 'val_iou':val_epoch_metrics['iou'], 'val_f1':val_epoch_metrics['f1']} # 'val_accuracy':val_epoch_accuracy
            ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,

                    'train_loss_history': train_loss_history,
                    'train_iou_history': train_metrics['iou'],
                    'train_f1_history': train_metrics['f1'],
                    'train_accuracy_history': train_metrics['accuracy'],

                    'val_loss_history': val_loss_history,
                    'val_iou_history': val_metrics['iou'],
                    'val_f1_history': val_metrics['f1'],
                    'val_accuracy_history': val_metrics['accuracy'],

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

    model_tracking = {'train_loss_history':train_loss_history, 'train_iou_history':train_metrics['iou'], 'train_f1_history':train_metrics['f1'], 'train_accuracy_history':train_metrics['accuracy'],
                      'val_loss_history':val_loss_history, 'val_iou_history':val_metrics['iou'], 'val_f1_history':val_metrics['f1'], 'val_accuracy_history':val_metrics['accuracy']}
    
    trained_model = load_best_model(model_out_path, **csl_init_args)
    return trained_model, model_tracking, best_epoch_metrics, train_duration

def fit_kfolds(models,generator, optimizers, schedulers, n_splits, csl_init_args, **model_args):
    start = time.time()
    sum_train_loss, sum_val_loss = 0.0, 0.0
    sum_metrics = {'sum_train_accuracy':0.0, 'sum_train_iou':0.0, 'sum_train_f1':0.0,
                    'sum_val_accuracy':0.0, 'sum_val_iou':0.0, 'sum_val_f1':0.0}

    foldwise_histories ={}
    foldwise_best_epoch_metrics = {}
    foldwise_durations = []

    trained_models = []
    
   
    for k, (train_data, val_data) in tqdm(enumerate(generator), desc='Folds', leave=False):

        
        model, model_tracking, best_epoch_metrics, fold_duration = train_model(model=models[k], optimizer=optimizers[k],
                                                                scheduler = schedulers[k], fold=k,
                                                                train_data=train_data, val_data=val_data, 
                                                                csl_init_args=csl_init_args, **model_args)
        
        foldwise_durations.append(fold_duration)

        # losses
        sum_train_loss += best_epoch_metrics['train_loss']
        sum_val_loss += best_epoch_metrics['val_loss']
        # metrics
        sum_metrics['sum_train_accuracy'] += best_epoch_metrics['train_accuracy']
        sum_metrics['sum_train_iou'] += best_epoch_metrics['train_iou']
        sum_metrics['sum_train_f1'] += best_epoch_metrics['train_f1']
        sum_metrics['sum_val_accuracy'] += best_epoch_metrics['val_accuracy']
        sum_metrics['sum_val_iou'] += best_epoch_metrics['val_iou']
        sum_metrics['sum_val_f1'] += best_epoch_metrics['val_f1']

        foldwise_histories[f'fold_{k}'] = model_tracking
        foldwise_best_epoch_metrics[f'fold_{k}'] = best_epoch_metrics
        trained_models.append(model)
    
    total_duration = time.time() - start

    cross_val_metrics = {}
    cross_val_metrics['train_loss'] = sum_train_loss / n_splits  # insert losses to the metrics dict
    cross_val_metrics['val_loss'] = sum_val_loss / n_splits

    # average metrics over all folds
    cross_val_metrics['train_accuracy'] = sum_metrics['sum_train_accuracy'] / n_splits
    cross_val_metrics['train_iou'] = sum_metrics['sum_train_iou'] / n_splits
    cross_val_metrics['train_f1'] = sum_metrics['sum_train_f1'] / n_splits
    cross_val_metrics['val_accuracy'] = sum_metrics['sum_val_accuracy'] / n_splits
    cross_val_metrics['val_iou'] = sum_metrics['sum_val_iou'] / n_splits
    cross_val_metrics['val_f1'] = sum_metrics['sum_val_f1'] / n_splits

    print(f"\nTraining completed for ALL all folds in {total_duration//60:.0f}m {total_duration % 60:.0f}s \n")

    # save training durations for each fold and total in csv
    durations = foldwise_durations
    durations.append(total_duration)
    durations = [duration//60 for duration in durations]  #in minutes
    print("List durations" , durations)
    durations_df = pd.DataFrame(durations, index=['fold_1', 'fold_2', 'fold_3', 'total'], columns=['duration'])
    durations_df.to_csv(os.path.join(model_args['out_paths'][1], 'durations.csv'), index=True)

    # cross_val_metrics are the average metrics for all folds    
    return trained_models, foldwise_histories, foldwise_best_epoch_metrics, cross_val_metrics


#-------------Training Metrics and Curves-------------#

# new function to save training curves based on the new tracking metrics iou and f1 added 

def save_training_curves(foldwise_histories, foldwise_best_epoch_metrics, cross_val_metrics, out_path):
    fig, axs = plt.subplots(2,2, figsize=(15, 5))
    colors = ['b', 'g', 'r']
    for i, (fold, history) in enumerate(foldwise_histories.items()):
        axs[0,0].plot(history['train_loss_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[0,0].plot(history['val_loss_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])

        axs[0,1].plot(history['train_iou_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[0,1].plot(history['val_iou_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])

        axs[1,0].plot(history['train_f1_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[1,0].plot(history['val_f1_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])

        axs[1,1].plot(history['train_accuracy_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[1,1].plot(history['val_accuracy_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])

    axs[0,0].set_title('Dice Loss', fontsize=15)
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_ylabel('Loss')
    axs[0,0].legend()

    axs[0,1].set_title('IoU', fontsize=15)
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_ylabel('IoU')
    axs[0,1].legend()

    axs[1,0].set_title('F1 Score', fontsize=15)
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_ylabel('F1 Score')
    axs[1,0].legend()

    axs[1,1].set_title('Accuracy', fontsize=15)
    axs[1,1].set_xlabel('Epoch')
    axs[1,1].set_ylabel('Accuracy')
    axs[1,1].legend()

    max_x_ticks = max([len(history['train_loss_history']) for history in foldwise_histories.values()])
    for ax in axs.flatten():
        ax.set_xticks(np.arange(0, max_x_ticks, step = 5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'training_curves.png'), dpi=300)
    plt.close()

    #save fold_metrics in csv
    index= list(foldwise_best_epoch_metrics['fold_1'].keys())  # get the keys from the first fold
    metrics = pd.DataFrame(foldwise_best_epoch_metrics, index=index)
    metrics.to_csv(os.path.join(out_path, 'train_fold_metrics.csv'), index=True, index_label='metric', float_format='%.4f')

    #save cross_val_metrics in csv

    metrics = pd.DataFrame(cross_val_metrics,  index=[0])
    metrics.to_csv(os.path.join(out_path, 'train_cross_val_metrics.csv'), index=False, float_format='%.4f')
