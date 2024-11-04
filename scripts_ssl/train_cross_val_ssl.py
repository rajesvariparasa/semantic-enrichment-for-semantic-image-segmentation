import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import segmentation_models_pytorch as smp
from model import UNetWithDropout
from model import SMP_Unet_Multitask_v0
from losses_metrics import MultiTaskLoss

def get_bandnum_classes(band_name):

    if band_name == 'siam_18':
        return {'band_num': 1, 'classes': 19} # 18 classes + 1 for no data - check data prep for clarification
    elif band_name == 'siam_33':
        return {'band_num': 2, 'classes': 34}
    elif band_name == 'siam_48':
        return {'band_num': 3, 'classes': 49}
    elif band_name == 'siam_96':
        return {'band_num': 4, 'classes': 97}
    else:
        raise ValueError(f"Band name {band_name} not implemented")

def load_best_model(path, **ssl_init_args):
   # ssl_init_args = {'encoder_name':'resnet50', 'in_channels':10, 'classes':49, 
   #                  'encoder_weights':None, 'activation':None, 'add_reconstruction_head':True}
    model =SMP_Unet_Multitask_v0(**ssl_init_args)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    best_e = ckpt['epoch']
    return model, best_e

    
# ------- Training functions for semi-supervised learning ------- #
def train_epoch_ssl(model, data, optimizer, ssl_type, criterion_t1,criterion_t2, metric_t1, metric_t2, siam_segment_bandnum,siam_segment_classes, omega, device):
    model.train()
    running_loss = 0.0
    running_loss_t1 = 0.0
    running_loss_t2 = 0.0

    running_metric = 0.0
    running_metric_t1 = 0.0
    running_metric_t2 = 0.0
    #correct_preds = 0
    num_batches = len(data)

    if ssl_type == 'dual':
        multitask_criterion = MultiTaskLoss()

    for _,batch in enumerate(tqdm(data, desc='Training', leave=False)): # for each batch
        #print(f"Batch {i}")
        features, labels,_ = batch
        
        #siam_band = 3                                  # band number 0,1,2,3,4, = scl, siam 18,33,48,96 - decide based on baseline experiments
        features, labels = features.to(device), labels[:,siam_segment_bandnum,:,:].to(device)
        optimizer.zero_grad()

        if ssl_type == 'dual':   
            #print(len(model(features)))                           
            outputs_t1, outputs_t2, log_vars = model(features)   
            #task 1 - siam prediction and task 2 - reflectance reconstruction
            log_var_seg, log_var_rec = log_vars[0], log_vars[1]
            loss, loss_t1, loss_t2 = multitask_criterion(outputs_t1, outputs_t2, labels, features, log_vars, criterion_t1, criterion_t2)

            preds = torch.argmax(outputs_t1, dim=1)
            tp, fp, tn, fn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=siam_segment_classes)
            acc_t1 = metric_t1(tp, fp, fn, tn)
            acc_t2 = metric_t2(outputs_t2.view(-1), features.view(-1))
            acc = acc_t1 + acc_t2
        
        elif ssl_type == 'single_segsiam':                                    
            outputs_t1, _,log_vars = model(features)      #task 1 - siam prediction
            log_var_seg, log_var_rec = log_vars[0], log_vars[1]
            loss_t1 = criterion_t1(outputs_t1, labels) 
            loss_t2 = torch.Tensor([0])
            loss = loss_t1                 

            preds = torch.argmax(outputs_t1, dim=1)
            tp, fp, tn, fn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=siam_segment_classes)
            acc_t1 = metric_t1(tp, fp, fn, tn)
            acc_t2 = torch.Tensor([0])
            acc = acc_t1     
        
        elif ssl_type == 'single_recon':
            _, outputs_t2,log_vars= model(features)
            log_var_seg, log_var_rec = log_vars[0], log_vars[1]
            loss_t1 = torch.Tensor([0])
            loss_t2 = criterion_t2(outputs_t2, features) #task 2 - predict input bands back
            loss = loss_t2

            acc_t1 = torch.Tensor([0])
            acc_t2 = metric_t2(outputs_t2.view(-1), features.view(-1))
            acc = acc_t2

        loss.backward()
        optimizer.step()

        # loss and number of correct predictions of the batch
        running_loss += loss.item() # loss per batch
        running_loss_t1 += loss_t1.item()
        running_loss_t2 += loss_t2.item()

        running_metric += acc.item()
        running_metric_t1 += acc_t1.item()
        running_metric_t2 += acc_t2.item()
        
        #preds= torch.argmax(outputs, dim=1)
        #correct_preds += torch.sum(preds == labels).item()
    
    avg_loss = running_loss / num_batches
    avg_loss_t1 = running_loss_t1 / num_batches
    avg_loss_t2 = running_loss_t2 / num_batches
    #overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)

    avg_metric = running_metric / num_batches
    avg_metric_t1 = running_metric_t1 / num_batches
    avg_metric_t2 = running_metric_t2 / num_batches

    return avg_loss, avg_loss_t1, avg_loss_t2, avg_metric, avg_metric_t1, avg_metric_t2, log_var_seg, log_var_rec

def validate_epoch_ssl(model, data, ssl_type, criterion_t1, criterion_t2,metric_t1, metric_t2, siam_segment_bandnum, siam_segment_classes,  omega, device):
    model.eval()
   
    running_loss = 0.0
    running_loss_t1 = 0.0
    running_loss_t2 = 0.0

    running_metric = 0.0
    running_metric_t1 = 0.0
    running_metric_t2 = 0.0

    #correct_preds = 0
    num_batches = len(data)
    if ssl_type == 'dual':
        multitask_criterion = MultiTaskLoss()

 
    with torch.no_grad():
        for _,batch in enumerate(tqdm(data, desc='Validation', leave=False)):
            features, labels,_ = batch
            
            #siam_band = 2                                  # band number 0,1,2,3,4, = scl, siam 18,33,48,96 - decide based on baseline experiments
            features, labels = features.to(device), labels[:,siam_segment_bandnum,:,:].to(device)

            if ssl_type == 'dual':                              
                outputs_t1, outputs_t2, log_vars = model(features)   
                #task 1 - siam prediction and task 2 - reflectance reconstruction
                #log_var_seg, log_var_rec = log_vars[0], log_vars[1]
                loss, loss_t1, loss_t2 = multitask_criterion(outputs_t1, outputs_t2, labels, features, log_vars, criterion_t1, criterion_t2)

                preds = torch.argmax(outputs_t1, dim=1)
                tp, fp, tn, fn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=siam_segment_classes)
                acc_t1 = metric_t1(tp, fp, fn, tn)
                acc_t2 = metric_t2(outputs_t2.view(-1), features.view(-1))
                acc = acc_t1 + acc_t2
            
            elif ssl_type == 'single_segsiam':                                    
                outputs_t1, _,_ = model(features)      #task 1 - siam prediction
                loss_t1 = criterion_t1(outputs_t1, labels) 
                loss_t2 = torch.Tensor([0])
                loss = loss_t1        

                preds = torch.argmax(outputs_t1, dim=1)
                tp, fp, tn, fn = smp.metrics.get_stats(preds, labels, mode='multiclass', num_classes=siam_segment_classes)
                acc_t1 = metric_t1(tp, fp, fn, tn)
                acc_t2 = torch.Tensor([0])  
                acc = acc_t1              
            
            elif ssl_type == 'single_recon':
                _, outputs_t2,_ = model(features)
                loss_t2 = criterion_t2(outputs_t2, features) #task 2 - predict input bands back
                loss_t1 = torch.Tensor([0])
                loss = loss_t2

                acc_t1 = torch.Tensor([0])
                acc_t2 = metric_t2(outputs_t2.view(-1), features.view(-1))
                acc = acc_t2

                
            # loss and number of correct predictions of the batch
            running_loss += loss.item()
            running_loss_t1 += loss_t1.item()
            running_loss_t2 += loss_t2.item()

            running_metric += acc.item()
            running_metric_t1 += acc_t1.item()
            running_metric_t2 += acc_t2.item()
            # preds = torch.argmax(outputs, dim=1)
            # correct_preds += torch.sum(preds == labels).item()

    avg_loss = running_loss / num_batches
    avg_loss_t1 = running_loss_t1 / num_batches
    avg_loss_t2 = running_loss_t2 / num_batches

    avg_metric = running_metric / num_batches
    avg_metric_t1 = running_metric_t1 / num_batches
    avg_metric_t2 = running_metric_t2 / num_batches
    # overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, avg_loss_t1,avg_loss_t2, avg_metric, avg_metric_t1, avg_metric_t2  #, overall_accuracy

def train_model_ssl(model, fold, train_data, val_data, ssl_type, batch_size, optimizer, scheduler, criterion_1,criterion_2, metric_t1, metric_t2, siam_segment_bandnum, siam_segment_classes, omega, device, patience,out_paths, epochs, ssl_init_args):
    start = time.time()
    model = model.to(device)

    min_val_loss = np.inf

    # individual losses and accuracies for each task - remember that one these can possibly be zero depending on the ssl_type (single/dual)
    train_loss_history_t1 = []
    val_loss_history_t1 = []

    train_loss_history_t2 = []
    val_loss_history_t2 = []

    train_acc_history_t1 = []
    val_acc_history_t1 = []

    train_acc_history_t2 = []
    val_acc_history_t2 = []

    #--------------------------
    log_var_rec_history = []
    log_var_seg_history = []

    train_loss_history = [] # this is the combined loss of both tasks
    val_loss_history = []

    train_acc_history = []    # this is the combined accuracy of both tasks
    val_acc_history = []
    best_epoch_metrics = {}

    model_out_path = os.path.join(out_paths[0], f'fold_{fold}_best_model_ssl.pth')
    patience_counter = 0

    for epoch in range(epochs):

        epoch_args = {'model':model, 'ssl_type':ssl_type, 
                      'criterion_t1':criterion_1, 'criterion_t2':criterion_2, 
                      'metric_t1':metric_t1, 'metric_t2':metric_t2, 
                      'siam_segment_bandnum': siam_segment_bandnum, 'siam_segment_classes': siam_segment_classes,
                      'omega':omega, 'device':device}
        
        # train_epoch_loss,train_epoch_loss_t1, train_epoch_loss_t2  = train_epoch_ssl(optimizer=optimizer, data=train_data, **epoch_args)
        # val_epoch_loss,val_epoch_loss_t1, val_epoch_loss_t2 = validate_epoch_ssl(data=val_data, **epoch_args)
        train_epoch_loss, train_epoch_loss_t1, train_epoch_loss_t2, train_epoch_acc, train_epoch_acc_t1, train_epoch_acc_t2,log_var_seg_updated, log_var_rec_updated  = train_epoch_ssl(optimizer=optimizer, data=train_data, **epoch_args)
        val_epoch_loss, val_epoch_loss_t1, val_epoch_loss_t2, val_epoch_acc, val_epoch_acc_t1, val_epoch_acc_t2 = validate_epoch_ssl(data=val_data, **epoch_args)

        print(f"\nEpoch {epoch+1}/{epochs} =>")
        print(f"Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f},\n Train Loss T1: {train_epoch_loss_t1:.4f}, Val Loss T1: {val_epoch_loss_t1:.4f}, Train Loss T2: {train_epoch_loss_t2:.4f}, Val Loss T2: {val_epoch_loss_t2:.4f}")
        print(f"Train Acc: {train_epoch_acc:.4f}, Val Acc: {val_epoch_acc:.4f},\n Train Acc T1: {train_epoch_acc_t1:.4f}, Val Acc T1: {val_epoch_acc_t1:.4f}, Train Acc T2: {train_epoch_acc_t2:.4f}, Val Acc T2: {val_epoch_acc_t2:.4f}")
        print(f"Log Var Seg: {log_var_seg_updated.item():.4f}, Log Var Rec: {log_var_rec_updated.item():.4f}")

        train_loss_history_t1.append(train_epoch_loss_t1)
        val_loss_history_t1.append(val_epoch_loss_t1)

        train_loss_history_t2.append(train_epoch_loss_t2)
        val_loss_history_t2.append(val_epoch_loss_t2)

        train_acc_history_t1.append(train_epoch_acc_t1)
        val_acc_history_t1.append(val_epoch_acc_t1)

        train_acc_history_t2.append(train_epoch_acc_t2)
        val_acc_history_t2.append(val_epoch_acc_t2)

        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)

        train_acc_history.append(train_epoch_acc)
        val_acc_history.append(val_epoch_acc)

        log_var_rec_history.append(log_var_rec_updated.item())
        log_var_seg_history.append(log_var_seg_updated.item())

        if scheduler is not None:
            scheduler.step()
        
        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            best_epoch_metrics= {'train_loss':train_epoch_loss,
                                 'val_loss':val_epoch_loss,
                                 'train_acc':train_epoch_acc,
                                 'val_acc':val_epoch_acc,

                                 'train_loss_t1':train_epoch_loss_t1,
                                 'val_loss_t1':val_epoch_loss_t1,
                                 'train_acc_t1':train_epoch_acc_t1,
                                 'val_acc_t1':val_epoch_acc_t1,

                                 'train_loss_t2':train_epoch_loss_t2,
                                 'val_loss_t2':val_epoch_loss_t2,
                                 'train_acc_t2':train_epoch_acc_t2,
                                 'val_acc_t2':val_epoch_acc_t2,

                                'log_var_seg':log_var_seg_updated.item(),
                                'log_var_rec':log_var_rec_updated.item()
                                 }
            
            ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'loss': min_val_loss,

                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'train_acc_history': train_acc_history,
                    'val_acc_history': val_acc_history,

                    'train_loss_history_t1': train_loss_history_t1,
                    'val_loss_history_t1': val_loss_history_t1,
                    'train_acc_history_t1': train_acc_history_t1,
                    'val_acc_history_t1': val_acc_history_t1,

                    'train_loss_history_t2': train_loss_history_t2,
                    'val_loss_history_t2': val_loss_history_t2,
                    'train_acc_history_t2': train_acc_history_t2,
                    'val_acc_history_t2': val_acc_history_t2,

                    'log_var_seg_history': log_var_seg_history,
                    'log_var_rec_history': log_var_rec_history
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
                      'val_loss_history':val_loss_history,
                      'train_acc_history':train_acc_history,
                      'val_acc_history':val_acc_history,

                      'train_loss_history_t1':train_loss_history_t1,
                      'val_loss_history_t1':val_loss_history_t1,
                      'train_acc_history_t1':train_acc_history_t1,
                      'val_acc_history_t1':val_acc_history_t1,

                      'train_loss_history_t2':train_loss_history_t2,
                      'val_loss_history_t2':val_loss_history_t2,
                      'train_acc_history_t2':train_acc_history_t2,
                      'val_acc_history_t2':val_acc_history_t2,

                     'log_var_seg_history': log_var_seg_history,
                     'log_var_rec_history': log_var_rec_history
                      }
    
    trained_model, best_e = load_best_model(model_out_path, **ssl_init_args)
    return trained_model, best_e, model_tracking, best_epoch_metrics, train_duration

def fit_kfolds_ssl(models,generator, optimizers, schedulers, n_splits, ssl_init_args,**model_args_ssl):
    start = time.time()
    sum_train_loss = 0.0
    sum_val_loss = 0.0
    sum_train_acc = 0.0
    sum_val_acc = 0.0

    sum_train_loss_t1 = 0.0
    sum_val_loss_t1 = 0.0
    sum_train_acc_t1 = 0.0
    sum_val_acc_t1 = 0.0

    sum_train_loss_t2 = 0.0
    sum_val_loss_t2 = 0.0
    sum_train_acc_t2 = 0.0
    sum_val_acc_t2 = 0.0

    sum_log_var_seg = 0.0
    sum_log_var_rec = 0.0

    fold_histories ={}
    fold_metrics = {}
    trained_models = []
    fold_durations = []
    best_epoch_nums = []
   
    for k, (train_data, val_data) in tqdm(enumerate(generator), desc='Folds', leave=False):

        
        model,best_e ,model_tracking, best_epoch_metrics, fold_duration = train_model_ssl(model=models[k], optimizer=optimizers[k],
                                                                scheduler = schedulers[k], fold=k,
                                                                train_data=train_data, val_data=val_data,
                                                                ssl_init_args=ssl_init_args, **model_args_ssl)
        trained_models.append(model)
        best_epoch_nums.append(best_e)
        fold_durations.append(fold_duration)

        sum_train_loss += best_epoch_metrics['train_loss']
        sum_val_loss += best_epoch_metrics['val_loss']
        sum_train_acc += best_epoch_metrics['train_acc']
        sum_val_acc += best_epoch_metrics['val_acc']

        sum_train_loss_t1 += best_epoch_metrics['train_loss_t1']
        sum_val_loss_t1 += best_epoch_metrics['val_loss_t1']
        sum_train_acc_t1 += best_epoch_metrics['train_acc_t1']
        sum_val_acc_t1 += best_epoch_metrics['val_acc_t1']

        sum_train_loss_t2 += best_epoch_metrics['train_loss_t2']
        sum_val_loss_t2 += best_epoch_metrics['val_loss_t2']
        sum_train_acc_t2 += best_epoch_metrics['train_acc_t2']
        sum_val_acc_t2 += best_epoch_metrics['val_acc_t2']

        sum_log_var_seg += best_epoch_metrics['log_var_seg']
        sum_log_var_rec += best_epoch_metrics['log_var_rec']

        fold_metrics[f'fold_{k}'] = best_epoch_metrics
        fold_histories[f'fold_{k}'] = model_tracking
        
    total_duration = time.time() - start
    print(f"Training completed for ALL all SSL folds in {total_duration//60:.0f}m {total_duration % 60:.0f}s \n")

    # save training durations for each fold and total in csv
    durations = fold_durations
    durations.append(total_duration)
    durations = [duration//60 for duration in durations]
    print("List durations" , durations)
    duration_names = ['fold_1', 'fold_2', 'fold_3', 'total']
    durations_df = pd.DataFrame(durations, index=duration_names, columns=['duration'])
    durations_df.to_csv(os.path.join(model_args_ssl['out_paths'][1], 'durations_ssl.csv'), index=True)

    cross_val_metrics = {'train_loss':sum_train_loss/n_splits,
                            'val_loss':sum_val_loss/n_splits,
                            'train_acc':sum_train_acc/n_splits,
                            'val_acc':sum_val_acc/n_splits,

                            'val_loss_t1':sum_val_loss_t1/n_splits,
                            'train_loss_t1':sum_train_loss_t1/n_splits,
                            'val_acc_t1':sum_val_acc_t1/n_splits,
                            'train_acc_t1':sum_train_acc_t1/n_splits,

                            'val_loss_t2':sum_val_loss_t2/n_splits,
                            'train_loss_t2':sum_train_loss_t2/n_splits,
                            'val_acc_t2':sum_val_acc_t2/n_splits,
                            'train_acc_t2':sum_train_acc_t2/n_splits,
                            
                            'log_var_seg':sum_log_var_seg/n_splits,
                            'log_var_rec':sum_log_var_rec/n_splits}
    
    # fold_metrics are the best metrics for each fold
    # cross_val_metrics are the average metrics for all folds    
    return trained_models, best_epoch_nums, fold_histories, fold_metrics, cross_val_metrics


def save_training_curves_ssl(best_epoch_nums, fold_histories, fold_metrics, cross_val_metrics, out_path):   
    # save each history list
    dir_hist = os.path.join(out_path, 'fold_histories_ssl')
    os.makedirs(dir_hist, exist_ok=True)
    for key, history in fold_histories.items():
        np.save(os.path.join(dir_hist, f'{key}_history.npy'), history)

    # check dual or single ssl: if sum of t1 accuracy history or t2 accuracy history is zero, then 1 by 2 plot, else 2 by 3 plot. check with in one fold
    if np.sum(fold_histories['fold_0']['train_acc_history_t1']) == 0 or np.sum(fold_histories['fold_0']['train_acc_history_t2']) == 0:
        fig, axs = plt.subplots(1,2, figsize=(12, 5))
        colors = ['b', 'g', 'r']

        for i, (fold, history) in enumerate(fold_histories.items()):
            axs[0].plot(history['train_loss_history'], label=f'Fold {i+1} Train', color=colors[i])
            axs[0].plot(history['val_loss_history'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[0].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')
            axs[1].plot(history['train_acc_history'], label=f'Fold {i+1} Train', color=colors[i])
            axs[1].plot(history['val_acc_history'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[1].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')

        axs[0].set_title('Loss', fontsize=15)
        axs[0].set_xlabel('Epochs', fontsize=12)
        axs[0].set_ylabel('Loss', fontsize=12)
        axs[0].legend()

        axs[1].set_title('Accuracy Metric', fontsize=15)
        axs[1].set_xlabel('Epochs', fontsize=12)
        axs[1].set_ylabel('Accuracy', fontsize=12)
        axs[1].legend()

        #fontsize ticks increase
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        max_x_ticks = max([len(history['train_loss_history']) for history in fold_histories.values()])  
        axs[0].set_xticks(np.arange(0, max_x_ticks, step = 5))
        axs[1].set_xticks(np.arange(0, max_x_ticks, step = 5))

        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'training_curves_ssl.png'), dpi=800, pad_inches=0.2, bbox_inches='tight')
        plt.close()

    
    else:
        fig, axs = plt.subplots(3,2, figsize=(12, 15))
        colors = ['b', 'g', 'r']

        for i, (fold, history) in enumerate(fold_histories.items()):
            axs[0,0].plot(history['train_loss_history'], label=f'Fold {i+1} Train', color=colors[i])
            axs[0,0].plot(history['val_loss_history'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[0,0].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')
            axs[0,1].plot(history['train_acc_history'], label=f'Fold {i+1} Train', color=colors[i])
            axs[0,1].plot(history['val_acc_history'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[0,1].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')

            axs[1,0].plot(history['train_loss_history_t1'], label=f'Fold {i+1} Train', color=colors[i])
            axs[1,0].plot(history['val_loss_history_t1'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[1,0].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')
            axs[1,1].plot(history['train_acc_history_t1'], label=f'Fold {i+1} Train', color=colors[i])
            axs[1,1].plot(history['val_acc_history_t1'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[1,1].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')

            axs[2,0].plot(history['train_loss_history_t2'], label=f'Fold {i+1} Train', color=colors[i])
            axs[2,0].plot(history['val_loss_history_t2'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[2,0].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')
            axs[2,1].plot(history['train_acc_history_t2'], label=f'Fold {i+1} Train', color=colors[i])
            axs[2,1].plot(history['val_acc_history_t2'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
            axs[2,1].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')

        axs[0,0].set_title('Combined Loss', fontsize=15)
        axs[0,1].set_title('Combined Accuracy Metric', fontsize=15)
        axs[1,0].set_title('Loss T1', fontsize=15)
        axs[1,1].set_title('Accuracy Metric T1', fontsize=15)
        axs[2,0].set_title('Loss T2', fontsize=15)
        axs[2,1].set_title('Accuracy Metric T2', fontsize=15)

        max_x_ticks = max([len(history['train_loss_history']) for history in fold_histories.values()])
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlabel('Epochs', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend()
            ax.set_xticks(np.arange(0, max_x_ticks, step = 5))

        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'training_curves_ssl.png'), dpi=800, pad_inches=0.2, bbox_inches='tight')
        plt.close()

        #--------------------------Plot weights and logvariance

        fig, axs = plt.subplots(1,2, figsize=(12, 5))
        colors = ['b', 'g', 'r']
        for i, (fold, history) in enumerate(fold_histories.items()):
            axs[0].plot(history['log_var_seg_history'], label=f'Fold {i+1} Segmentation', color=colors[i])
            axs[0].plot(history['log_var_rec_history'], label=f'Fold {i+1} Reconstruction', color=colors[i], linestyle='--')
            axs[0].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')

            weights_seg = [1/np.exp(log_var) for log_var in history['log_var_seg_history']] 
            weights_rec = [1/(2*np.exp(log_var)) for log_var in history['log_var_rec_history']]
            axs[1].plot(weights_seg, label=f'Fold {i+1} Segmentation', color=colors[i])
            axs[1].plot(weights_rec, label=f'Fold {i+1} Reconstruction', color=colors[i], linestyle='--')
            axs[1].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')

        # set title, labels and ticks., ticks params
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xticks(np.arange(0, max_x_ticks, step = 5))
            ax.set_xlabel('Epochs', fontsize=12)
            ax.legend()

        axs[0].set_title('Log Variance', fontsize=15)
        axs[0].set_ylabel('Log Variance', fontsize=12  )

        axs[1].set_title('Weights', fontsize=15)
        axs[1].set_ylabel('Weight', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'weights_log_variance_ssl.png'), dpi=800, pad_inches=0.2, bbox_inches='tight')
        plt.close()

    #save fold_metrics in csv
    index = list(fold_metrics['fold_0'].keys())
    metrics = pd.DataFrame(fold_metrics, index=index)
    metrics.to_csv(os.path.join(out_path, 'train_fold_metrics_ssl.csv'), index=True, index_label='metric', float_format='%.4f')

    #save cross_val_metrics in csv
    metrics = pd.DataFrame(cross_val_metrics, index=[0])
    metrics.to_csv(os.path.join(out_path, 'train_cross_val_metrics_ssl.csv'), index=False, float_format='%.4f')

# def save_training_curves_ssl(best_epoch_nums, fold_histories, fold_metrics, cross_val_metrics, out_path):
       
#      # save each history list
#     dir_hist = os.path.join(out_path, 'fold_histories_ssl')
#     os.makedirs(dir_hist, exist_ok=True)
#     for key, history in fold_histories.items():
#         np.save(os.path.join(dir_hist, f'{key}_history.npy'), history)

#     # plot training and validation losses and accuracies of all three folds. One plot for losses and one for accuracies side by side
#     fig, axs = plt.subplots(1,2, figsize=(12, 5))
#     colors = ['b', 'g', 'r']

#     for i, (fold, history) in enumerate(fold_histories.items()):
#         axs[0].plot(history['train_loss_history'], label=f'Fold {i+1} Train', color=colors[i])
#         axs[0].plot(history['val_loss_history'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
#         axs[0].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')
#         axs[1].plot(history['train_acc_history'], label=f'Fold {i+1} Train', color=colors[i])
#         axs[1].plot(history['val_acc_history'], label=f'Fold {i+1} Validation', color=colors[i], linestyle='--')
#         axs[1].axvline(x=best_epoch_nums[i], color=colors[i], linestyle='-.', label=f'Fold {i+1} Best Epoch')

#     axs[0].set_title('Loss', fontsize=15)
#     axs[0].set_xlabel('Epochs', fontsize=12)
#     axs[0].set_ylabel('Loss', fontsize=12)
#     axs[0].legend()

#     axs[1].set_title('Accuracy Metric', fontsize=15)
#     axs[1].set_xlabel('Epochs', fontsize=12)
#     axs[1].set_ylabel('Accuracy', fontsize=12)
#     axs[1].legend()

#     #fontsize ticks increase
#     for ax in axs:
#         ax.tick_params(axis='both', which='major', labelsize=12)
    
#     max_x_ticks = max([len(history['train_loss_history']) for history in fold_histories.values()])  
#     axs[0].set_xticks(np.arange(0, max_x_ticks, step = 5))
#     axs[1].set_xticks(np.arange(0, max_x_ticks, step = 5))

#     plt.tight_layout()
#     plt.savefig(os.path.join(out_path, 'training_curves_ssl.png'), dpi=800, pad_inches=0.2, bbox_inches='tight')
#     plt.close()

#     #save fold_metrics in csv
#     index = list(fold_metrics['fold_0'].keys())
#     metrics = pd.DataFrame(fold_metrics, index=index)
#     metrics.to_csv(os.path.join(out_path, 'train_fold_metrics_ssl.csv'), index=True, index_label='metric', float_format='%.4f')

#     #save cross_val_metrics in csv
#     metrics = pd.DataFrame(cross_val_metrics, index=[0])
#     metrics.to_csv(os.path.join(out_path, 'train_cross_val_metrics_ssl.csv'), index=False, float_format='%.4f')

