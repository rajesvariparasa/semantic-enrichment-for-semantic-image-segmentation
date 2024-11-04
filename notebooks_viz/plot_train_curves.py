import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_training_curves(best_epoch_nums, foldwise_histories, foldwise_best_epoch_metrics, cross_val_metrics, out_path):
    fig, axs = plt.subplots(2,2, figsize= (12,5*2))
    colors = ['b', 'g', 'r']
    for i, (fold, history) in enumerate(foldwise_histories.items()):
        axs[0,0].plot(history['train_loss_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[0,0].plot(history['val_loss_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])
        axs[0,0].axvline(x=best_epoch_nums[i], color = colors[i], linestyle='-.', label=f"Fold {i+1} Best Epoch")

        axs[0,1].plot(history['train_iou_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[0,1].plot(history['val_iou_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])
        axs[0,1].axvline(x=best_epoch_nums[i], color = colors[i], linestyle='-.', label=f"Fold {i+1} Best Epoch")

        axs[1,0].plot(history['train_f1_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[1,0].plot(history['val_f1_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])
        axs[1,0].axvline(x=best_epoch_nums[i], color = colors[i], linestyle='-.', label=f"Fold {i+1} Best Epoch")

        axs[1,1].plot(history['train_accuracy_history'], label=f"Fold {i+1} Train", color = colors[i])
        axs[1,1].plot(history['val_accuracy_history'], label=f"Fold {i+1} Validation", linestyle='--', color = colors[i])
        axs[1,1].axvline(x=best_epoch_nums[i], color = colors[i], linestyle='-.', label=f"Fold {i+1} Best Epoch")
    
    axs[0,0].set_title('Dice Loss', fontsize=15)
    axs[0,0].set_xlabel('Epoch', fontsize=12)
    axs[0,0].set_ylabel('Loss', fontsize=12)
    axs[0,0].legend()

    axs[0,1].set_title('IoU', fontsize=15)
    axs[0,1].set_xlabel('Epoch', fontsize=12)
    axs[0,1].set_ylabel('IoU', fontsize=12)
    axs[0,1].legend()

    axs[1,0].set_title('F1 Score', fontsize=15)
    axs[1,0].set_xlabel('Epoch', fontsize=12)
    axs[1,0].set_ylabel('F1 Score')
    axs[1,0].legend()

    axs[1,1].set_title('Accuracy', fontsize=15)
    axs[1,1].set_xlabel('Epoch', fontsize=12)
    axs[1,1].set_ylabel('Accuracy', fontsize=12)
    axs[1,1].legend()

    #fontsize ticks increase
    for ax in axs.flatten():
        ax.tick_params(axis='both', which='major', labelsize=12)

    # save each history list
    dir_hist = os.path.join(out_path, 'fold_histories')
    os.makedirs(dir_hist, exist_ok=True)
    for key, history in foldwise_histories.items():
        np.save(os.path.join(dir_hist, f'{key}_history.npy'), history)

    max_x_ticks = max([len(history['train_loss_history']) for history in foldwise_histories.values()])
    for ax in axs.flatten():
        ax.set_xticks(np.arange(0, max_x_ticks, step = 5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'training_curves.png'), dpi=300,  pad_inches=0.2, bbox_inches='tight')
    plt.close()

    #save fold_metrics in csv
    index= list(foldwise_best_epoch_metrics['fold_1'].keys())  # get the keys from the first fold
    metrics = pd.DataFrame(foldwise_best_epoch_metrics, index=index)
    metrics.to_csv(os.path.join(out_path, 'train_fold_metrics.csv'), index=True, index_label='metric', float_format='%.4f')

    #save cross_val_metrics in csv

    metrics = pd.DataFrame(cross_val_metrics,  index=[0])
    metrics.to_csv(os.path.join(out_path, 'train_cross_val_metrics.csv'), index=False, float_format='%.4f')

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
