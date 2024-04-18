import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def train_epoch(model, data, batch_size, n_classes, optimizer, criterion, device):
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
        outputs = outputs.view(batch_size,n_classes,-1)
        labels = labels.view(batch_size,-1).long()

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

def validate_epoch(model, data,batch_size, n_classes, criterion, device):
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
            labels, outputs = labels.view(batch_size,-1).long(), outputs.view(batch_size,n_classes,-1)
            loss = criterion(outputs, labels)
            
            # loss and number of correct predictions of the batch
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(preds == labels).item()

    avg_loss = running_loss / num_batches
    overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, overall_accuracy

def fit_kfolds(generator,model,batch_size, n_classes, optimizer, criterion, device):
    train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history = [], [],[],[]
   
    for k, (train_data, val_data) in tqdm(enumerate(generator), desc='Folds', leave=False):
        train_loss, train_accuracy = train_epoch(model, train_data, batch_size, n_classes, optimizer, criterion, device)
        val_loss, val_accuracy = validate_epoch(model, val_data, batch_size, n_classes, criterion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)
        print(f"Fold {k+1}/5")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

def train_model_cross_val(model, generator_func,generator_args, batch_size, n_classes, optimizer, scheduler, criterion, device, patience,out_path, epochs=10):
    start = time.time()
    model = model.to(device)
    min_val_loss = np.inf
    train_loss_history, train_accuracy_history,val_loss_history, val_accuracy_history  = [],[],[],[]
    folds_train_loss_history, folds_train_accuracy_history, folds_val_loss_history, folds_val_accuracy_history  = [],[],[],[]
    model_out_path = os.path.join(out_path, 'best_model.pth')

    patience_counter = 0

    for epoch in range(epochs):

        generator = generator_func(**generator_args) #reset generator for each epoch
        folds_train_losses, folds_train_accuracies, folds_val_losses, folds_val_accuracies = fit_kfolds(generator, model, batch_size, n_classes, optimizer, criterion, device)
        train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy = np.mean(folds_train_losses), np.mean(folds_train_accuracies), np.mean(folds_val_losses), np.mean(folds_val_accuracies)
        
        print(f"Epoch {epoch+1}/{epochs} => Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f} , Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")
    
        train_loss_history.append(train_epoch_loss)
        train_accuracy_history.append(train_epoch_accuracy)
        val_loss_history.append(val_epoch_loss)
        val_accuracy_history.append(val_epoch_accuracy)

        folds_train_loss_history.extend(folds_train_losses)
        folds_train_accuracy_history.extend(folds_train_accuracies)
        folds_val_loss_history.extend(folds_val_losses)
        folds_val_accuracy_history.extend(folds_val_accuracies)

        if scheduler is not None:
            scheduler.step()
        
        if val_epoch_loss < min_val_loss:
            min_val_loss = val_epoch_loss
            
            ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    # 'criterion': criterion,
                    'train_loss_history': train_loss_history,
                    'train_accuracy_history': train_accuracy_history,
                    'val_loss_history': val_loss_history,
                    'val_accuracy_history': val_accuracy_history,
                    'folds_train_loss_history': folds_train_loss_history,
                    'folds_train_accuracy_history': folds_train_accuracy_history,
                    'folds_val_loss_history': folds_val_loss_history,
                    'folds_val_accuracy_history': folds_val_accuracy_history,
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
    print(f"Training completed in {train_duration//60:.0f}m {train_duration % 60:.0f}s")
    return model, (train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history), (folds_train_loss_history, folds_train_accuracy_history, folds_val_loss_history, folds_val_accuracy_history)


def save_training_curves(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, out_path):
    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].plot(train_loss_history, label='Train Loss')
    ax[0].plot(val_loss_history, label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(train_accuracy_history, label='Train Accuracy')
    ax[1].plot(val_accuracy_history, label='Val Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plot_out_path = os.path.join(out_path, 'training_curves.png')
    plt.savefig(plot_out_path)
    plt.close()


def save_folds_training_curves(folds_train_loss_history, folds_train_accuracy_history, folds_val_loss_history, folds_val_accuracy_history, out_path):
    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].plot(folds_train_loss_history, label='Train Loss')
    ax[0].plot(folds_val_loss_history, label='Val Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(folds_train_accuracy_history, label='Train Accuracy')
    ax[1].plot(folds_val_accuracy_history, label='Val Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Batch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plot_out_path = os.path.join(out_path, 'folds_training_curves.png')
    plt.savefig(plot_out_path)
    plt.close()


# plot both curbes on top of each other for comparison
def compare_training_curves(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, folds_train_loss_history, folds_train_accuracy_history, folds_val_loss_history, folds_val_accuracy_history, out_path):
    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].plot(train_loss_history, label='Train Loss')
    ax[0].plot(val_loss_history, label='Val Loss')
    ax[0].plot(folds_train_loss_history, label='Folds Train Loss')
    ax[0].plot(folds_val_loss_history, label='Folds Val Loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Batch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(train_accuracy_history, label='Train Accuracy')
    ax[1].plot(val_accuracy_history, label='Val Accuracy')
    ax[1].plot(folds_train_accuracy_history, label='Folds Train Accuracy')
    ax[1].plot(folds_val_accuracy_history, label='Folds Val Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Batch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plot_out_path = os.path.join(out_path, 'compare_training_curves.png')
    plt.savefig(plot_out_path)
    plt.close()


def save_curves(out_path, epoch_metrics, folds_metrics):
    save_training_curves(train_loss_history= epoch_metrics[0], train_accuracy_history= epoch_metrics[1], 
                         val_loss_history = epoch_metrics[2], val_accuracy_history = epoch_metrics[3], out_path = out_path)
    save_folds_training_curves(folds_train_loss_history= folds_metrics[0], folds_train_accuracy_history= folds_metrics[1], 
                         folds_val_loss_history = folds_metrics[2], folds_val_accuracy_history = folds_metrics[3], out_path = out_path)
    compare_training_curves(train_loss_history= epoch_metrics[0], train_accuracy_history= epoch_metrics[1], 
                         val_loss_history = epoch_metrics[2], val_accuracy_history = epoch_metrics[3], 
                         folds_train_loss_history= folds_metrics[0], folds_train_accuracy_history= folds_metrics[1], 
                         folds_val_loss_history = folds_metrics[2], folds_val_accuracy_history = folds_metrics[3], out_path = out_path)
    
