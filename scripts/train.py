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
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        #print(f"before {outputs.shape}, {labels.shape}")
        outputs = outputs.view(batch_size,n_classes,-1)
        labels = labels.view(batch_size,-1).long()
        #print(f"after {outputs.shape}, {labels.shape}")

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
        #data_with_progress = tqdm(data, desc='Validation', leave=False)

        for _,batch in enumerate(tqdm(data, desc='Validation', leave=False)):
            features, labels,_ = batch
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


# early stopping, checkpoint
def train_model(model, train_data, val_data, batch_size, n_classes, optimizer, scheduler, criterion, device, patience,out_path, epochs=10):
    start = time.time()
    model = model.to(device)

    min_val_loss = np.inf
    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    model_out_path = os.path.join(out_path, 'best_model.pth')
    patience_counter = 0

    for epoch in range(epochs):
        #print(f"Epoch {epoch+1}/{epochs} started")
        train_epoch_loss, train_epoch_accuracy = train_epoch(model, train_data, batch_size, n_classes, optimizer, criterion, device)
        val_epoch_loss, val_epoch_accuracy = validate_epoch(model, val_data, batch_size, n_classes, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} => Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.4f} , Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")
    
        train_loss_history.append(train_epoch_loss)
        val_loss_history.append(val_epoch_loss)
        train_accuracy_history.append(train_epoch_accuracy)
        val_accuracy_history.append(val_epoch_accuracy)

        if scheduler is not None:
            scheduler.step(val_epoch_loss)
        
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

    
    return model, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history


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
