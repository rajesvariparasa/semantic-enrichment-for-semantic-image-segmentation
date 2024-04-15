import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import pandas as pd
from PIL import Image

def save_predicted_files(model, data, batch_size, n_classes, criterion, device, out_path):
    predictions_dir = os.path.join(out_path, 'predictions') # prepare directory for prediction files
    os.makedirs(predictions_dir, exist_ok=True)
    
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    num_batches = len(data)   
    cm = np.zeros((n_classes, n_classes)) 
    class_names = list(range(n_classes))
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data, desc='Predicting', leave=False)):
            features, labels, PIDs = batch
            dw_band = 5 # 6th band is the DW band
            features, labels = features.to(device), labels[:,dw_band,:,:].to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            for j in range(preds.shape[0]):
                pred = preds[j].cpu().numpy()
                PID = PIDs[j]
                #np.save(os.path.join(predictions_dir, f'{PID}.npy'), pred)
                pred = Image.fromarray(pred.astype(np.uint8))
                pred.save(os.path.join(predictions_dir, f'{PID}.png'))

            labels, outputs, preds = labels.view(batch_size,-1).long(), outputs.view(batch_size,n_classes,-1), preds.view(batch_size,-1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            correct_preds += torch.sum(preds == labels).item()
            cm += confusion_matrix(labels.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), labels=class_names)
    avg_loss = running_loss / num_batches
    overall_accuracy = correct_preds / (num_batches* batch_size * 510 * 510)
    return avg_loss, overall_accuracy, cm, class_names


def save_cm_metrics(avg_loss, overall_accuracy, cm, class_names, out_path):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize confusion matrix
    cm = np.round(np.nan_to_num(cm, nan=0),2)

    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax)
    # resize the matrix cells to fit the values
    disp.ax_.set_title(f'Average Loss: {avg_loss:.4f}, Overall Accuracy: {overall_accuracy:.4f}')
    plt.savefig(os.path.join(out_path, 'confusion_matrix.png'))
    plt.close()

    class_accuracy = cm.diagonal()
    class_iou = cm.diagonal() / (cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal())
    class_precision = cm.diagonal() / cm.sum(axis=0)
    class_recall = cm.diagonal() / cm.sum(axis=1)
    metrics = pd.DataFrame({'Class': class_names, 'Accuracy': class_accuracy, 'IoU': class_iou, 'Precision': class_precision, 'Recall': class_recall})
    metrics.to_csv(os.path.join(out_path, 'class_metrics.csv'), index=False)
    return metrics
