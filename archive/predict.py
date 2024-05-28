import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import pandas as pd
from PIL import Image

def save_predicted_files(preds, PIDs, predictions_dir):
    for j in range(preds.shape[0]):
        pred = preds[j].cpu().numpy()
        PID = PIDs[j]
        #np.save(os.path.join(predictions_dir, f'{PID}.npy'), pred)
        pred = Image.fromarray(pred.astype(np.uint8))
        pred.save(os.path.join(predictions_dir, f'{PID}.png'))
    return None

def save_cm_metrics(avg_loss, overall_accuracy, cm, class_names, out_path, dw_name=None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize confusion matrix
    cm = np.round(np.nan_to_num(cm, nan=0),2)

    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax)
    # resize the matrix cells to fit the values
    disp.ax_.set_title(f'Average Loss: {avg_loss:.4f}, Overall Accuracy: {overall_accuracy:.4f}')
    plt.savefig(os.path.join(out_path, f'{dw_name}_confusion_matrix.png'))
    plt.close()

    class_accuracy = cm.diagonal()
    class_iou = cm.diagonal() / (cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal())
    class_precision = cm.diagonal() / cm.sum(axis=0)
    class_recall = cm.diagonal() / cm.sum(axis=1)
    metrics = pd.DataFrame({'Class': class_names, 'Accuracy': class_accuracy, 'IoU': class_iou, 'Precision': class_precision, 'Recall': class_recall})
    metrics.to_csv(os.path.join(out_path, f'{dw_name}_class_metrics.csv'), index=False)
    return metrics


def losses_accuracies_cms(outputs, preds, labels, criterion, class_names):

    labels = labels.long()
    lis_losses = []
    lis_correct_preds = []
    lis_cms = []

    # Calculating metrics for each DW band
    # Remember that labels is already sliced to only include DW bands. So first band is dw_consensus and so on 
    for i in range(labels.shape[1]):
        labels_i = labels[:,i,:,:]
        loss = criterion(outputs, labels_i).item()
        lis_losses.append(loss)

        correct_preds = torch.sum(preds == labels_i).item()
        lis_correct_preds.append(correct_preds)

        cm = confusion_matrix(labels_i.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), labels=class_names)
        lis_cms.append(cm)
    return lis_losses, lis_correct_preds, lis_cms

def predict(trained_models, data, batch_size, n_classes, criterion, device, out_paths):
    predictions_dir = os.path.join(out_paths[0], 'predictions') # prepare directory for prediction files
    os.makedirs(predictions_dir, exist_ok=True)
    
    trained_models = [model.to(device) for model in trained_models]
    trained_models = [model.eval() for model in trained_models]

    dw_bands = 4
    running_losses = [0.0 for i in range(dw_bands)]
    correct_preds = [0 for i in range(dw_bands)]
    cms = [np.zeros((n_classes, n_classes)) for i in range(dw_bands)]

    num_batches = len(data)   
    class_names = list(range(n_classes))

    with torch.no_grad():
        for _, batch in enumerate(tqdm(data, desc='Predicting', leave=False)):
            features, labels, PIDs = batch
            features, labels = features.to(device), labels[:,5:,:,:].to(device) # take only the DW bands

            outputs1 = trained_models[0](features)              #shape: (batch_size, n_classes, 510, 510)
            outputs2 = trained_models[1](features)
            outputs3 = trained_models[2](features)
            outputs = (outputs1 + outputs2 + outputs3) / 3              #model ensemble
            
            # save predicted files
            preds = torch.argmax(outputs, dim=1)
            save_predicted_files(preds, PIDs, predictions_dir)

            # getting metrics for each DW band
            lis_losses, lis_correct_preds, lis_cms= losses_accuracies_cms(outputs, preds, labels, criterion,class_names)
            running_losses = [running_losses[i] + lis_losses[i] for i in range(dw_bands)]
            correct_preds = [correct_preds[i] + lis_correct_preds[i] for i in range(dw_bands)]
            cms = [cms[i] + lis_cms[i] for i in range(dw_bands)]
            
    #print("before list operation",running_losses, correct_preds)
    avg_losses = [running_losses[i] / num_batches for i in range(dw_bands)]
    overall_accuracies = [correct_preds[i] / (num_batches* batch_size * 510 * 510) for i in range(dw_bands)]
    print("Average test losses and overall test accuraccies",avg_losses, overall_accuracies)

    # save confusion matrices and class-wise metrics
    dw_dict = {0: 'dw_consensus', 1: 'dw_majority', 2: 'dw_simple_majority', 3: 'dw_strict'}
    for i in range(dw_bands):
        cm = cms[i]
        dw_name = dw_dict[i]
        save_cm_metrics(avg_losses[i], overall_accuracies[i], cm, class_names, out_paths[1], dw_name)

    # save final test metrics
    final_test_metrics = pd.DataFrame({'DW Band': list(dw_dict.values()), 'Average Loss': avg_losses, 'Overall Accuracy': overall_accuracies})
    final_test_metrics.to_csv(os.path.join(out_paths[1], 'final_test_metrics.csv'), index=False)
    return final_test_metrics  
            



