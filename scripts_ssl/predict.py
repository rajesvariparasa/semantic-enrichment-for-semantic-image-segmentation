import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import pandas as pd
from PIL import Image
import segmentation_models_pytorch as smp

def save_predicted_files(preds, PIDs, predictions_dir):
    for j in range(preds.shape[0]):
        pred = preds[j].cpu().numpy()
        PID = PIDs[j]
        #np.save(os.path.join(predictions_dir, f'{PID}.npy'), pred)
        pred = Image.fromarray(pred.astype(np.uint8))
        pred.save(os.path.join(predictions_dir, f'{PID}.png'))
    return None

def save_cm_metrics(avg_loss, avg_iou, avg_f1, avg_accuracy, cm, class_names, out_path, dw_name=None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalize confusion matrix
    cm = np.round(np.nan_to_num(cm, nan=0),2)

    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax)
    # resize the matrix cells to fit the values
    disp.ax_.set_title(f'Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, F1: {avg_f1:.4f}, Accuracy: {avg_accuracy:.4f}')
    plt.savefig(os.path.join(out_path, f'{dw_name}_confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()

    class_accuracy = cm.diagonal()
    class_iou = cm.diagonal() / (cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal())
    class_precision = cm.diagonal() / cm.sum(axis=0)
    class_recall = cm.diagonal() / cm.sum(axis=1)
    metrics = pd.DataFrame({'Class': class_names, 'Accuracy': class_accuracy, 'IoU': class_iou, 'Precision': class_precision, 'Recall': class_recall})
    metrics.to_csv(os.path.join(out_path, f'{dw_name}_class_metrics.csv'), index=False)
    return metrics


def losses_accuracies_cms(outputs, preds, labels, criterion, metrics, class_names):

    labels = labels.long()
    lis_losses = []
    lis_ious = []
    lis_f1s = []
    lis_accuracies = []
    #lis_correct_preds = []
    lis_cms = []

    # Calculating metrics for each DW band
    # Remember that labels is already sliced to only include DW bands. So first band is dw_consensus and so on 
    for i in range(labels.shape[1]):
        labels_i = labels[:,i,:,:]
        loss = criterion(outputs, labels_i).item()
        
        preds = torch.argmax(outputs, dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(preds, labels_i, mode='multiclass', num_classes=11)
        iou = metrics['IoUScore'](tp, fp, fn, tn).item()
        f1 = metrics['F1Score'](tp, fp, fn, tn).item()
        accuracy = metrics['Accuracy'](tp, fp, fn, tn).item()

        lis_losses.append(loss)
        lis_ious.append(iou)
        lis_f1s.append(f1)
        lis_accuracies.append(accuracy)


        #correct_preds = torch.sum(preds == labels_i).item()
        #lis_correct_preds.append(correct_preds)

        cm = confusion_matrix(labels_i.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), labels=class_names)
        lis_cms.append(cm)

    dict_lis_metrics = {'iou': lis_ious, 'f1': lis_f1s, 'accuracy': lis_accuracies}
    return lis_losses, dict_lis_metrics, lis_cms

def predict(trained_models, data, batch_size, n_classes, criterion, metrics, device, out_paths):
    predictions_dir = os.path.join(out_paths[0], 'predictions') # prepare directory for prediction files
    os.makedirs(predictions_dir, exist_ok=True)
    
    trained_models = [model.to(device) for model in trained_models]
    trained_models = [model.eval() for model in trained_models]
    num_batches = len(data)   
    class_names = list(range(n_classes))

    dw_bands = 4            #number of DW bands 
    running_losses = [0.0 for i in range(dw_bands)]
    #correct_preds = [0 for i in range(dw_bands)]
    running_ious = [0.0 for i in range(dw_bands)]   
    running_f1s = [0.0 for i in range(dw_bands)]    
    running_accuracies = [0.0 for i in range(dw_bands)]

    cms = [np.zeros((n_classes, n_classes)) for i in range(dw_bands)]

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
            lis_losses, dict_lis_metrics, lis_cms= losses_accuracies_cms(outputs, preds, labels, criterion,metrics, class_names)
            running_losses = [running_losses[i] + lis_losses[i] for i in range(dw_bands)]
            #correct_preds = [correct_preds[i] + lis_correct_preds[i] for i in range(dw_bands)]
            running_ious = [running_ious[i] + dict_lis_metrics['iou'][i] for i in range(dw_bands)]
            running_f1s = [running_f1s[i] + dict_lis_metrics['f1'][i] for i in range(dw_bands)]
            running_accuracies = [running_accuracies[i] + dict_lis_metrics['accuracy'][i] for i in range(dw_bands)]
    
            cms = [cms[i] + lis_cms[i] for i in range(dw_bands)]
            
    #print("before list operation",running_losses, correct_preds)
    avg_losses = [running_losses[i] / num_batches for i in range(dw_bands)]
    #overall_accuracies = [correct_preds[i] / (num_batches* batch_size * 510 * 510) for i in range(dw_bands)]
    avg_ious = [running_ious[i] / num_batches for i in range(dw_bands)]
    avg_f1s = [running_f1s[i] / num_batches for i in range(dw_bands)]
    avg_accuracies = [running_accuracies[i] / num_batches for i in range(dw_bands)]

    print(f"\nAverage Losses: ", avg_losses)   
    print("Average Accuracies: ", avg_accuracies)
    print("Average IoUs: ", avg_ious)
    print("Average F1s: ", avg_f1s)

    # save confusion matrices and class-wise metrics
    dw_dict = {0: 'dw_consensus', 1: 'dw_majority', 2: 'dw_simple_majority', 3: 'dw_strict'}
    for i in range(dw_bands):
        cm = cms[i]
        dw_name = dw_dict[i] 
        save_cm_metrics(avg_loss=avg_losses[i], avg_iou= avg_ious[i], avg_f1=avg_f1s[i], avg_accuracy= avg_accuracies[i],
                        cm=cm, class_names=class_names, out_path= out_paths[1], dw_name= dw_name)

    # save final test metrics
    # final_test_metrics = pd.DataFrame({'DW Band': list(dw_dict.values()), 'Average Loss': avg_losses, 'Overall Accuracy': overall_accuracies})
    # final_test_metrics.to_csv(os.path.join(out_paths[1], 'final_test_metrics.csv'), index=False)
    # return final_test_metrics  

    # save final test metrics
    final_test_metrics = pd.DataFrame({'DW Band': list(dw_dict.values()), 'Average Loss': avg_losses, 
                                       'Average Accuracy': avg_accuracies, 'Average IoU': avg_ious, 'Average F1': avg_f1s})
    final_test_metrics.to_csv(os.path.join(out_paths[1], 'final_test_metrics.csv'), index=False, float_format='%.4f')

    return final_test_metrics
            



