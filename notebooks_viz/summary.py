import os
import pandas as pd
import torch

def get_run_summary(path, modelname):
    path_ = os.path.join(path, modelname,'quickview' ,'train_fold_metrics.csv')
    fold_metrics = pd.read_csv(path_)

    #df = pd.DataFrame(columns=['job_id','model','fold_0_iou', 'fold_1_iou', 'fold_2_iou', 'fold_0_f1', 'fold_1_f1', 'fold_2_f1'])
    df = pd.DataFrame()
    df['basename'] = [modelname]
    df['model'] = [modelname[15:]]
    df['job_id'] = [modelname[:5]]
   
   # fold metrics
    df['fold_0_iou'] = fold_metrics.loc[fold_metrics['metric']== 'val_iou', 'fold_0'].values
    df['fold_1_iou'] = fold_metrics.loc[fold_metrics['metric']== 'val_iou', 'fold_1'].values
    df['fold_2_iou'] = fold_metrics.loc[fold_metrics['metric']== 'val_iou', 'fold_2'].values
    df['fold_0_f1'] = fold_metrics.loc[fold_metrics['metric']== 'val_f1', 'fold_0'].values
    df['fold_1_f1'] = fold_metrics.loc[fold_metrics['metric']== 'val_f1', 'fold_1'].values
    df['fold_2_f1'] = fold_metrics.loc[fold_metrics['metric']== 'val_f1', 'fold_2'].values

    # test metrics
    path_ = os.path.join(path, modelname,'quickview' ,'final_test_metrics.csv')
    test_metrics = pd.read_csv(path_)
    df['test_iou'] = test_metrics.loc[test_metrics['DW Band']=='dw_majority', 'Average IoU'].values
    df['test_f1'] = test_metrics.loc[test_metrics['DW Band']=='dw_majority', 'Average F1'].values

    # duration
    path_ = os.path.join(path, modelname,'quickview' ,'durations.csv')
    durations = pd.read_csv(path_)
    df['tot_duration'] = durations.iloc[-1]['duration']
    return df


def get_best_epoch(path, modelname, convergence_analysis=True, thresh=0.98):
    df = pd.DataFrame()
    df['basename'] = [modelname]
    df['model'] = [modelname[15:]]
    df['job_id'] = [modelname[:5]]

    # for each fold add best epoch to the df
    for i in range(3):
        path_ = os.path.join(path, modelname,'model_outputs' ,f'fold_{i}_best_model.pth')
        checkpoint = torch.load(path_, map_location='cpu')
        best_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['val_iou_history'][-1]
        df[f'fold_{i}_best_epoch'] = best_epoch
    
    # if convergence analysis is required, add convergence epoch
        if convergence_analysis:
            conv_accuracy = thresh * best_accuracy
            conv_epoch = None
            for epoch, accuracy in enumerate(checkpoint['val_iou_history']):
                if accuracy >= conv_accuracy:
                    conv_epoch = epoch
                    # print(conv_epoch)
                    break
            df[f'fold_{i}_conv_epoch'] = conv_epoch
            
    return df


def perform_analysis(summary_df):
    join_df = summary_df.copy()


    join_df['stdev_val_iou'] = join_df[['fold_0_iou', 'fold_1_iou', 'fold_2_iou']].std(axis=1)
    join_df['stdev_val_f1'] = join_df[['fold_0_f1', 'fold_1_f1', 'fold_2_f1']].std(axis=1)

    join_df['avg_val_iou'] = join_df[['fold_0_iou', 'fold_1_iou', 'fold_2_iou']].mean(axis=1)
    join_df['avg_val_f1'] = join_df[['fold_0_f1', 'fold_1_f1', 'fold_2_f1']].mean(axis=1)

    cols = join_df.columns.tolist()
    if 'fold_0_conv_epoch' in cols:

        splits = 3
        epochs_total = 80
        join_df['fold_0_conv_time'] = (join_df['tot_duration']/(splits*epochs_total)) * join_df['fold_0_conv_epoch']
        join_df['fold_1_conv_time'] = (join_df['tot_duration']/(splits*epochs_total)) * join_df['fold_1_conv_epoch']
        join_df['fold_2_conv_time'] = (join_df['tot_duration']/(splits*epochs_total)) * join_df['fold_2_conv_epoch']

        join_df['stddev_conv_time'] = join_df[['fold_0_conv_time', 'fold_1_conv_time', 'fold_2_conv_time']].std(axis=1)
        join_df['stddev_conv_epoch'] = join_df[['fold_0_conv_epoch', 'fold_1_conv_epoch', 'fold_2_conv_epoch']].std(axis=1)
        join_df['avg_conv_time'] = join_df[['fold_0_conv_time', 'fold_1_conv_time', 'fold_2_conv_time']].mean(axis=1)
        join_df['avg_conv_epoch'] = join_df[['fold_0_conv_epoch', 'fold_1_conv_epoch', 'fold_2_conv_epoch']].mean(axis=1)    

    join_df.reset_index(drop=True, inplace=True)
    return join_df

def add_modelname_traintype(calc_df):

    calc_df['model_name'] = calc_df['model'].str.replace('_scratch', '').str.replace('_fe', '').str.replace('_ft', '')
    mapping_dic = {'s2_scratch': 'from scratch',
                'siam_18_scratch': 'from scratch' , 
                    'siam_33_scratch':'from scratch',
                    'siam_48_scratch': 'from scratch',
                    'siam_96_scratch': 'from scratch',
                    's2_siam_96_scratch': 'from scratch',
                    'single_recon_fe': 'feature extraction',
                    'single_segsiam_fe': 'feature extraction',
                    'dual_fe': 'feature extraction',
                    'single_recon_ft': 'fine-tuning',
                    'single_segsiam_ft': 'fine-tuning',
                    'dual_ft': 'fine-tuning'
                }

    calc_df['train_type'] = calc_df['model'].map(mapping_dic)
    calc_df

    #move last two columns to the front
    cols = calc_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    calc_df = calc_df[cols]
    calc_df
    return calc_df
    

def class_wise_summary(path, modelname):

    name_dict = {
    0: 'na',
    1: 'water',
    2: 'tree',
    3: 'grass',
    4: 'flood',
    5: 'crop',
    6: 'scrub',
    7: 'built',
    8: 'bare_gr',
    9: 'snow',
    10: 'cloud'
}
    path_ = os.path.join(path, modelname,'quickview' ,'dw_majority_class_metrics.csv')
    class_wise = pd.read_csv(path_)
    class_wise['Class'] = class_wise['Class'].map(name_dict)
    cols = class_wise['Class'].unique() 
    vals = class_wise['IoU'].values
    df = pd.DataFrame(data=[vals], columns=cols)
    df['basename'] = [modelname]
    df['model'] = [modelname[15:]]
    return df