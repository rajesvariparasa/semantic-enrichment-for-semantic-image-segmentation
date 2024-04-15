import sys
#sys.path.append('/share/home/e2208165/scripts')
from data_prep import prepare_loaders
import pandas as pd
import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='SIAM for Semantic Image Segmentation')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the outputs directory')
    # since we don't need features info for this script, it doesn't matter which input type we choose
    parser.add_argument('--input_type', type=str, default='s2', help='Type of input data: s2, siam_18, siam_33, siam_48, siam_96')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--process_level', type=str, default='l1c', help='Process level of the data: l1c or l2a')
    parser.add_argument('--learn_type', type=str, default='csl', help='Type of learning: cls or ssl')
    parser.add_argument('--num_classes', type=int, default=11, help='Number of classes in the output task')
    return parser.parse_args()

def count_class_pixels(data_loader, num_classes, batch_size,path, file_name, metadata= None, target_band=5):
    tot_class_count = np.zeros(num_classes)
    for _,target,id in tqdm(data_loader, desc='Counting class pixels', leave=False):
        target,id = target[:,target_band,:,:].numpy().astype(int), np.array(id)
        class_count = np.zeros((batch_size, num_classes))
        for i in range(num_classes):
            class_count[:,i] = np.sum(target == i, axis=(1,2))
        tot_class_count += np.sum(class_count, axis=0)


        if metadata is not None:
            for i in range(num_classes):
                print("Shape ID, Target",id.shape, target.shape)
                metadata.loc[metadata['patch_id'].isin(id), f'class_{i}'] += class_count[:,i]     
                
        else:
            continue

    df_tot_class_count = pd.DataFrame({'class': list(range(num_classes)), 'count_value': tot_class_count})  
    df_tot_class_count.to_csv(os.path.join(path,file_name), index=False)
    if metadata is not None:
        return tot_class_count, metadata
    else:
        return tot_class_count

def main():
    args = parse_args()
    #device = torch.device('cpu') # to be used for debugging when gpu debugging isn't useful
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
   
    # Prepare data loaders
    train_loader, test_loader = prepare_loaders(input_dir=args.input_dir, 
                                            process_level = args.process_level,                                         
                                            learn_type = args.learn_type, 
                                            input_type = args.input_type,
                                            batch_size= args.batch_size)
    print(f"Train data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(test_loader.dataset)} samples")
    print(f"Number of batches in train: {len(train_loader)}")
    print(f"Number of batches in validation: {len(test_loader)}")
   
    metadata = pd.read_csv(os.path.join(args.input_dir, 'meta_patches.csv'))
    for i in range(args.num_classes):
        metadata[f'class_{i}'] = 0

    # train class count, edit respective patch metadata
    train_class_count, metadata = count_class_pixels(train_loader, args.num_classes, args.batch_size, 
                                                     args.out_path, 'train_class_count.csv', metadata)
    
    # test class count, edit respective patch metadata
    test_class_count, metadata_dw_consensus = count_class_pixels(test_loader, args.num_classes, args.batch_size, 
                                                                args.out_path, 'test_class_count_dw_consensus.csv', 
                                                                metadata, target_band=5)
    # write metadata csv with class counts per patch
    metadata_dw_consensus.to_csv(os.path.join(args.out_path, 'metadata_classcount.csv'), index=False)

    # test class count, DON'T edit respective patch metadata
    test_class_count_dw_majority= count_class_pixels(test_loader, args.num_classes, args.batch_size, 
                                                     args.out_path, 'test_class_count_dw_majority.csv', 
                                                     target_band=6, metadata = None)
    
    test_class_count_dw_simple_majority= count_class_pixels(test_loader, args.num_classes, args.batch_size, 
                                                           args.out_path, 'test_class_count_dw_simple_majority.csv', 
                                                           target_band=7, metadata=None)
    
    test_class_count_dw_strict= count_class_pixels(test_loader, args.num_classes, args.batch_size, 
                                                  args.out_path, 'test_class_count_dw_strict.csv', 
                                                  target_band=8, metadata=None)
    
    print("Train class count:")
    print(train_class_count)
    print("Test class count DW consensus:")
    print(test_class_count)
    print("Test class count DW majority:")
    print(test_class_count_dw_majority)
    print("Test class count DW simple majority:")
    print(test_class_count_dw_simple_majority)
    print("Test class count DW strict:")
    print(test_class_count_dw_strict)

    print("All done!")
if __name__ == "__main__":
    main()