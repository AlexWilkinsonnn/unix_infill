import os, sys, time, datetime, random, math, pickle, argparse, itertools
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

sys.path.append("../model")
from loss_sparse_infill_induction import SparseInfillLoss #from loss_sparse_infill import SparseInfillLoss
from sparseinfill import SparseInfill


def train(model, train_files, valid_files, maskpatterns, epochs, batchsize, info):
    overtain_cntr = 0
    train_times = []
    train_losses, valid_losses = [], []

    batchsizes_train = [batchsize]*(int((len(train_files)/batchsize)))
    batchsizes_train.append(len(train_files) % batchsize)
    batchsizes_valid = [batchsize]*(int((len(valid_files)/batchsize)))
    batchsizes_valid.append(len(valid_files) % batchsize)
    if batchsizes_train[-1] == 0:
        batchsizes_train.pop()
    if batchsizes_valid[-1] == 0:
        batchsizes_valid.pop()

    for epoch in range(epochs):
        model.train()
        
        start = time.time()
                
        cnt = 0
        epoch_running_train_loss = 0.0
        running_loss = 0.0

        random.shuffle(train_files)
        files_for_batches = np.split(np.array(train_files), [ sum(batchsizes_train[:i]) for i in range(1, len(batchsizes_train)) ])

        for batch_files in files_for_batches:
            for batch_idx, filepath in enumerate(batch_files):
                with open(filepath, 'r') as f:
                    arr = np.load(filepath)

                    maskpattern = random.sample(maskpatterns, 1)[0]
                    offset = random.randint(0,1147)
                    offset_maskpattern = [ i + offset if (i + offset) < 1148 else i - 1148 + offset for i in maskpattern ]
                    arr_mask = np.copy(arr)
                    arr_mask[offset_maskpattern,:] = 0
                    
                    coords = np.array(np.argwhere(np.abs(arr_mask) > 10))
                    for wire in offset_maskpattern:
                        coords = np.row_stack((coords, np.c_[np.full((6000 ,1), wire), np.array((range(6000))).reshape(-1, 1)]))
                    sparse_arr = arr[coords[:,0],coords[:,1]]
                    sparse_arr_mask = arr_mask[coords[:,0],coords[:,1]]

                    if batch_idx == 0:
                        masked_tensor_batch = torch.FloatTensor(sparse_arr_mask.reshape(sparse_arr_mask.size, 1))
                        true_tensor_batch = torch.FloatTensor(sparse_arr.reshape(sparse_arr.size, 1))
                        loc_tensor_batch = torch.LongTensor(np.column_stack((coords, np.full((coords.shape[0], 1), batch_idx))))

                    else:
                        masked_tensor_batch = torch.cat((masked_tensor_batch, torch.FloatTensor(sparse_arr_mask.reshape(sparse_arr_mask.size, 1))), 0)
                        true_tensor_batch = torch.cat((true_tensor_batch, torch.FloatTensor(sparse_arr.reshape(sparse_arr.size, 1))), 0)
                        loc_tensor_batch = torch.cat((loc_tensor_batch, torch.LongTensor(np.column_stack((coords, np.full((coords.shape[0], 1), batch_idx))))), 0)

            masked_tensor_batch = masked_tensor_batch.to(info["DEVICE"])
            true_tensor_batch = true_tensor_batch.to(info["DEVICE"])
            loc_tensor_batch = loc_tensor_batch.to(info["DEVICE"])

            info["optimizer"].zero_grad()
            outputs = model(loc_tensor_batch, masked_tensor_batch, batch_files.size)
            loss = info["criterion"](outputs, true_tensor_batch, masked_tensor_batch)[5]
            loss.backward()
            info["optimizer"].step()

            cnt += 1
            running_loss += loss.item()
            epoch_running_train_loss += loss.item()
            if (cnt + 1) % 2 == 0:
                print('[{}, {:2.2%}] loss: {:.2f}'.format(epoch + 1, (cnt*batchsize)/float(len(train_files)), running_loss/2))
                running_loss = 0.0
        
        end = time.time()
        print('Training time: {:.1f}s'.format(end - start))
        train_times.append(end - start)
        train_losses.append(epoch_running_train_loss/len(files_for_batches))

        model.eval()

        running_loss = 0.0

        files_for_batches = np.split(np.array(valid_files), [ sum(batchsizes_valid[:i]) for i in range(1, len(batchsizes_valid)) ])
        maskpattern_pool = itertools.cycle(maskpatterns)

        with torch.no_grad():
            for idx, batch_files in enumerate(files_for_batches):   
                for batch_idx, filepath in enumerate(batch_files):
                    with open(filepath, 'r') as f:
                        arr = np.load(filepath)

                        # maskpattern = maskpatterns[(len(maskpatterns) + batch_idx) % len(maskpatterns)]
                        maskpattern = next(maskpattern_pool)
                        arr_mask = np.copy(arr)
                        arr_mask[maskpattern,:] = 0
                        
                        coords = np.array(np.argwhere(np.abs(arr_mask) > 10))
                        for wire in maskpattern:
                            coords = np.row_stack((coords, np.c_[np.full((6000 ,1), wire), np.array((range(6000))).reshape(-1, 1)]))
                        sparse_arr = arr[coords[:,0],coords[:,1]]
                        sparse_arr_mask = arr_mask[coords[:,0],coords[:,1]]
                        
                        if batch_idx == 0:
                            masked_tensor_batch = torch.FloatTensor(sparse_arr_mask.reshape(sparse_arr_mask.size, 1))
                            true_tensor_batch = torch.FloatTensor(sparse_arr.reshape(sparse_arr.size, 1))
                            loc_tensor_batch = torch.LongTensor(np.column_stack((coords, np.full((coords.shape[0], 1), batch_idx))))

                        else:
                            masked_tensor_batch = torch.cat((masked_tensor_batch, torch.FloatTensor(sparse_arr_mask.reshape(sparse_arr_mask.size, 1))), 0)
                            true_tensor_batch = torch.cat((true_tensor_batch, torch.FloatTensor(sparse_arr.reshape(sparse_arr.size, 1))), 0)
                            loc_tensor_batch = torch.cat((loc_tensor_batch, torch.LongTensor(np.column_stack((coords, np.full((coords.shape[0], 1), batch_idx))))), 0)

                masked_tensor_batch = masked_tensor_batch.to(info["DEVICE"])
                true_tensor_batch = true_tensor_batch.to(info["DEVICE"])
                loc_tensor_batch = loc_tensor_batch.to(info["DEVICE"])

                outputs = model(loc_tensor_batch, masked_tensor_batch, batch_files.size)
                loss = info["criterion"](outputs, true_tensor_batch, masked_tensor_batch)[5]

                running_loss += loss.item()

        valid_losses.append(running_loss/len(files_for_batches))
        print("Validation loss: {:.2f}".format(running_loss/len(files_for_batches)))

        if epoch == 0:
            torch.save(model.module.state_dict(), info["model_name"] + '_' + str(epoch + 1) + 'epochs.pth')
            old_valid_loss = valid_losses[0]

        else:
            if (valid_losses[-1] - old_valid_loss) < 0:
                torch.save(model.module.state_dict(), info["model_name"] + '_' + str(epoch + 1) + 'epochs.pth')
                old_valid_loss = valid_losses[-1]
                overtain_cntr = 0 

            else:
                overtain_cntr += 1

        if overtain_cntr > 2:
            break

    print(train_losses)
    print(valid_losses)
    losses = [train_losses, valid_losses]
    with open(info["losscurve_name"] + '_' + str(epoch + 1 - overtain_cntr) + 'epochs' + '_data.pkl', 'w') as f:
        pickle.dump(losses, f)


def main(train_dir, valid_dir, collection, induction, epochs, batchsize, model_name):
    DEVICE = torch.device("cuda:0")

    imgdims = 2
    ninput_features  = 16
    noutput_features = 16
    nplanes = 5 
    reps = 1
    filter_size = 3
    downsample = [2,2]
    model = SparseInfill((6000, 6000), reps, filter_size, ninput_features, noutput_features, nplanes, show_sizes=False, downsample=downsample)
    model = nn.DataParallel(model)
    model.to(DEVICE)

    # Need to use the other loss if collection plane
    criterion = SparseInfillLoss().to(device=DEVICE)

    lr = 1.0e-4
    momentum = 0.9
    weight_decay = 1.0e-4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_files = [ os.path.join(train_dir, filename) for filename in os.listdir(train_dir) if filename.endswith(".npy") ]
    valid_files = [ os.path.join(valid_dir, filename) for filename in os.listdir(valid_dir) if filename.endswith(".npy") ]

    losscurve_name = "losscurve_" + model_name

    if induction:
        maskpatterns = [[0, 1, 42, 47, 53, 55, 56, 72, 77, 171, 173, 401, 402, 404, 494, 600, 602, 604, 606, 608, 725, 800, 801, 842, 847, 853, 855, 856, 872, 877, 971, 973],
                        [0, 334, 339, 346, 400, 401, 476, 591, 593, 595, 597, 599, 800, 1134, 1139, 1146],
                        [114, 273, 401, 914, 1073],
                        [181, 183, 301, 303, 701, 703, 781, 783, 981, 983, 1101, 1103],
                        [405, 551, 601, 641, 643, 657, 680, 703],
                        [612],
                        [0, 1, 238, 400, 648, 661, 800, 801, 1038],
                        [0, 21, 23, 341, 343, 781, 783, 800, 821, 823, 1141, 1143],
                        [57, 160, 267, 384, 857, 960, 1067],
                        [17, 19, 23, 29, 77, 229, 239, 563, 630, 817, 819, 823, 829, 877, 1029, 1039],
                        [1, 201, 281, 563, 801, 1001, 1081]]
    
    elif collection:
        # maskpatterns = [[...], ...]
        pass

    info = { 
             "DEVICE" : DEVICE,
             "criterion" : criterion,
             "optimizer" : optimizer,
             "model_name" : model_name,
             "losscurve_name" : losscurve_name 
           }

    train(model, train_files, valid_files, maskpatterns, epochs, batchsize, info)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir")
    parser.add_argument("valid_dir")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collection',action="store_true")
    group.add_argument('--induction',action="store_true")

    parser.add_argument("-e", "--epochs", nargs='?', type=int, default=10, action="store", dest="EPOCHS")
    parser.add_argument("-b", "--batchsize", nargs='?', type=int, default=12, action="store", dest="BATCHSIZE")
    parser.add_argument("--model_name", nargs='?', type=str, action="store", dest="MODEL_NAME",
                        default="{}".format(datetime.datetime.now().strftime("_%d%m%Y-%H%M%S")))

    args = parser.parse_args()

    return (args.train_dir, args.valid_dir, args.collection, args.induction, args.EPOCHS,
            args.BATCHSIZE, args.MODEL_NAME)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)