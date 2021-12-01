import os,sys,commands
import shutil
import time
import traceback
import numpy as np
from matplotlib import pyplot as plt

import torch

sys.path.append("../model")
sys.path.append("../training")
from sparseinfill import SparseInfill
from loss_sparse_infill import SparseInfillLoss

DEVICE = torch.device("cpu")

imgdims = 2
ninput_features  = 16
noutput_features = 16
nplanes = 5
reps = 1
model = SparseInfill([6000, 480], reps, ninput_features, noutput_features, nplanes, show_sizes=False).to(DEVICE)

criterion = SparseInfillLoss().to(device=DEVICE)

pretrained_dict = torch.load("dset1_newloss_5epochs.pth", map_location="cpu")

pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}

model.load_state_dict(pretrained_dict)

test_masked_lst, test_true_lst, test_loc_lst, filenames = [], [], [], []
directory = "/unix/dune/awilkinson/infill_work/pdune_sparsedata/test"
for filename in os.listdir(directory):
    if "Event388_TPC6" in filename:
        with open(os.path.join(directory, filename), "r") as f:
                arr = np.load(f)
                test_loc_lst.append(arr[:,[0,1]])
                test_true_lst.append(arr[:,2].reshape(arr[:,2].size, 1))
                test_masked_lst.append(arr[:,3].reshape(arr[:,3].size, 1))
                filenames.append(filename)
    
    if len(test_loc_lst) == 1:
        break 

print("Data Loaded")

model.eval()

for idx, locs in enumerate(test_loc_lst):
    masked_tensor = torch.FloatTensor(test_masked_lst[idx])
    true_tensor = torch.FloatTensor(test_true_lst[idx])
    loc_tensor = torch.LongTensor(locs)

    masked_tensor.to(DEVICE)
    true_tensor.to(DEVICE)
    loc_tensor.to(DEVICE)

    outputs = model(loc_tensor, masked_tensor, 1)

    loss = criterion(outputs, true_tensor, masked_tensor)[5]
    print("Loss: {}".format(loss))

    coords = loc_tensor.detach().numpy()
    predict = outputs.detach().numpy()

    # for idx, val in enumerate(true_tensor.numpy()):
    #     if val == 0.0:
    #         print('{} : {}'.format(val, predict[idx]))

    img_pred = np.zeros((480, 6000))
    img_masked = np.zeros((480, 6000))
    img_true = np.zeros((480, 6000))

    for iidx, coord in enumerate(coords - 1):
        img_true[coord[0], coord[1]] = true_tensor[iidx]
        img_masked[coord[0], coord[1]] = masked_tensor[iidx]
        if masked_tensor[iidx] == 0:
            img_pred[coord[0], coord[1]] = -predict[iidx]
        else:
            img_pred[coord[0], coord[1]] = predict[iidx]

    for i in range(480):
        img_pred[i,480] = 10000
        img_pred[i,481] = 10000

    plt.imshow(img_pred.T, cmap='coolwarm', aspect='auto', vmin=-100, vmax=100)
    plt.title('Prediction')
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches(16, 10)
    plt.savefig('/unix/dune/awilkinson/infill_work/images/480line.png', dpi=800)
    plt.clf()
