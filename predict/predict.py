import os, sys, commands, time, argparse
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

import torch

sys.path.append("../model")
sys.path.append("../training")
from sparseinfill import SparseInfill
from loss_sparse_infill_induction import SparseInfillLoss #from loss_sparse_infill import SparseInfillLoss


def predict(model, test_dir, N, plane, trace, info):
    test_masked_lst, test_true_lst, test_loc_lst, filenames = [], [], [], []
    for filename in os.listdir(test_dir):
        if filename.endswith(".npy"):
            with open(os.path.join(test_dir, filename), "r") as f:
                    arr = np.load(f)

                    test_loc_lst.append(arr[:,[0,1]])
                    test_true_lst.append(arr[:,2].reshape(arr[:,2].size, 1))
                    test_masked_lst.append(arr[:,3].reshape(arr[:,3].size, 1))
                    filenames.append(filename)

        if len(test_loc_lst) >= N:
            break 

    print("Data Loaded")

    model.eval()

    for idx, locs in enumerate(test_loc_lst):
        print(filenames[idx])
        masked_tensor = torch.FloatTensor(test_masked_lst[idx])
        true_tensor = torch.FloatTensor(test_true_lst[idx])
        loc_tensor = torch.LongTensor(locs)

        masked_tensor.to(info["DEVICE"])
        true_tensor.to(info["DEVICE"])
        loc_tensor.to(info["DEVICE"])

        outputs = model(loc_tensor, masked_tensor, 1)

        loss = info["criterion"](outputs, true_tensor, masked_tensor)[5]
        print("Loss: {}".format(loss))

        coords = loc_tensor.detach().numpy()
        predict = outputs.detach().numpy()

        if plane=="induction":
            img_masked = np.zeros((1148, 6000))
            img_true = np.zeros((1148, 6000))
            img_pred = np.zeros((1148,6000))

            for iidx, coord in enumerate(coords - 1):
                img_true[coord[0], coord[1]] = true_tensor[iidx]
                img_masked[coord[0], coord[1]] = masked_tensor[iidx]
                if masked_tensor[iidx] == 0:
                    img_pred[coord[0], coord[1]] = predict[iidx] 
            dead_ch = [ idx for idx, col in enumerate(img_masked) if np.all(col == 0) ]
            print('Dead channles: {}'.format(dead_ch))

            # img_infill = np.ma.masked_array(img_pred, img_masked != 0)
            # img_masked = np.ma.masked_array(img_masked, img_masked == 0)

            # fig, ax = plt.subplots()
            # fig.set_size_inches(16, 10)
            # im1 = ax.imshow(img_masked.T, aspect='auto', cmap='coolwarm', vmin=-30, vmax=30, interpolation='None')
            # im2 = ax.imshow(img_infill.T, aspect='auto', cmap='PRGn', vmin=-30, vmax=30, interpolation='None')
            # plt.show()

            if trace:
                for ch in dead_ch:
                    print('Wire {}'.format(ch))
                    tick_adc_true = img_true[ch, :]
                    tick_adc_pred = img_pred[ch, :]
                    tick = np.arange(1, 6001) 

                    plt.rc('font', family='serif')
                    # plt.hist(tick, bins=len(tick), weights=tick_adc_true_left, histtype='step', label='Network left', linewidth=0.7)
                    # plt.hist(tick, bins=len(tick), weights=tick_adc_true_right, histtype='step', label='Network right', linewidth=0.7)
                    plt.hist(tick, bins=len(tick), weights=tick_adc_true, histtype='step', label='True', linewidth=0.7)
                    plt.hist(tick, bins=len(tick), weights=tick_adc_pred, histtype='step', label='Network', linewidth=0.7)
                    plt.xlim(1,6001)
                    plt.xlabel('time tick', fontsize=20)
                    plt.ylabel('total ADC in dead regions', fontsize=20)
                    plt.title('Wire {}'.format(ch))
                    ax = plt.gca()
                    ax.tick_params(axis='both', which='major', labelsize=16)
                    ax.tick_params(axis='both', which='minor', labelsize=16)
                    tx = ax.yaxis.get_offset_text()
                    tx.set_fontsize(20)
                    handles, labels = ax.get_legend_handles_labels()
                    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
                    plt.legend(handles=new_handles, labels=labels, prop={'size': 20})
                    plt.show()

        if plane == "collection":
            # implement
            pass


def main(weights, test_dir, collection, induction, n, traces):
    DEVICE = torch.device("cpu")

    imgdims = 2
    ninput_features  = 16
    noutput_features = 16
    nplanes = 5
    reps = 1
    filter_size = 3
    downsample = [2,2]
    model = SparseInfill((6000, 6000), reps, filter_size, ninput_features, noutput_features, nplanes, show_sizes=False, downsample=downsample).to(DEVICE)

    #Need to use the other loss if working with collection plane
    criterion = SparseInfillLoss().to(device=DEVICE)

    pretrained_dict = torch.load(weights, map_location="cpu")
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)

    info = {
            "DEVICE" : DEVICE,
            "criterion" : criterion
           }

    if induction:
        predict(model, test_dir, n, "induction", traces, info)

    elif collection:
        predict(model, test_dir, n, "collection", traces, info)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("weights")
    parser.add_argument("test_dir")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collection',action="store_true")
    group.add_argument('--induction',action="store_true")

    parser.add_argument("-n", nargs="?", type=int, action="store", default=5, dest="N")
    parser.add_argument("-t", "--traces", action="store_true")

    args = parser.parse_args()

    return (args.weights, args.test_dir, args.collection, args.induction, args.N, args.traces)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)



