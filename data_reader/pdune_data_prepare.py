"""
Prepare training data in numpy format from TH2s.
"""

import os, argparse
import uproot
import numpy as np
from matplotlib import pyplot as plt


def root_to_numpy(filepath, saveloc, plane):
    """
    mode = 1: Get truth images (used for training data)
    mode = 2: Get truth images along with masked images (used for test data)
    """
    file = uproot.open(filepath)

    for idx, key in enumerate(file["dec"]["Truth"].keys()):
        z, _ = file["dec"]["Truth"][key].numpy()

        # print(z.dtype)
        # plt.imshow(z.T, aspect='auto', vmin=-20, vmax=20, cmap='coolwarm')
        # plt.show()

        with open(os.path.join(saveloc, "{}.npy".format(key[:-2])), "w") as f:
            np.save(f, z)

        if (idx + 1) % 50 == 0:
            print("{}/{}".format(idx + 1, len(file["dec"]["Truth"].keys())))

    # if mode == 2:
    #     for idx, key in enumerate(file["dec"]["Truth"].keys()):
    #         # if plane == "collection":
    #         #     if "TPC10" in key[:-2]: # No dead wires
    #         #         continue
    #         # elif plane == "induction":
    #         #     if ("TPC10" in key[:-2]) and ("plane1" in key[:-2]): # No dead wires
    #         #         continue

    #         z, xy = file["dec"]["Truth"][key].numpy()
    #         z_masked, xy_masked = file["dec"]["Masked"][key].numpy()

    #         np.savez(os.path.join(saveloc, "{}.npz".format(key[:-2])), z=z, z_masked=z_masked)

    #         if (idx + 1) % 20 == 0:
    #             print("{}/{}".format(idx + 1, len(file["dec"]["Truth"].keys())))



def main(input_file, output_dir, collection, truth):

    # if truth:
    #     mode = 1
    # else:
    #     mode = 2

    if collection:
        plane = "collection"
    else:
        plane = "induction"

    root_to_numpy(input_file, output_dir, plane)


def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")
    parser.add_argument("output_dir")

    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument("--collection", action='store_true')
    group1.add_argument("--induction", action='store_true')

    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument("--truth", action='store_true', help="Get truth images")
    group2.add_argument("--true_mask", action='store_true', help="Get truth images and masked images")

    args = parser.parse_args()

    return (args.input_file, args.output_dir, args.collection, args.truth)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
