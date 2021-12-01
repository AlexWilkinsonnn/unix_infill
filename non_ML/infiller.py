import numpy as np
import math
import os
from matplotlib import pyplot as plt

def ranges(lst):
    """
    """
    ranges_lst = []
    lower_idx = 0
    for idx, val in enumerate(lst[1:]):
        if (val - lst[idx]) == 1:
            pass
        else:
            ranges_lst.append((lst[lower_idx], lst[idx]))
            lower_idx = idx + 1
        if idx == len(lst[1:]) - 1:        
            if ranges_lst[-1][1] == lst[-2]:
                ranges_lst.append((val,val))
            else:
                ranges_lst.append((lst[lower_idx], val))

    return ranges_lst

def infiller(fileloc):
    """
    """
    with open(fileloc, "r") as f:
        arr = np.load(f)
        locs = arr[:,[0,1]].astype(int)
        true_adcs = arr[:,2].reshape(arr[:,2].size, 1)
        masked_adcs = arr[:,3].reshape(arr[:,3].size, 1)

    masked_channels = list(set([ coord[0] for idx, coord in enumerate(locs) if masked_adcs[idx][0] == 0 ]))
    masked_channels.sort()
    masked_channel_ranges = ranges(masked_channels)
    # print(masked_channel_ranges)

    img = np.zeros((6000,480))
    img_true = np.zeros((6000,480))
    for idx, coord in enumerate(locs - 1):
        img[coord[1], coord[0]] = masked_adcs[idx]
        img_true[coord[1], coord[0]] = true_adcs[idx]

    # plt.imshow(img, cmap='coolwarm', vmin=-20, vmax=20, aspect='auto')
    # plt.show()

    for channel_range in masked_channel_ranges:
        for idx in range(int(math.floor(((channel_range[1] - channel_range[0] + 1)/2)))):
            ch_L, ch_R = channel_range[0] - 1 + idx, channel_range[1] - 1 - idx

            averaged_channel_L = np.zeros((6000,1), dtype=np.float32)
            averaged_channel_R = np.zeros((6000,1), dtype=np.float32)
            avg_chs = [averaged_channel_L, averaged_channel_R]

            for idx, ch in enumerate([ch_L, ch_R]):
                for tick in range(6000):
                    sum, cnt = 0, 0

                    for j in range(-10, 11):
                        for i in [-1, 0, 1]:
                            try:
                                if img[tick + j, ch + i] > 10:
                                    sum += img[tick + j, ch + i]
                                    cnt += 1 
                                else:
                                    continue

                            except IndexError:
                                continue

                    if cnt == 0:
                        continue
                    else:
                        avg_chs[idx][tick, 0] = sum/cnt
        
                img[:,ch] = -np.abs(avg_chs[idx][:,0])

        if not (channel_range[1] - channel_range[0]) % 2: # odd number of dead channels
            ch_mid = int((channel_range[0] + channel_range[1])/2) - 1

            averaged_channel = np.zeros((6000,1), dtype=np.float32)

            for tick in range(6000):
                sum, cnt = 0, 0

                for j in range(-10, 11):
                    for i in [-1, 0, 1]:
                        try:
                            if img[tick + j, ch_mid + i] > 10:
                                sum += img[tick + j, ch_mid + i]
                                cnt += 1
                            else:
                                continue
                                
                        except IndexError:
                            continue

                if cnt == 0:
                    continue
                else:
                    averaged_channel[tick, 0] = sum/cnt

            img[:,ch_mid] = -np.abs(averaged_channel[:,0])

    out = np.zeros((2, 480, 6000))
    out[0] = img.T # To match network output shape
    out[1] = img_true.T
  
    return out 


def main():
    # infiller("/unix/dune/awilkinson/infill_work/pdune_sparsedata/true_mask/train/Event1742_TPC2.npy")

    directory = "/unix/dune/awilkinson/infill_work/pdune_sparsedata/special"
    for idx, filename in enumerate(os.listdir(directory)):
        infilled = infiller(os.path.join(directory, filename))
        with open("/unix/dune/awilkinson/infill_work/analysis/data/special/nonML_infill/id{}_{}.npy".format(idx, filename[:-4]), "w") as f:
            np.save(f, infilled)
        print(idx)


if __name__ == "__main__":
    main()
