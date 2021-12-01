import time
import torch
import numpy as np
from matplotlib import pyplot as plt

from model import unet, unet_small, unet_small_collect


def main():
    DEVICE = torch.device("cpu")
    model = unet_small().to(DEVICE)
    model.eval()
    pretrained_dict = torch.load('dsetinduct800randmaskdense_small_68epochs.pth', map_location="cpu")
    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict)

    arr = np.load('/unix/dune/awilkinson/infill_work/pdune_densedata/numpy/induction/valid/Ev2515Run22866473SRun98_plane1_TPC6.npy').T[:,0:800]
    maskpattern = [114, 273, 401]#914, 1073]
    # maskpattern = [55, 66, 78, 81, 89, 238, 370]
    arr[:, maskpattern] = 0
    example_img = torch.FloatTensor(arr.reshape(1, *arr.shape))
    example_img = torch.stack([example_img])

    print("Python")
    with torch.no_grad(): 
        start = time.time()
        model(example_img)
        end = time.time()
        print(end - start)
    
    with torch.no_grad():  
        traced_model = torch.jit.trace(model, example_img)
    traced_model.save('unetdense_induct800_small_68e_070421.pt')
    print(traced_model)

    loaded_model = torch.jit.load('unetdense_induct800_small_68e_070421.pt')
    
    print("Loaded TorchScript")
    with torch.no_grad():
        start = time.time() 
        output = loaded_model(example_img)
        end = time.time()
        print(end - start)
    out_img = output.detach().numpy()[0, 0, :, :].T
    in_img = example_img.detach().numpy()[0, 0, :, :].T

    mask = np.zeros_like(in_img)
    mask[np.array(maskpattern), :] = 1
    img_masked = np.ma.masked_array(in_img, mask)
    img_infill = np.ma.masked_array(out_img, np.logical_not(mask))

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    im1 = ax.imshow(img_masked.T, aspect='auto', cmap='coolwarm', vmin=-30, vmax=30, interpolation='None')
    im2 = ax.imshow(img_infill.T, aspect='auto', cmap='PRGn', vmin=-30, vmax=30, interpolation='None')
    plt.show()


if __name__ == "__main__":
    main()