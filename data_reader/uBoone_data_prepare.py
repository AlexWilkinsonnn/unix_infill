import uproot
import numpy as np
import awkward
from matplotlib import pyplot as plt


def root_to_numpy(filepath, number):
    """
    """
    file = uproot.open(filepath)
    adc_tree = file["sparseimg_ADC_tree"]
    ev_adc_lsts = adc_tree["sparseimg_ADC_branch"]["_image_v"]["_image_v._pixelarray"].array(entrystart=0, entrystop=number)
    adcMasked_tree = file["sparseimg_ADCMasked_tree"]
    ev_adcMasked_lsts = adcMasked_tree["sparseimg_ADCMasked_branch"]["_image_v"]["_image_v._pixelarray"].array(entrystart=0, entrystop=number)

    for ev in range(len(ev_adc_lsts)):
        for plane in range(3): # I think this index is plane
            x = np.array(ev_adc_lsts[ev][plane][0::3]).reshape(-1,1)
            y = np.array(ev_adc_lsts[ev][plane][1::3]).reshape(-1,1)
            z = np.array(ev_adc_lsts[ev][plane][2::3]).reshape(-1,1)
            z_masked = np.array(ev_adcMasked_lsts[ev][plane][2::3]).reshape(-1,1)

            locations_features = np.column_stack((x, y, z, z_masked))

            # img = np.full((512, 496), -100)
            # img_masked = np.full((512, 496), -100)

            # for i, x_coord in enumerate(locations_features[:,0]):
            #     img[int(x_coord), int(locations_features[i,1])] = locations_features[i,2]
            #     img_masked[int(x_coord), int(locations_features[i,1])] = locations_features[i,3]

            # plt.imshow(img, cmap='coolwarm', aspect='auto')
            # plt.colorbar()
            # plt.show()

            # plt.imshow(img_masked, cmap='coolwarm', aspect='auto')
            # plt.colorbar()
            # plt.show()

            with open("/unix/dune/awilkinson/infill_work/uBoone_sparsedata/npy_format/Event{}plane{}.npy".format(ev, plane), 'w') as f:
                np.save(f, locations_features)
        
        if (ev + 1) % 50 == 0:
            print("{}/{}".format(ev + 1, len(ev_adc_lsts)))
            
        if ev*3 >= number:
            break

def main():
    root_to_numpy('/unix/dune/awilkinson/infill_work/uBoone_sparsedata/sparseinfill_data_train.root', 2700)


if __name__ == "__main__":
    main()



