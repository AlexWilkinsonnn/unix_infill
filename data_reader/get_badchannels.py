import argparse
import uproot
import numpy as np

def main(input_file):

    file = uproot.open(input_file)

    for idx, key in enumerate(file["dec"]["Truth"].keys()):
        z, xy = file["dec"]["Masked"][key].numpy()
        x, y = xy[0][0][1:], xy[0][1][1:]

        bad_channels = [ i - 1 for i in x if np.count_nonzero(z[int(i) - 1,:]) == 0 ]
        print('{}: {}'.format(key, bad_channels))


def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file")

    args = parser.parse_args()

    return (args.input_file,)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)