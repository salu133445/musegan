"""Store a numpy array to shared memory via SharedArray package.
"""
import os.path
import argparse
import numpy as np
import SharedArray as sa

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help="Path to the data file.")
    parser.add_argument('--name', help="File name to save in SharedArray. "
                                       "Default to use the original file name.")
    parser.add_argument('--prefix', help="Prefix to the file name to save in "
                                         "SharedArray. Only effective when "
                                         "`name` is not given.")
    args = parser.parse_args()
    return args.filepath, args.name, args.prefix

def main():
    """Main function"""
    filepath, name, prefix = parse_arguments()

    data = np.load(filepath)

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
        if prefix is not None:
            name = prefix + '_' + name

    sa_array = sa.create(name, data.shape, data.dtype)
    np.copyto(sa_array, data)

    print("Successfully saved: {}, {}, {}".format(name, data.shape, data.dtype))

if __name__ == '__main__':
    main()
