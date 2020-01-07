"""This script loads and saves an array to shared memory."""
import os.path
import argparse
import numpy as np
import SharedArray as sa


def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to the data file.")
    parser.add_argument(
        "--name",
        help="File name to save in SharedArray. "
        "Default to use the original file name.",
    )
    parser.add_argument(
        "--prefix",
        help="Prefix to the file name to save in "
        "SharedArray. Only effective when "
        "`name` is not given.",
    )
    parser.add_argument(
        "--dtype", default="bool", help="Datatype of the array. Default to bool."
    )
    args = parser.parse_args()
    return args.filepath, args.name, args.prefix, args.dtype


def main():
    """Main function"""
    filepath, name, prefix, dtype = parse_arguments()

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
        if prefix is not None:
            name = prefix + "_" + name

    print("Loading data from '{}'.".format(filepath))
    if filepath.endswith(".npy"):
        data = np.load(filepath)
        data = data.astype(dtype)
        try:
            sa_array = sa.create(name, data.shape, data.dtype)
        except FileExistsError:
            response = ""
            while response not in ["yes", "no"]:
                response = input(
                    "Existing array (also named "
                    + name
                    + ") was found. Replace it? (Respond with 'yes' or 'no') "
                )
            if response == "yes":
                sa.delete(name)
                sa_array = sa.create(name, data.shape, data.dtype)
            else:
                raise e
        finally:
            print("Saving data to shared memory...")
            np.copyto(sa_array, data)
    else:
        with np.load(filepath) as loaded:
            try:
                sa_array = sa.create(name, loaded["shape"], dtype)
            except FileExistsError as e:
                response = ""
                while response not in ["yes", "no"]:
                    response = input(
                        "Existing array (also named "
                        + name
                        + ") was found. Replace it? (Respond with 'yes' or "
                        + "'no') "
                    )
                if response == "yes":
                    sa.delete(name)
                    sa_array = sa.create(name, loaded["shape"], dtype)
                else:
                    raise e
            finally:
                print("Saving data to shared memory...")
                sa_array[[x for x in loaded["nonzero"]]] = True

    print(
        "Successfully saved: (name='{}', shape={}, dtype={})".format(
            name, sa_array.shape, sa_array.dtype
        )
    )


if __name__ == "__main__":
    main()
