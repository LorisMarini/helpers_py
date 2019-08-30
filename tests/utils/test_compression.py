from helpers.utils import *
from helpers.imports import *


def test_tar_untar():
    """Call to test the tar_and_compress_files() and untar_and_uncompress_path()"""
    orig_dir = "/tmp/orig/"
    comp_dir = "/tmp/comp/"
    uncomp_dir = '/tmp/uncomp/'

    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(uncomp_dir, exist_ok=True)

    # Prepare dummy txt files
    for i in range(3):
        with open(f"/tmp/orig/file_{i}.txt", 'w') as file:
            file.write("Skepsi is awesome.")

    # Grub files abspaths
    in_a = [file for file in glob.glob(f"{orig_dir}*.txt")]

    compression = "bz2"
    archive_name = f"act_raw_archive.tar.{compression}"
    tar_abspath = os.path.join(comp_dir, archive_name)

    # Compress and act_raw_archive
    tar_and_compress_files(compression=compression, src_list=in_a, dest=tar_abspath)

    # Unarchive and Uncompress
    untar_and_uncompress_path(compression=compression, src=tar_abspath, dest=uncomp_dir)

    return True
