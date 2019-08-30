from skepsi.utils.utils import *
from skepsi.utils.imports import *


def test_check_type():
    assert check_type(2.3, float)
    assert check_type(2, int)

    with pytest.raises(TypeError):
        check_type(2, float)
        check_type("skepsi", [22, "no-type"])


def test_check_file(file_local):

    # With a sample txt file it should pass
    check_file(abspath=file_local)
    check_file(abspath=file_local, ext='.test')

    with pytest.raises(ValueError):
        check_file(abspath='')   # abspath empty string
        check_file(abspath='~')  # abspath not a file
        check_file(abspath=file_local, ext='.csv')  # Wrong extension

    with pytest.raises(TypeError):
        check_file(abspath=44)   # abspath is not a string


def test_check_path_remote(buckets):

    for bucket in buckets:
        assert check_path_remote(bucket.name)

# check_dir(directory)

# check_numeric(array, numeric_kinds='buifc')

# check_json_files(jsonnames, last_line=True)

# check_json_file(json_file, log_file=None, last_line=True)
