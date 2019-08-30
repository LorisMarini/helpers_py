from helpers.utils import *
from helpers.imports import *
from helpers.etl import *


def test_path_exists(file_local, dir_local):

    assert path_exists(file_local, filesystem=None)
    assert path_exists(dir_local, filesystem=None)


def test_paths(tmpdir, buckets, list_numeric):

    test_id = "test-paths"
    file_name = f'list-numeric.pkl'

    # Create local empty directory
    empty_dir = tmpdir.join("empty-dir").strpath
    os.makedirs(empty_dir, exist_ok=True)

    # Set paths
    p_locals = [tmpdir.join(file_name).strpath]
    p_remotes = [f'{bucket.tests}/{test_id}/{file_name}' for bucket in buckets]

    # all paths
    paths = p_locals + p_remotes

    # ------------- PATH EXISTS ------------------

    # Create all objects
    _ = [obj_serialize(list_numeric, saveas=path, verbose=False) for path in paths]

    # Check that they exist
    _ = [path_exists(path, filesystem=None) for path in paths]

    # ------------- PATH IS EMPTY ------------------

    # path_is_empty should return False for local path
    assert not path_is_empty(os.path.dirname(p_locals[0]))
    assert path_is_empty(empty_dir)

    with pytest.raises(ValueError):
        for path in p_remotes:
            path_is_empty(path)

    # ------------- PATH IS DIR ------------------

    for path in p_locals + [empty_dir]:
        assert path_is_dir(os.path.dirname(path))

    with pytest.raises(ValueError):
        for path in p_remotes:
            path_is_dir(path)

    # ------------- PATH_DESCRIBE ------------------

    # Get all directories (local and remote)
    dirs = [os.path.dirname(path) for path in paths]

    for dir in dirs:
        path_describe(dirname=dir, sort_by="size", ascending=False, minimal=True, extensions=[], attribute=False, log=True)

# TODO path_describe_many(dirlist, minimal=True, sort_by="size", ascending=False, log=True, extensions=[])

# TODO path_consolidate(file_abs_path)

# TODO path_dirs_from_paths(paths, log=True)

# TODO path_bucket_id(path)

# TODO path_splitext(path)
