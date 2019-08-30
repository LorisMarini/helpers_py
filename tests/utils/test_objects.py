from skepsi.utils.utils import *
from skepsi.utils.imports import *


def test_obj_serialize(tmpdir, buckets, list_numeric, dataframe_1):

    # ------------------- List  --------------------

    file_name = 'list-numeric.pkl'
    saveas = tmpdir.join(file_name).strpath

    # Serialize
    obj_serialize(list_numeric, saveas=saveas, verbose=True)

    # De-serialize
    _ = obj_deserialize(saveas, log=True, verbose=False)

    for bucket in buckets:

        print(f"Testing on {bucket.name}...")
        saveas = f'{bucket.tests}/test-obj-serialize/{file_name}'

        # Serialize
        obj_serialize(list_numeric, saveas=saveas, verbose=False)

        # Deserialize
        _ = obj_deserialize(saveas, log=True, verbose=False)

    # ------------------- Dataframe  --------------------

    file_name = 'dataframe-1.pkl'
    obj_serialize(dataframe_1, saveas=tmpdir.join(file_name).strpath, log=True)

    # Dataframe Remote
    for bucket in buckets:
        print(f"Testing on {bucket.name}...")
        saveas = f'{bucket.tests}/test-obj-serialize/{file_name}'
        obj_serialize(list_numeric, saveas=saveas, verbose=False)

    with pytest.raises(ValueError):
        # Serialization supports only pkl files if force_ext=True
        obj_serialize(dataframe_1, saveas=tmpdir.join('dataframe-1.ext').strpath, force_ext=True)


def test_obj_extract_load(tmpdir, buckets, list_numeric):

    test_id = "test-obj-extract-load"
    file_name = f'list-numeric.pkl'

    # Set paths
    p_local = tmpdir.join(file_name).strpath
    remote_paths = [f'{bucket.tests}/{test_id}/{file_name}' for bucket in buckets]
    missing_remote_paths = [f"{bucket.tests}/random-path-that-does-not-exist"  for bucket in buckets]

    # Serialize object locally
    obj_serialize(list_numeric, saveas=p_local, verbose=False)

    # Test Loading
    for p_remote in remote_paths:
        kwargs = {}
        obj_load(p_local, p_remote, **kwargs)

    # Test Extraction
    for p_remote in remote_paths:
        obj_extract(p_remote, p_local, use_cache=True, log=True)

    # do again with dest_overright=False (should pass and do nothing)
    obj_extract(remote_paths[0], p_local, use_cache=False, log=True)

    for p_remote in missing_remote_paths:
        # Test switch missing_src_ok
        obj_extract(src=p_remote, dest=p_local, nosrc_ok=True)

    with pytest.raises(ValueError):
        for p_remote in missing_remote_paths:
            # Test switch missing_src_ok
            obj_extract(src=p_remote, dest=p_local, nosrc_ok=False)
