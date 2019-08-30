from skepsi.utils.utils import *
from skepsi.utils.imports import *


def test_buckets_connection():

    # Get all AirflowBuckets you can from BUCKETS env variable
    all_buckets = airflow_bucket_from_id()

    # Try to ls (read) all of them
    for bucket in all_buckets:
        filesystem = bucket.fs()
        assert filesystem.ls(bucket.name)


# filesystem_for_path(path)

# file_paginate(file_object, chunk_size=10)

# file_last_line(file_name)

# wc(filename)
