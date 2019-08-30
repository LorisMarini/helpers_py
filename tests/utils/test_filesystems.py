from helpers.utils import *
from helpers.imports import *


def test_buckets_connection():

    # Get all AirflowBuckets you can from BUCKETS env variable
    all_buckets = airflow_bucket_from_id()

    # Try to ls (read) all of them
    for bucket in all_buckets:
        filesystem = bucket.fs()
        assert filesystem.ls(bucket.name)


# TODO filesystem_for_path(path)

# TODO file_paginate(file_object, chunk_size=10)

# TODO file_last_line(file_name)

# TODO wc(filename)
