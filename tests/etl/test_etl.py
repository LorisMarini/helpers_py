from helpers_py.utils import *


def test_etl_object():

    bucket_ids = available_bucket_ids()

    for bid in bucket_ids:
        obj = ETLObject(bucket_id=bid,
                        insight="test",
                        stage="test",
                        version="v1",
                        date_time=pendulum.datetime(2000, 1, 1),
                        name="all",
                        ext=".parquet")

        _ = obj.make_collection(days=90, direction="past")
        _ = obj.make_collection(days=90, direction="future")
        _ = obj.extract(nosrc_ok=True)
