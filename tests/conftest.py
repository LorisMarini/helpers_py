"""
File containing all fixtures used in different test files.
NOTE: This file name must be conftest.py as detailed here:
https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions
"""
from helpers.etl import *


@pytest.fixture(scope="session")
def file_local(tmpdir_factory):
    """Absolute path to a simmple file on local filesystem"""

    fn = tmpdir_factory.mktemp("data").join("test_file.test")
    fn.write("Hi Pytest!")
    output = fn.strpath
    return output


@pytest.fixture(scope="session")
def file_s3(file_local):
    remote_test_dir = airflow_bucket_from_id("s3://").tests
    remote_path = "/".join([remote_test_dir, "test_file.test"])
    fs = filesystem_s3()
    fs.put(file_local, remote_path)
    return remote_path


@pytest.fixture(scope="session")
def file_gs(file_local):

    remote_test_dir = airflow_bucket_from_id("gs://").tests
    remote_path = "/".join([remote_test_dir, "test_file.test"])
    fs = filesystem_gcs()
    fs.put(file_local, remote_path)
    return remote_path


@pytest.fixture(scope="session")
def dir_local(tmpdir_factory):
    """Absolute path to a directory on local filesystem"""

    path = tmpdir_factory.mktemp("data")
    output = path.strpath
    return output


@pytest.fixture(scope="session")
def dataframe_1(tmpdir_factory):
    """Absolute path to a directory on local filesystem"""

    df = pd.DataFrame({'integers': [1, 2, 3, 4],
                       'strings': ['one', 'two', 'three', 'four'],
                       'floats': [1.0, 2.0, 3.0, 4.0],
                       'dates': ["2019-02-01", "2019-03-01", "2019-04-01", "2019-05-01"],
                       'dicts': [{"a": 23, "b": 'skepsi', "c": datetime.now()},
                                 {"a": 23.0, "b": 'skepsi', "c": datetime.now()},
                                 {"a": 23.0, "b": 'skepsi', "c": datetime.now()},
                                 {"a": 23.0, "b": 'skepsi', "c": datetime.now()}]})

    df["dates"] = pd.to_datetime(df["dates"])
    return df


@pytest.fixture(scope="session")
def dataframe_2(tmpdir_factory):
    """Absolute path to a directory on local filesystem"""

    df = pd.DataFrame({'integers': [1, 2, 3, 4],
                       'strings': ['one', 'two', 'three', 'four'],
                       'floats': [1.0, 2.0, 3.0, 4.0],
                       'dates': ["2019-02-01", "2019-03-01", "2019-04-01", "2019-05-01"]})

    df["dates"] = pd.to_datetime(df["dates"])
    return df


@pytest.fixture(scope="session")
def list_numeric():
    return [1, 2, 3, 4, 5]


@pytest.fixture(scope="session")
def buckets():
    return airflow_bucket_from_id()
