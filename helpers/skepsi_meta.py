from skepsi.utils.utils import *


def get_skepsi_commit_info():

    repo = Repo(env_get('SKEPSI_DOCKER', required=True))

    # Extract the first commit on the log (last commit)
    commit = list(repo.iter_commits())[0]

    # Return git metadata
    meta = {"_skepsi_commit_hash": commit.hexsha,
            "_skepsi_commit_author": commit.author.name,
            "_skepsi_commit_time": str(commit.authored_datetime),
            "_skepsi_commit_message": commit.summary}

    return meta


def get_execution_time_info():
    """
    Builds a simple one key-value dict with the current time.
    Returns
    -------
    output  :   dict

    """
    # All servers are in UTC, set timezone to UTC
    pdts = pd.Timestamp(datetime.now(), tz="UTC")
    output = {"_skepsi_local_exectime": str(pdts)}

    return output


def skepsi_meta_dict():
    """
    Builds a dict with all metadata for Skepsi (combines different dicts)
    Returns
    -------
    all_meta    :   dict
                    Key-value pairs of metadata for binario
    """
    meta_to_merge = [get_skepsi_commit_info(), get_execution_time_info()]

    all_meta = {}
    for meta in meta_to_merge:
        all_meta.update(meta)

    # Validate metadata dictionary
    validate_metadata_dict(all_meta)

    return all_meta


def skepsi_add_meta_to_df(*, df, status, message, started_at=None, ended_at=None):
    """
    Extends the input table df, by joining with a customers table with as many
    rows as the input table (horizontal extension).
    Parameters
    ----------
    df          :   pd.DataFrame
    status      :   str
                    One of ["ok", "failed"]
    message     :   str
                    Additional info to explain the status.
    started_at  :   pd.Timestamp, None
                    Timestamp signalling the beginning of the execution
    ended_at    :   pd.Timestamp, None
                    Timestamp signalling the end of the execution
    Returns
    -------
    output      :   pd.DataFrame
    """

    # Type Check
    check_type(df, pd.DataFrame)
    check_type(status, str)
    check_type(message, str)
    check_type(started_at, [pd.Timestamp, type(None)])
    check_type(ended_at, [pd.Timestamp, type(None)])

    if status not in ['ok', 'failed']:
        raise ValueError(f"status = '{status}' invalid. Possible values are 'ok' and failed'")

    # Get execution metadata
    exec_meta = {'_exec_status': status,
                 '_exec_message': message,
                 '_exec_started_at': started_at,
                 '_exec_ended_at': ended_at}

    # Get code metadata
    code_meta = skepsi_meta_dict()

    # Merge dictionaries
    code_meta.update(exec_meta)

    if df.empty:
        output = pd.DataFrame(code_meta, index=[0])
    else:
        # Create a dataframe from dictionary with same index as input table
        meta_df = pd.DataFrame(code_meta, index=df.index)

        # Join df with meta_df on the indexes
        output = df.merge(meta_df, left_index=True, right_index=True)

    return output


def validate_metadata_dict(d):

    for key, value in d.items():
        try:
            check_type(key, str)
        except Exception:
            raise ValueError(f"The keys in a metadata dict must be strings, found {type(key)}")
        try:
            check_type(value, str)
        except Exception:
            raise ValueError(f"The values in a metadata dict must be strings, found {type(value)}")

    return True
