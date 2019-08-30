from helpers.imports import *

"""
---------------------------------------------- CHECKS -------------------------------------------
-------------------------------------------------------------------------------------------------
"""


def check_path_remote(path):
    """If path does not point to a bucket it throws an error."""

    # Get filesystem for the path
    fs = filesystem_for_path(path)

    if not fs:
        # Passed local path
        raise ValueError(f"expected path to a remote bucket, passed local path {path}")

    return True


def check_dir(directory):

    # output_dir must be a valid directory
    if not os.path.isdir(directory):
        raise ValueError(f'directory {directory} is not a valid directory.')


def check_type(obj, type_):
    """
    Check the type of an object (obj) and if this does not match returns a
    TypeError with a standard format: Expected type and received type.
    Parameters
    ----------
    obj     :   any
                The python object you want to check the type of

    type_   :   type(obj)
                The type expected for obj
    Returns
    -------
    True if type(obj) == type_, else raises an error.
    """
    if isinstance(type_, list):
        # Indicated more than one type

        # Validate type of indicated types
        valid_types = [isinstance(t, type) for t in type_]
        if not all(valid_types):
            message = f"When type_ is a list, each element should be a valid type object."
            report_message(message, verbose=True, log=True, level='info')
            raise TypeError(message)

        # Check that obj is a type of at least one of the indicated types
        checks = [isinstance(obj, t) for t in type_]
        if not any(checks):
            message = f" obj expected to be of types {str(type_)}, passed {type(obj)}"
            report_message(message, verbose=True, log=True, level='info')
            raise TypeError(message)

    elif isinstance(type_, type):
        if not isinstance(obj, type_):
            message = f" obj expected to be of types {str(type_)}, passed {type(obj)}"
            report_message(message, verbose=True, log=True, level='info')
            raise TypeError(message)
    else:
        raise TypeError(f"type_ is {type(type_)} and is not a valid type.")

    return True


def check_file(abspath, ext=None):
    """
    Raises ValueError if abspath does not exist.
    Parameters
    ----------
    abspath :   str
                absolute path to the file (with extension)
    ext :       str, None
                Makes sure the file has correct extension is specified.
    """
    check_type(abspath, str)
    check_type(ext, [str, type(None)])

    if not os.path.isfile(abspath):
        raise ValueError(f'File {abspath} does not exist. '
                         f'Specify a valid path to file.')

    if ext:
        # print(os.path)
        file_name, extension = os.path.splitext(str(abspath))
        if not extension == ext:
            raise ValueError(f'File {abspath} should be a {ext} file. '
                             f'Specify a valid path to a {ext} file.')


def check_numeric(array, numeric_kinds='buifc'):
    """
    Converts array to a NumPy array. If the type after conversion
    is any of those specified in numeric_kinds if passes, otherwise
    it raises an exception. Numpy NaN are floats, so they are numeric!

    Parameters
    ----------
    array : array-like
            The array to check.

    numeric_kinds :     string
                        indicates the numerical types:
                        b --> boolean
                        u --> unsigned integer
                        i --> integer
                        f --> float
                        c --> complex
    """
    numeric_kinds = set(numeric_kinds)
    kind = np.asarray(array).dtype.kind

    if kind not in numeric_kinds:
        raise TypeError(('Array is expected to be of type ' +
                         str(numeric_kinds) + ' When converted to NumPy array, not ' +
                         str(kind)))


def check_json_files(jsonnames, last_line=True):
    """
    Takes the list of names for the json files dumped from ElasticSearch,
    and runs check_json_file() on each of them to identify issues.
    Parameters
    ----------
    jsonnames
    last_line

    Returns
    -------

    """
    parentdir = '/media/loris/2d735f67-81b6-428f-ad82-d79dba395a47/'
    jsondir = os.path.join(parentdir, 'json_dump')
    json_logsdir = os.path.join(jsondir, 'logs')
    log_file = os.path.join(json_logsdir, 'json_logs.log')

    for json_name in jsonnames:
        json_name_full = os.path.join(jsondir, json_name)

        if os.path.isfile(json_name_full):
            print(json_name_full)
            check_json_file(json_name_full, log_file=log_file, last_line=last_line)

    print('Validation_complete.')
    return True


def check_json_file(json_file, log_file=None, last_line=True):
    '''
        Tries to load the json file line by line.
        If file corrupted or any error is found, it raises an error.
    '''
    try:
        if last_line:
            # only parse last line of json file
            json.loads(file_last_line(json_file))

        elif not last_line:
            # Parse entire json file
            with open(json_file) as file:
                for i, line in enumerate(file):
                    json.loads(line)

    except Exception as ex:
        if not last_line:
            print('Something went wrong on line #' + str(i) + ' of file ' + json_file)
        elif last_line:
            print('Corrupted last line in file ' + json_file)

        raise ValueError('json invalid.')



"""
-------------------------------------------- FILESYSTEMS ---------------------------------------
-------------------------------------------------------------------------------------------------
"""


def filesystem_gcs():
    """
    It will try to read the env variables S3_ACCESS_KEY_ID and S3_ACCESS_KEY_PASSWORD.
    If they are not set, it will attempt to use the default permissions for the machine
    (e.g. an AWS role for ec2 instances).
    Parameters
    ----------
    profile_name

    Returns
    -------
    fs      :   s3fs.core.S3FileSystem
                The filesystem to work with the bucket (https://s3fs.readthedocs.io/en/latest/)
    """

    creds_file = env_get("GOOGLE_APPLICATION_CREDENTIALS", required=True)

    # Get Google filesystem via gcsfs
    fs = gcsfs.GCSFileSystem(token=creds_file)

    return fs


def filesystem_s3(profile_name=None):
    """
    It will try to read the env variables S3_ACCESS_KEY_ID and S3_ACCESS_KEY_PASSWORD.
    If they are not set, it will attempt to use the default permissions for the machine
    (e.g. an AWS role for ec2 instances).
    Parameters
    ----------
    profile_name

    Returns
    -------
    fs      :   s3fs.core.S3FileSystem
                The filesystem to work with the bucket (https://s3fs.readthedocs.io/en/latest/)
    """
    check_type(profile_name, [type(None), str])

    with open(env_get("AWS_APPLICATION_CREDENTIALS"), 'r') as file:
        aws_creds = json.load(file)

    # Get filesystem
    fs = s3fs.S3FileSystem(key=aws_creds["AWS_USER_KEY_ID"],
                           secret=aws_creds["AWS_USER_KEY_PASSWORD"],
                           profile_name=profile_name,
                           s3_additional_kwargs={'ServerSideEncryption': 'AES256'})
    return fs


def filesystem_for_path(path):
    """
    If path is to a remote bucket, returns the filesystem to talk to it,
    otherwise returns none.
    Parameters
    ----------
    path
    """
    check_type(path, str)

    if path.startswith("s3://"):
        # Path to an S3 bucket
        fs = filesystem_s3()
        return fs
    elif path.startswith("gs://"):
        # Path to a GCS bucket
        fs = filesystem_gcs()
        return fs
    else:
        fs = None
        return fs


def file_paginate(file_object, chunk_size=10):
    """
    Lazy function (generator) to read a file piece by piece.

    Parameters
    ----------
    file_object :
                    A python hook to a currently open file

    chunk_size :    int
                    How many rows to read at a time.
    Returns
    -------
    Next chunk of data every time you call it (chunk_size at a time).
    """

    while True:
        data = file_object.readlines(chunk_size)
        if not data:
            break
        yield data


def file_last_line(file_name):
    """
    Reads the last line of a file without going through the entire content.
    https://www.quora.com/How-can-I-read-the-last-line-from-a-log-file-in-Python
    Parameters
    ----------
    file_name   :   str
                    Absolute path of the file on disk

    Returns
    -------
    output :    str
                The last line of fname
    """
    with open(file_name) as source:
        mapping = mmap.mmap(source.fileno(), 0, prot=mmap.PROT_READ)

    # Get last line
    output = mapping[mapping.rfind(b'\n', 0, -1) + 1:]

    # Return it
    return output


def wc(filename):
    """
    Executed bash command wc (word count) with the flag -l which
    returns the number of lines in a file.

    Parameters
    ----------
    filename :  str
                Absolute path to the file on disk.
    Returns
    -------
    lines :     int, None
                The number of lines in the file
    """
    lines = None
    try:
        lines = int(check_output(["wc", "-l", filename]).split()[0])
    except Exception as ex:
        print(ex)
    finally:
        return lines


"""
---------------------------------------- STATISTICS ---------------------------------------------
-------------------------------------------------------------------------------------------------
"""


def time_series_uniform(*, t_zero, max_days=1, size=100):
    """
    Generates a uniform distribution of `size` timestamps from `t_zero` to `max_days`.
    Parameters
    ----------
    t_zero      :   pd.Timestamp
                    Origin of time
    max_days    :   int
                    Number of days in the future
    size        :   int
                    Number of samples in the timeseries
    Returns
    -------
    output      :   pd.Series
                    Pandas Series of timestamps with an integer index
    """
    check_type(t_zero, pd.Timestamp)
    check_type(max_days, int)
    check_type(size, int)

    max_seconds = max_days * 86400
    min_seconds = 0

    seconds = np.random.randint(low=min_seconds, high=max_seconds, size=size)

    deltas = [pd.Timedelta(second, unit='s') for second in seconds]
    times = [t_zero + delta for delta in deltas]

    output = pd.Series(data=times, name="timestamp").sort_values().reset_index(drop=True)
    return output


def time_series_lognormal(*, t_zero, mean=0, sigma=1, size=100):
    """
    Same as time_series_uniform() but using a lognormal distribution.
    Parameters
    ----------
    t_zero      :   pd.Timestamp
                    Origin of time
    mean        :   numeric
                    Number of days in the future
    sigma       :   numeric
                    Uncertainty around the mean value
    size        :   int
                    Number of samples in the timeseries
    Returns
    -------
    output      :   pd.Series
                    Pandas Series of timestamps with an integer index
    """
    check_type(t_zero, pd.Timestamp)
    check_numeric(mean)
    check_numeric(sigma)
    check_type(size, int)

    seconds = 86400 * np.random.lognormal(mean=mean, sigma=sigma, size=size)

    deltas = [pd.Timedelta(second, unit='s') for second in seconds]
    times = [t_zero + delta for delta in deltas]

    output = pd.Series(data=times, name="timestamp").sort_values().reset_index(drop=True)
    return output


def time_series_hist_diffs(series, units='d'):
    """
    Plots the histograms of differences between timestamps.
    Parameters
    ----------
    series
    units

    Returns
    -------

    Examples
    --------
    ts_ln = time_series_lognormal(t_zero=t_zero, mean=1, sigma=1, size=100)
    time_series_hist_diffs(ts_ln, units='d')

    """
    if units == 'd':
        scaling = 1e9 * 60 * 60 * 24

    elif units == 'h':
        scaling = 1e9 * 60 * 60

    elif units == 'm':
        scaling = 1e9 * 60

    elif units == 's':
        scaling = 1e9

    else:
        raise ValueError(f"units can only be one of {['d', 'h', 'm', 's']}")

    to_numbers = pd.to_numeric(series).diff() / scaling

    to_numbers.hist(color='k', alpha=0.5, bins=50)

    return None


def estimate_kernel_cdf(kernel, N=1000, verbose=False):
    """

    TODO Generalize for a kernel that is not normalized between 0 and 1
    Parameters
    ----------
    kernel  :
    N       :
    verbose

    Returns
    -------
    result  :
                result.x.values
                result.cdf.values
    """

    cdf_x = np.linspace(0, 1, N)

    cdf = [kernel.integrate_box_1d(0, x) for x in cdf_x]

    cdf = np.array(cdf)

    d = {'x': cdf_x, 'cdf': cdf}
    result = pd.DataFrame(d).set_index('x', drop=False)

    message = 'estimate_kernel_cdf() successful.'
    report_message(message, verbose=verbose, log=True, level='info')

    return result


def series_kde_gaussian_kernel(series, mss):
    """
    Returns the KDE kernel (Guassian) of the Series with scipy.stats.gaussian_kde:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Parameters
    ----------
    series      :   pd.Seriees
    mss         :   int

    Returns
    -------
    kernel      :   scipy.stats.kde.gaussian_kde
    """
    check_type(series, pd.Series)
    check_type(mss, int)

    if not is_numeric_dtype(series):
        raise ValueError(f"series dtype is not numeric.")

    # Substitute zeros with missing value (numpy.nan)
    series[series == 0] = np.nan
    # Substitute np.infinite with missing value (numpy.nan)
    series.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop missing values
    dataset = series.dropna(axis=0, how='any').values

    if len(dataset) > mss:
        kernel = gaussian_kde(dataset)
    else:
        kernel = None

    return kernel


def evaluate_pdf(kernel, series):

    dataset = series.values
    dataset_pdf = kernel.pdf(dataset)

    return dataset_pdf


def df_ecdf(*, df, column, percent=True):
    """
    Returns an interactive plot with the empirical cumulative
    distribution function of a dataset.
    Parameters
    ----------
    df
    column
    percent

    Returns
    -------

    """
    check_type(df, pd.DataFrame)
    check_type(percent, bool)

    # Make sure column exists in df
    if column not in df.columns:
        raise ValueError(f"column {column} no in dataframe")

    # Make sure column contains numeric values
    check_numeric(df[column].iloc[0])

    # Get bins (sorted unique values in column)
    num_bins = list(pd.Series(df[column].unique()).sort_values())

    # Calculate histogram
    counts, bin_edges = np.histogram(df[column], bins=num_bins, density=True)

    # Cumulative sum of values in histogram
    cdf = np.cumsum(counts)

    if percent:
        # Convert histogram density values to percentage
        y = cdf / cdf[-1] * 100
        x = bin_edges[1:]
        df = pd.DataFrame(data=y, index=x, columns=["% <= x"])
        plot_title = f"Distribution of column {column} (ecdf -  %)"

    else:
        y = cdf / cdf[-1]
        x = bin_edges[1:]
        df = pd.DataFrame(data=y, index=x, columns=["P <= x"])
        plot_title = f"Distribution of column {column} (ecdf - probability)"

    df.index.name = column

    return df


def estimate_kernel_mean(kernel, x=None):
    """

    Parameters
    ----------
    kernel      :
    x           :   None

    Returns
    -------

    """
    # TODO IMPROVE KDE ALGORITHMS
    # Improve this using
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html

    if x is not None:
        np.sum(np.multiply(kernel.pdf(x), x)) / len(x)
    else:
        min_value = kernel.dataset.min()
        max_value = kernel.dataset.max()

        N = 10000
        x = np.linspace(min_value, max_value, N)
        mean = np.sum(np.multiply(kernel.pdf(x), x)) / N

    return mean


def estimate_kernel_variance(kernel):

    min_value = kernel.dataset.min()
    max_value = kernel.dataset.max()
    N = 10000
    x = np.linspace(min_value, max_value, N)

    m = estimate_kernel_mean(kernel)

    square_diffs = np.square(x - m)

    scaled_diffs = np.multiply(kernel.pdf(x), square_diffs)
    var = np.sum(scaled_diffs) / N

    return var


def get_x_from_pcdf(kernel, queried_cdf, verbose=False):
    """
    Returns x from an estimation of the kernel CDF(x).
    Parameters
    ----------
    kernel
    queried_cdf :   numeric
                    The numeric value corresponding to CDF(x)
    verbose

    Returns
    -------

    """
    check_numeric(queried_cdf)

    cdf = estimate_kernel_cdf(kernel)

    returned_cdf = get_closest(cdf.cdf, queried_cdf)

    queried_x = cdf.loc[cdf.cdf == returned_cdf].x.values[0]

    message = f'Asked x for CDF(x)={queried_cdf} ' \
              f'closest found is x={queried_x} for which CDF({queried_x})={returned_cdf}'

    report_message(message, verbose=verbose)

    return queried_x


"""
----------------------------------------- PANDAS-NUMPY ----------------------------------------
-------------------------------------------------------------------------------------------------
"""


def pd_memory_usage(pandas_obj, verbose=True, log=False):
    """
    Returns the memory usage of a pandas DataFrame or Series in units.
    Parameters
    ----------
    pandas_obj  :   pd.Series, pd.DataFrame
    Returns
    -------
    memory      :   float
                    Number representing the amount of memory used in {units}
    """
    if isinstance(pandas_obj, pd.DataFrame):
        # df is a DataFrame
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        # df is a Series
        usage_b = pandas_obj.memory_usage(deep=True)

    return usage_b


def pd_validate_timezone(timezone):

    check_type(timezone, str)

    try:
        # See if tz is a valid timezone
        pd.Series().tz_localize(tz=timezone)
        return True

    except Exception as ex:
        raise ex


def get_closest(series, value):
    """
    Get Series value closest to indicated value
    Parameters
    ----------
    series  :   pd.Series
    value   :

    Returns
    -------
    closest_value   :
    """
    closest_value = series.iloc[(series - value).abs().argsort()[0]]
    return closest_value


def normalize_ndarray(array, verbose=False):
    """
    Takes a Nx2 numpy array and returns the same but normalized across the rows.
    The normalization is done subtracting the mean and scaling according to the
    empirical standard deviation (np.std). If the standard deviation is zero,
    normalization is not performed.

    Parameters
    ----------
    array :     np.ndarray
                Array to normalize

    Returns
    -------
    normalized :    np.ndarray
                    Normalized array

    """

    # Check array is a numpy array
    check_type(array, np.ndarray)
    # Check kind of xy_array
    check_numeric(array)

    mu = np.zeros([2])
    sigma = np.zeros([2])

    normalized = array.copy()

    for i in range(array.shape[1]):

        slice = array[:, i]

        # Calculate the mean of the variable on the first axis of the array
        mu_i = np.mean(slice, axis=0)
        report_message(f'mu_i = {mu_i}', verbose=verbose)

        # Calculate the standard deviation of the variable on the first axis of the array
        sigma_i = np.std(slice, axis=0)
        report_message(f'sigma_i = {sigma_i}', verbose=verbose)

        mu[i] = mu_i
        sigma[i] = sigma_i

        if sigma_i != 0:
            # Return a normalized version of the data.
            normalized_array = (slice - mu_i) / sigma_i

            normalized[:, i] = normalized_array
            report_message(f'normalized_array = {normalized_array}', verbose=verbose)
        else:
            normalized[:, i] = normalized[:, i]

    return normalized


"""
# -------------------------------------- DATAFRAME --------------------------------------
Aims ...
# ---------------------------------------------------------------------------------------
"""


def df_to_text(*, df, saveas, force_ext=False, verbose=False, log=True, orient='index'):
    """
    Uses the correct API call to save the DataFrame to either a json or a csv file.
    Parameters
    ----------
    df          :   pd.DataFrame
                    Valid pandas DataFrame to export as a csv to disk.
    saveas      :   str
                    Absolute path where to save the object (.json). If None does not save.
    force_ext   :   bool
                    If True and saveas does not end with '.csv' raises error.
    verbose     :   bool
                    Increases output verbosity
    log         :   bool
    orient      :   str

    # TODO Implement df_from_text if you have time!
    """

    # ------ TYPE CHECKS -------
    check_type(df, pd.DataFrame)
    check_type(saveas, [type(None), str])
    check_type(force_ext, bool)
    check_type(verbose, bool)
    check_type(orient, str)

    # ------ VALIDATIONS -------

    if saveas is None:
        message = f'Object not saved.'
        report_message(message, verbose=verbose, log=log, level='info')
        return None

    allowed_extensions = [".json", ".csv", ".agg"]
    extension = path_splitext(saveas)[1]

    # Get filesystem for the path
    fs = filesystem_for_path(saveas)

    if fs:
        # Remote object
        if ".json" in extension:
            # Convert to json and encode
            df_as_bytes = df.to_json(None, orient=orient).encode(encoding='utf-8')
            encoding_message = f'Dataframe converted to json, encoded with utf-8'

        elif ".csv" in extension or ".agg" in extension:
            # Convert to csv and encode
            df_as_bytes = df.to_csv(None).encode(encoding='utf-8')
            encoding_message = f'Dataframe converted to csv, encoded with utf-8'
        else:
            raise ValueError("File extension not supported.")

        # Write bytes to the bucket
        with fs.open(saveas, 'wb') as f:
            # Write to bucket
            f.write(df_as_bytes)

        # Report on status
        report_message(f"{encoding_message} and uploaded to {saveas}", verbose=verbose, log=log, level='info')

    else:
        # Local path
        if ".json" in extension:
            # Save to json
            df.to_json(saveas, orient=orient)
            message = f'DataFrame converted to json and saved locally as {saveas}.'

        elif ".csv" in extension or ".agg" in extension:
            # Save to csv
            df.to_csv(saveas)
            message = f'DataFrame converted to csv and saved locally as {saveas}.'
        else:
            raise ValueError("File extension not supported.")

        # Report on status
        report_message(message, verbose=verbose, log=log, level='info')

    return True


def df_to_parquet(*, df, path,
                  file_scheme='simple',
                  compression="snappy",
                  object_encoding="infer",
                  get_size=True,
                  verbose=False,
                  log=True):
    """
    Saves the DataFrame to parquet file (locally or to s3) using the fastparquet engine.
    Parameters
    ----------
    df              :   pd.DataFrame
                        Valid pandas DataFrame to export as a csv to disk.
    path            :   str, None
                        Absolute path where to save the object (.json). If None does not save.
    verbose         :   bool
                        Increases output verbosity
    log             :   bool
    compression     :   str
    object_encoding :   dict, None
    file_scheme     :   str
    """

    # ------ TYPE CHECKS -------

    check_type(df, pd.DataFrame)
    check_type(path, [type(None), str])
    check_type(verbose, bool)
    check_type(object_encoding, [str, dict])
    check_type(file_scheme, str)
    check_type(compression, str)

    validate_comp_fastparquet(compression)

    if path is None:
        message = f'Object not saved.'
        report_message(message, verbose=verbose, log=log, level='info')
        return False

    if ".parquet" not in path:
        message = f'saving to a parquet file but .parquet not found in filename'
        report_message(message, verbose=verbose, log=log, level='warning')

    # Make sure you are not trying to save a dataframe with tz aware datetimeindex.
    # See https://github.com/dask/fastparquet/issues/433
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz:
        raise ValueError("Current version of fastparquet does not support tz-aware datetime indexes. "
                         "For more info on this issue see https://github.com/dask/fastparquet/issues/433")

    # Get filesystem for the path
    fs = filesystem_for_path(path)

    if fs:
        # Remote path
        # Write with the fastparquet
        fpq_write(path, df, open_with=fs.open, compression=compression,
                  file_scheme=file_scheme, object_encoding=object_encoding)

        # Report memory of the object in RAM (if requested)
        if get_size:
            try:
                size_bytes = fs.info(path)["Size"]
            except KeyError:
                size_bytes = fs.info(path)["size"]

            size_message = f"size {sizeof_fmt(size_bytes)}"
        else:
            size_message = ""
        # Report on status
        report_message(f"Table {size_message} wrote as parquet file at {path}",
                       verbose=verbose, log=log, level='info')
    else:
        # Local path
        df.to_parquet(path, engine="fastparquet", compression=compression,
                      file_scheme=file_scheme, object_encoding=object_encoding)

        # Report memory of the object in RAM (if requested)
        if get_size:
            size_bytes = os.path.getsize(path)
            size_message = f"size {sizeof_fmt(size_bytes)}"
        else:
            size_message = ""
        # Report on status
        report_message(f"Table {size_message} wrote as parquet file at {path}",
                       verbose=verbose, log=log, level='info')

    return True


def df_from_parquet(*, path, columns=None, verbose=False, get_size=True, log=True):
    """
    Compression is inferred.
    Parameters
    ----------
    path
    columns
    verbose
    get_size
    log

    Returns
    -------
    df      :   pd.DataFrame
    """
    # ------ TYPE CHECKS -------

    check_type(path, str)
    check_type(verbose, bool)
    check_type(columns, [type(None), list])

    if not path_exists(path):

        message = f"File {path} not found. skipping."
        report_message(message, level='warning', log=log)
        df = pd.DataFrame()
        return df

    try:
        # Get filesystem for the path
        fs = filesystem_for_path(path)

        if fs:
            # Remote path
            # Read with the fastparquet
            df = ParquetFile(path, open_with=fs.open).to_pandas()
        else:
            # Local path
            df = pd.read_parquet(path, engine='fastparquet', columns=columns)

    except OSError as ex:
        message = f"path {path} might not be a valid parquet file: failed to retrieve the '_metadata' directory."
        raise TypeError(message) from ex

    if get_size:
        size_bytes = pd_memory_usage(df, verbose=verbose, log=log)
        size_message = sizeof_fmt(size_bytes)
    else:
        size_message = ""

    # Report on status
    report_message(f"Table {size_message} read from parquet file at {path}",
                   verbose=verbose, log=log, level='info')
    return df


def df_to_google_sheet(*, df, name, sfn, page):
    """
    Writes a dataframe to a google sheets
    Parameters
    ----------
    df      :   pd.DataFrame
                Table to write to Google Sheet
    name    :   str
                Name of Google Sheet
    """

    check_type(df, pd.DataFrame)
    check_type(name, str)
    check_type(sfn, str)
    check_type(page, int)

    sfn_path = env_get("GOOGLE_SHEET_SUCCESS_SPREADSHEET")

    # Read the email field to who you are supposed to share the sheet
    with open(sfn_path, 'r') as creds:
        share_with = json.load(creds)["client_email"]

    try:
        # authorization
        gc = pygsheets.authorize(service_file=sfn_path)

    except Exception as ex:

        message = f"authorization failed for google sheet `{name}` with error `{ex}`. " \
                  f"Service file used {sfn_path}, make sure that the sheet has been " \
                  f"shared with {share_with}"

        raise ValueError(message)

    try:
        # open the google spreadsheet
        sh = gc.open(name)

    except pygsheets.SpreadsheetNotFound as ex:

        message = f"google sheet `{name}` not found. Service file used {sfn_path}, " \
                  f"make sure that the sheet has been shared with {share_with}"

        raise ValueError(message)

    # update the first sheet with df, starting at cell B2.
    wks = sh[page].set_dataframe(df, (1, 1), fit=True)

    return True


def df_from_google_sheet(*, name, sfn, page=0):
    """
    Uses pygsheets module to read data from an authorized Google Sheet.
    Parameters
    ----------
    name
    sfn         :   str
                    Name of the google service file associated with this sheet
    page        :   int
                    Page number to read
    Returns
    -------
    output      :   pd.DataFrame
                    Table of data as read from google sheet
    Examples
    --------
    sfn="creds_success_spreadsheet.json"
    df_from_google_sheet(name=, sfn=sfn, page=0)

    """
    check_type(name, str)
    check_type(sfn, str)
    check_type(page, int)

    # Read the email field to who you are supposed to share the sheet
    with open(sfn, 'r') as creds:
        share_with = json.load(creds)["client_email"]

    try:
        # authorization
        gc = pygsheets.authorize(service_file=sfn)

    except Exception as ex:

        message = f"authorization failed for google sheet `{name}` with error `{ex}`. " \
                  f"Service file used {sfn}, make sure that the sheet has been " \
                  f"shared with {share_with}"

        raise ValueError(message)

    try:
        # open the google spreadsheet
        sh = gc.open(name)

    except pygsheets.SpreadsheetNotFound as ex:

        message = f"google sheet `{name}` not found. Service file used {sfn}, " \
                  f"make sure that the sheet has been shared with {share_with}"

        raise ValueError(message)

    # Open Google Sheet and load 5th page table into a pandas DataFrame
    output = sh[page].get_as_df()

    return output


def df_types_table(df):
    """
    Prints a table with names of df columns and their dtype (str).
    Parameters
    ----------
    df          :   pd.DataFrame
    Returns
    -------
    type_table  :   pd.DataFrame
    """
    check_type(df, pd.DataFrame)
    type_table = df.dtypes.to_frame().reset_index().rename(columns={0: "type", "index": "column"})
    type_table["type"] = type_table["type"].astype(str)

    return type_table


def df_fill_missing_values(df, filler=-1):
    """
    Describes policies to fill missing values (NaN or NaT),
    across the columns of the dataframe.
    Parameters
    ----------
    df      :   pd.DataFrame
    filler  :   obj

    Returns
    -------
    filled  :   pd.DataFrame
    """
    filled = df.fillna(filler)
    return filled


def df_remove_nan(df, how='any', axis=0, inplace=False, verbose=False):
    """
    Cleans NaN from a pandas Dataframe and reports on the effect of operation.
    """
    # sample_size before removing NaN
    len_before_dropna = len(df)

    # Drop NaN
    df_cleaned = df.dropna(axis=axis, how=how, inplace=inplace)

    # Difference in length
    len_diff = len_before_dropna - len(df_cleaned)

    if len_diff >= 1:
        # Inform of missing values
        message = f'{len_diff} lines on {len_before_dropna} ' \
                  f'have missing values. Will discard them.'

        report_message(message, verbose=verbose, log=True, level='info')

    return df_cleaned


def df_list_unique(df, column='instanceId', verbose=False, mask=None):

    instances = []
    if column not in df:
        raise ValueError('{0} is not in dataframe.'.format(column))
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be a pandas DataFrame')
    if mask is not None:
        if not isinstance(mask, pd.Series):
            raise ValueError('mask must be a pandas Series object.')
        elif mask.dtypes.name != 'bool':
            raise ValueError('mask must be boolean.')

    if mask is not None:
        instances = df[mask][column].dropna().unique()
    else:
        instances = df[column].dropna().unique()

    message = 'Found ' + str(len(instances)) + ' unique ' + column + '.'
    report_message(message, verbose=verbose)

    return instances


def df_largest_unique(df, column='action', dropna=True, verbose=False):
    '''
        1. Take a dataframe as input, and a column name.
        2. Calculates unique values in column
        3. Selects the column value most occurring
        4. Extracts dataframe with that column value if return_df=True
    '''
    if isinstance(column, list):
        raise ValueError('')
    if not isinstance(column, str):
        raise ValueError('')

    if not isinstance(df, pd.DataFrame):
        raise ValueError('')

    if not len(df) > 0:
        raise ValueError('DataFrame cannot be empty.')

    # Create a Series with values from the selected column and grab unique values
    series = df[column].dropna() if dropna else df[column]
    uniques = series.unique()

    # Create a dict from uniques, where keys are the elements in uniques, and values their length
    counts = dict((key, len(df[df[column] == value])) for key, value in enumerate(uniques))

    # Get the key used the most
    position = max(counts.items(), key=operator.itemgetter(1))[0]
    value = uniques[position]
    count = counts[position]

    length_original = len(df)
    length_most_frequent = len(df[df[column] == value])
    percentage = np.round(100 * (length_most_frequent / length_original), 2)

    reduced_df = df[df[column] == value]

    message = ('Most occurring ' + column + ' is ' + str(value) + ' and represents ' +
               str(percentage) + '% of the original dataframe.')
    report_message(message, verbose=verbose)

    return value, count, reduced_df


def df_group_columns(df, groupname='new_group', column_names=None, copy=True):
    """
    This function assigns an extra layer to the column Series of a dataframe,
    whose name is the value of groupname.

    Parameters
    ----------
    df :        pd.DataFrame
                Input dataframe

    groupname : str

    Returns
    -------
    df :        pd.DataFrame
    """
    check_type(df, pd.DataFrame)
    check_type(groupname, str)
    check_type(column_names, [type(None), list])
    check_type(copy, bool)

    if isinstance(column_names, list):
        for name in column_names:
            check_type(name, str)

    if copy:
        df = df.copy()

    # initial_cols = df.columns.values
    new_cols = pd.MultiIndex.from_product([[groupname], df.columns])

    # Assign new columns to df
    df.columns = new_cols

    if column_names:
        df.columns.names = column_names

    return df


def df_histogram(*, df, columns, bins, percent=True, apply_fun=None):

    check_type(df, pd.DataFrame)
    check_type(columns, [str, list])

    columns = np.atleast_1d(columns)

    available_columns = df.columns
    missing = [col for col in columns if col not in available_columns]
    selected_columns = [col for col in columns if col not in missing]

    if missing:
        message = f"Columns {missing} not found in df and skipped."
        report_message(message, log=True, level='warning')

    buffer = []
    for col in selected_columns:

        hist, edge_value = np.histogram(df[col].dropna().values, density=False, bins=bins)

        if percent:
            value = np.round(hist / hist.sum() * 100, 2)
        else:
            value = hist

        # Assign value to dict key
        out = pd.DataFrame(index=edge_value[1:].astype(int), data={col: value})
        out.index.name = "bins"

        if apply_fun:
            out = out.reset_index().apply(apply_fun).set_index("bins")

        buffer.append(out)

    result = pd.concat(buffer, axis=1)

    return result


def df_timedelta_to_seconds(df, column, inplace=False, verbose=False):

    check_type(df, pd.DataFrame)
    check_type(column, str)
    check_type(inplace, bool)
    check_type(verbose, bool)

    df_column_check_type(df, column, pd.Timedelta)

    if not inplace:
        # Make a copy of the DataFrame
        df = df.copy()

    # Extract time_delta (time)
    time_delta_series = df.loc[:, column]

    # Transform to seconds
    total_seconds = time_delta_series.apply(lambda x: x.total_seconds())

    # Transform to numpy 32 (single precision)
    total_seconds_int32 = total_seconds.astype(np.int32, copy=False)

    # Assign back to the original DataFrame
    df.loc[:, column] = total_seconds_int32

    # Check that conversion backwards makes sense
    recalculated = total_seconds_int32.apply(lambda x: pd.Timedelta(x, unit='s'))

    diff = time_delta_series - recalculated

    # Make sure the conversion is at most different by one second (rounding error)
    if len(diff[diff > pd.Timedelta(1, unit='s')]) > 0:
        message = f'conversion from timedelta to nanosecond failed.'
        report_message(message, verbose=verbose, log=True, level='info')
        raise ValueError(message)
    else:
        return df


def df_timestamp_to_iso(df, column, inplace=False, tz='utc', verbose=False):
    """

    Parameters
    ----------
    df :        pd.DataFrame

    column :    Numpy type

    inplace :   bool
                If True modify input

    tz :        str
                timezone

    verbose :   bool
                Improves output verbosity

    Returns
    -------
    df :        pd.DataFrame
    """

    check_type(df, pd.DataFrame)
    check_type(column, str)
    check_type(inplace, bool)
    check_type(verbose, bool)

    # Make sure the column contains pandas Timestamps
    df_column_check_type(df, column, pd.Timestamp)

    # Make sure tz is a valid timezone
    pd_validate_timezone(tz)

    # Try to
    pd.Series().tz_localize(tz=tz)

    if not inplace:
        # Make a copy of the dataframe
        df = df.copy()

    # Convert last in UTC iso-formatted string
    timezone_unaware = df.loc[:, column]

    # Force the timezone to be UTC. We can do this because we know that the original data is all in UTC.
    timezone_aware = timezone_unaware.dt.tz_localize(tz='utc')

    # Convert to iso format (string)
    df.loc[:, column] = timezone_aware.apply(lambda x: x.isoformat())

    message = f'column={column} converted to isoformat.'
    report_message(message, verbose=verbose, log=True, level='info')

    return df


def df_column_check_type(df, column, type):
    """
    Applies the check_type() function to all elements of a df column.
    Parameters
    ----------
    df :        pd.DataFrame
                The pandas DataFrame containing the column 'column'
    column :    any Numpy type
                column of dd for which you want to check the type
    type :      type
                type of Object
    verbose :   Increases verbosity

    Returns
    -------
    True or raises an error
    """
    check_type(df, pd.DataFrame)

    # Make sure column is contained in df
    if column not in df:
        message = f'column is not in df. Indicate one of {list(df.columns.values)}'
        raise ValueError(message)
    else:
        # Extract the Series to convert
        column_series = df.loc[:, column]

        # check that is indeed a pandas Series
        check_type(column_series, pd.Series)

    # check that each element is a pandas Timestamp
    try:
        column_series.map(lambda x: check_type(x, type))
        return True

    except TypeError as ex:
        message = f'df_column_check_type(): df.loc[:, column] should contain ' \
                  f'variables of type {type}.'
        report_message(message, verbose=True, log=True, level='info')
        raise ex


def df_select(*, df, mode="all", **kwargs):
    """
    Allowws to formulate multiple selections across columns of a dataframe.
    Parameters
    ----------
    df
    mode
    kwargs

    Returns
    -------

    """
    # Check mode is supported
    allowed_modes = ["all", "any"]
    if mode not in allowed_modes:
        raise ValueError(f"{mode} must be one of {allowed_modes}")

    # Check for missing keys
    keys_missing = [key for key in kwargs if key not in df.columns]
    if keys_missing:
        raise ValueError(f"{keys_missing} not found in df columns {df.columns}")

    # Build boolean selector for table
    if kwargs:
        filters = []
        for key, value in kwargs.items():

            if value is None:
                # Ignore this value
                report_message(f"{key}={value} skipped.", log=True, level='warning')
                pass
            else:
                possible_values = np.atleast_1d(value)
                cond = df[key].isin(possible_values)
                filters.append(cond)
    else:
        filters = []

    if filters:
        # Build boolean selector based on mode
        if mode == "all":
            tot_filter = pd.concat(filters, axis=1).all(axis=1)
        else:
            tot_filter = pd.concat(filters, axis=1).any(axis=1)

        output = df[tot_filter]

    # Just return full table
    else:
        print("No filter conditions passed, returning full table.")
        output = df

    return output


def df_columns_cat_codes(*, df, newcol, columns=["actorId", "batchCampaignId"]):
    """
    Provides a multi-column version of .astype('categorical').cat.code which
    only works on one column.
    Parameters
    ----------
    df      :   pd.DataFrame
    newcol  :
    columns :

    Returns
    -------
    output  :   pd.DataFrame
    """
    check_type(df, pd.DataFrame)
    check_type(newcol, str)
    check_type(columns, list)

    # Validate schema
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{missing_columns} missing from df")

    # Get table subset
    data = df[columns]

    # Deduplicate to get unique values (remove rows that look identical)
    unique = data.drop_duplicates().reset_index(drop=True)

    unique.index.name = newcol

    # Reset index
    codes_table = unique.reset_index()

    # Perform a merge to make newcol appear in the original df
    output = pd.merge(df, codes_table, how="inner", on=columns)

    if len(output) != len(df):
        raise ValueError("Something went wrong during the merge!")

    return output


def df_most_frequent(*, df, what, sort_col=None, out="sorted_df", n_max=None, ascending=False):
    """
    Looks for the most frequent objects in "what" column of the pandas DataFrame df.
    If n_max is not None, the output is truncated to the top n_max elements.
    Parameters
    ----------
    df          :   pd.DataFrame
    what        :   object
                    The col
    sort_col    :   object
                    The column you want to sort by

    out         :   str
                    One of ["values", "table", "series", "sorted_df"]
                    "values" = Returns a np array containing the sorted column 'what'
                    "table"  = Returns the result of the groupby('what').count()
                    "series" = Same as "values" but in a pd.Series
                    "sorted_df" = The sorted version of df

    n_max       :   None, int
    ascending   :   bool

    Returns
    -------
    output      :   list, Series, pd.DataFrame, np.array
    """
    check_type(n_max, [type(None), int])
    check_type(df, pd.DataFrame)

    out_values = ["values", "table", "series", "sorted_df"]

    if out not in out_values:
        raise ValueError(f"out must be one of {out_values}")

    if what not in df.columns:
        raise ValueError(f"Column {what} not in dataframe.")

    if sort_col and sort_col not in df.columns:
        raise ValueError(f"Column {sort_col} not in dataframe.")

    if not df.columns.name:
        df.columns.name = "columns"
        report_message("unnamed columns in df. Name set to 'columns'.")

    if not df.index.name:
        df.index.name = "index"
        report_message("unnamed index in df. Name set to 'index'.")

    # Rank them
    if sort_col:
        ranking = df.groupby(what).count().sort_values(by=sort_col, ascending=ascending)
    else:
        count = df.groupby(what).count()
        sort_col = count.columns[0]
        ranking = count.sort_values(by=sort_col, ascending=ascending)

    if out == "values":

        if n_max:
            output = ranking[:n_max].index.values
        else:
            output = ranking.index.values

    elif out == "table":
        if n_max:
            output = ranking[:n_max].reset_index()
        else:
            output = ranking.reset_index()

    elif out == "series":
        if n_max:
            output = ranking[:n_max].reset_index()[sort_col]
        else:
            output = ranking.reset_index()[sort_col]

    elif out == "sorted_df":

        # Must order the input dataframe from the most frequent "what" to the least frequent "what"

        output = df.assign(freq=df.groupby(what)[what].transform('count')) \
            .sort_values(by=['freq', what], ascending=[False, True]).drop(columns="freq")

        # Validate output
        if len(output) != len(df):
            raise ValueError("df and merged should have the same size. You lost data!")

        if not df.columns.sort_values().equals(output.columns.sort_values()):
            raise ValueError("df and merged should contain the same columns after merge!")

        # Order columns just as they were before sorting
        output = output[df.columns]

    return output


def df_parse_dates(*, df, columns, errors='coerce'):
    """
    Parses columns from strings to datetime.
    Parameters
    ----------
    df      :   pd.DataFrame
                Table with columns that can be converted to datetime
    columns :   list
    errors  :   str
                Default error policy see link in references
    Returns
    -------
    df      :   pd.DataFrame
                Table with parsed columns
    References
    ----------
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html
    """
    for col in columns:
        df.loc[:, col] = pd.to_datetime(df.loc[:, col], errors=errors)
    return df


def df_lower_case(*, df, columns):
    """
    Normalizes text in the columns of df to be lowercase.
    Parameters
    ----------
    df          :   pd.DataFrame
                    Input table
    columns     :   list
    Returns
    -------
    df          :   pd.DataFrame
                    Table with modified columns
    """
    for col in columns:
        df.loc[:, col] = df.loc[:, col].str.lower()
    return df


def df_percent_missing(df, width=400, height=250):
    """
    Makes interactive bar plot of the percentage of missing values for columns in df.
    Parameters
    ----------
    df          :   pd.DataFrame
                    Table that might contain missing values
    Returns
    -------
    plot        :   hv.element.chart.Bars
                    Interactive plot
    """

    title = f"{df.columns.name}: % of missing values"

    percent = df.isnull().sum() / df.isnull().count() * 100

    plot = percent.hvplot.bar(title=title, value_label="%", width=width, height=height)

    return plot


def df_to_smart_dashboard_api(*, df, title, saveas=None, legend=None, ylabel=None, xlabel=None,
                              xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Takes a pandas dataframe and writes it as a json file on disk at saveas.
    Any column in df is casted to string.
    Parameters
    ----------
    df
    title
    saveas  :   str, None
    ylabel
    xlabel
    xmin
    xmax
    ymin
    ymax

    Returns
    -------

    """
    check_type(df, pd.DataFrame)
    check_type(title, str)
    check_type(saveas, [str, type(None)])
    check_type(ylabel, [str, type(None)])
    check_type(xlabel, [str, type(None)])
    check_type(legend, [str, type(None)])

    # Validate dataframe
    if len(df.columns) < 2:
        raise ValueError(f"At least two columns need to be in DataFrame, "
                         f"the first is the labels for the x-axis, "
                         f"the second contains the values.")

    # Create the options dictionary as shown in
    # https://jsfiddle.net/api/post/library/pure/
    options_dict = {
        "legend": legend,
        "title": title,
        "hAxis": {
            "title": xlabel,
            "minValue": xmin,
            "maxValue": xmax,
        },
        "vAxis": {
            "title": ylabel,
            "minValue": ymin,
            "maxValue": ymax
        }
    }

    # Cast columns to string
    df = df.astype(str)

    # Create the nested structure used for the data
    buffer = list()
    buffer.append(list(df.columns))
    for i in range(len(df)):
        buffer.append(list(df.iloc[i].values))

    d = {"data": buffer,
         "options": options_dict}

    if saveas:

        # Save to jason
        with open(saveas, 'w') as fp:
            json.dump(d, fp)

        # Report
        message = f"df_to_smart_dashboard_api() file saved as {saveas}"
        report_message(message, level='info', log=True)
    else:
        # Report
        message = f"df_to_smart_dashboard_api() dict built successfully."
        report_message(message, level='info', log=True)

    return d


def get_instance_name_from_path(abspath, must_exist=True):
    """
    Extracts the instance name from the absolute path,
    taking into account the fact that in the legacy system
    (bash implementation) the basename of a file did not contain "_",
    while the ELTObject Class enforces a more descriptive basename with "_".
    Parameters
    ----------
    abspath     :   str

    Returns
    -------
    name        :   str
    """
    check_type(abspath, str)

    if must_exist:
        check_file(abspath)

    filename = os.path.basename(abspath)
    if "_" in filename:

        # The file name in abspath follows the new convention (Pipeline V2)
        name = filename.split("_")[-1].split(".")[0]
    else:
        name = filename.split(".")[0]
    return name


def df_action_columns(df, log=True):
    """
    Returns a list of column names for df if their name matches any Autopilot action.
    Parameters
    ----------
    df      :   pd.DataFrame
    log     :   bool
    Returns
    -------
    actions :   list
    """

    check_type(df, pd.DataFrame)
    check_type(log, bool)

    apa = AutopilotActions()

    # Keep list of actions columns
    actions = [c for c in df.columns if c in apa.actions_list]

    # Report
    message = f"{len(actions)} columns in df are autopilot actions: {actions}"
    report_message(message, level='info', log=log)

    return actions


def validate_comp_fastparquet(compression):
    """
    MAkes sure that the string compression is valid for fastparquet engine.
    Parameters
    ----------
    compression
    """
    check_type(compression, str)
    allowed_compressions = ["snappy", "gzip", "brotli", None]
    if compression not in allowed_compressions:
        raise ValueError(f"compression {compression} not supported, must be one of {allowed_compressions}")
    return True


def df_from_gbq(*, sql_query, gcp_project=None, dry_run=True):
    """
    Performs the SQL query `sql_query` in the BigQuery project `gcp_project`
    returning a pandas DataFrame.
    Parameters
    ----------
    sql_query       :   str
                        Standard SQL query
    gcp_project     :   str, None
                        IF None, gcp_project is read from env variable
    dry_run         :   bool
                        If True the query is performed in dry mode,
    Returns
    -------

    """
    check_type(gcp_project, [str, type(None)])
    check_type(sql_query, str)

    if not gcp_project:

        # Extract GCP env variables
        gcp_project = env_get_gcp_project_id()

        report_message(log=True, level="info",
                       message=f"gcp_project read from env variable via env_get_gcp_project_id(),"
                               f"because None was passed. gcp_project={gcp_project}")
    if dry_run:
        query_job_config = bigquery.QueryJobConfig(dry_run=True)
    else:
        query_job_config = bigquery.QueryJobConfig(dry_run=False)

    try:
        # Open a client
        client = bigquery.Client(project=gcp_project)
        report_message(message=f"BigQuery client instantiated for project "
                               f"{client.project}", log=True, level="info")
    except Exception as ex:
        raise ValueError("Error establishing a client - {ex}")

    # Load query job
    query_job = client.query(sql_query, job_config=query_job_config)
    report_message(message="query loaded successfully", log=True, level="info")

    if dry_run:
        # A dry run query completes immediately, wait for it to complete
        assert query_job.state == "DONE"
        processing_message = f"This query will process {sizeof_fmt(query_job.total_bytes_processed)}. "
        output = None
    else:
        # Execute query and load to DataFrame
        output = query_job.to_dataframe()
        processing_message = f"This query has processed {sizeof_fmt(query_job.total_bytes_processed)}. "

    # Estimate cost
    estimated_usd = 5 * query_job.total_bytes_processed / 1e12
    cost_message = f"Estimated cost is {estimated_usd} USD"
    report_message(message=processing_message + cost_message, level="info", log=True)

    return output


"""
# ------------------------------------- PATHS ----------------------------------
Aims ...
# -------------------------------------------------------------------------------
"""


def path_exists(abspath, filesystem=None):
    """

    Parameters
    ----------
    abspath

    Returns
    -------
    answer  :   bool
    """
    check_type(abspath, str)
    check_type(filesystem, [type(None), gcsfs.core.GCSFileSystem, s3fs.core.S3FileSystem])

    # Either find correct filesystem of use the one passed int
    if filesystem:
        fs = filesystem
    else:
        fs = filesystem_for_path(abspath)

    if fs:
        # Remote
        answer = fs.exists(abspath)
    else:
        # Local
        answer = os.path.exists(abspath)

    check_type(answer, bool)

    return answer


def path_is_dir(abspath):

    # Get filesystem for the path
    fs = filesystem_for_path(abspath)

    if fs:
        raise ValueError(f"{abspath} points to a bucket, cannot check if path is a directory")

    if not path_exists(abspath):
        raise ValueError(f"Path {abspath} does not exist.")

    # Work out if it is a file or a folder
    answer = os.path.isdir(abspath)

    return answer


def path_is_empty(abspath):

    # Get filesystem for the path
    fs = filesystem_for_path(abspath)

    if fs:
        # Path is to a remote location in a bucket
        raise ValueError(f"no concept of directory for remote bucket, passed {abspath}")

    if not path_exists(abspath):
        raise ValueError(f"{abspath} does not exist.")

    if not path_is_dir(abspath):
        raise ValueError(f"{abspath} is not a directory.")

    # List all files in path
    files_in_dir = os.listdir(abspath)

    if len(files_in_dir) > 0:
        answer = False
    else:
        answer = True

    return answer


def path_consolidate(file_abs_path):
    """
    Makes sure the parent directory of file_abs_path exists on disk.
    Parameters
    ----------
    file_abs_path

    Returns
    -------

    """
    if not os.path.isabs(file_abs_path):
        raise ValueError(f"os.path.isabs(dest) returned false for file_abs_path={file_abs_path}. "
                         f"Invalid destination.")

    # Extract destination directory
    dest_dir = os.path.dirname(file_abs_path)

    # --- Check that destination exists ---
    os.makedirs(dest_dir, exist_ok=True)

    return True


def path_dirs_from_paths(paths, log=True):
    """
    Returns list of directories from paths. Raises a warning for
    paths that are not directories.
    Parameters
    ----------
    paths       :   list
    log         :   bool
    Returns
    -------
    path_dirs     :   list
    """

    check_type(paths, list)

    # Make sure it's a list of strings
    for d in paths:
        check_type(d, str)

    # Find those that are not directories
    not_dirs = [dirname for dirname in paths if not os.path.isdir(dirname)]

    if not_dirs:
        message = f"Directories {not_dirs} not found on disk and ignored."
        report_message(message, log=log, level='warning')

        # Get subset of actual directories
        path_dirs = [d for d in paths if d not in not_dirs]
    else:
        path_dirs = paths

    return path_dirs


def path_describe(*, dirname, sort_by="size",
                  ascending=False, minimal=True, extensions=None,
                  attribute=False, log=True, error_if_empty=False):

    check_type(dirname, str)
    check_type(sort_by, str)
    check_type(ascending, bool)
    check_type(minimal, bool)
    check_type(extensions, [type(None), list])

    # Get filesystem for the path
    fs = filesystem_for_path(dirname)

    # Describe path
    if fs:
        df = path_describe_bucket_dir(dirname)
    else:
        df = path_describe_local_dir(dirname)

    if extensions:
        # Select only results whose extension is in extensions
        df = df[df["ext"].isin(extensions)]
        message = f"{len(df)} rows left after filtering for df['ext'] in {extensions}"
        report_message(message, log=log, level='info')

    if attribute:
        # Attribute instance to file
        df["instance"] = df["abspath"].apply(lambda x: get_instance_name_from_path(x, must_exist=False))

    # Sort and reindex
    df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    if extensions:
        # Report message
        message = f"{len(df)} files/dirs with extension {extensions} found in {dirname}"
    else:
        message = f"{len(df)} files/dirs found in {dirname}"

    report_message(message, log=log, level='info')

    if error_if_empty and len(df) == 0:
        raise ValueError(f"no files found in {dirname}, raise error becasue error_if_empty={error_if_empty}")
    return df


def path_describe_many(dirlist, minimal=True, sort_by="size", ascending=False, log=True, extensions=[]):
    """
    Returns a table of metadata on all files found in dirlist.
    Parameters
    ----------
    dirlist     :   list
    minimal     :   bool
    log         :   bool
    extensions  :   list
    Returns
    -------
    df          :   pd.DataFrame
    """

    buffer = []
    tmp_filenames = [f"/tmp/path_describe_many-{os.path.basename(d)}.pkl" for d in dirlist]

    def single_describe_and_save(d, fn):
        # Describe directory
        df = path_describe(dirname=d, minimal=minimal, sort_by=sort_by, ascending=ascending, extensions=extensions)
        # Save description
        obj_serialize(df, saveas=fn, force_ext=False, verbose=False, log=log)
        return True

    # ------ MAP -------

    # Prepare computation map
    jobs = pd.DataFrame({"d": dirlist, "fn": tmp_filenames})

    f = single_describe_and_save
    parallel_exec_map(f=f, execution_map=jobs, workers=max_processes())

    # ------ REDUCE -------

    for path in jobs["fn"]:
        buffer.append(obj_deserialize(path, verbose=False, log=log))

    # Concatenate tables
    output = pd.concat(buffer, axis=0).sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    # ------ CLEAN -------

    for path in jobs["fn"]:
        os.remove(path)

    return output


def path_bucket_id(path):
    """Returns the first two letters identifying a bucket ('gs' for GCS, 's3' for AWS S3)"""

    check_path_remote(path)
    output = path.split("://")[0]

    expected_outputs = ["gs", "s3"]
    if output not in expected_outputs:
        raise ValueError(f"remote path is expected to begin with one of {expected_outputs}, passed {path}")

    return output


def path_describe_bucket_dir(dir, log=True):
    """

    Parameters
    ----------
    dir
    minimal
    log

    Returns
    -------

    """
    check_type(dir, str)
    check_type(log, bool)

    # Get filesystem for the path
    fs = filesystem_for_path(dir)

    # Make sure you get a filesystem
    if not fs:
        raise ValueError(f"dir should be the path to a branch on a remote bucket, passed {dir}")

    # Prepare output table
    features = ["abspath", "isfile", "isdir", "size", "ext", "last_access",
                "last_change", "creation_time", "basename", "dirname"]

    # List all assets
    all_assets = fs.ls(dir)

    # Get list of things in dir
    bid = path_bucket_id(dir)

    all_abspaths = [bid + "://" + name for name in all_assets]
    all_basename = [os.path.basename(p) for p in all_assets]
    all_ext = [os.path.splitext(p)[-1] for p in all_basename]

    df = pd.DataFrame(index=range(len(all_assets)), columns=features)

    df["abspath"] = all_abspaths
    df["basename"] = all_basename
    df["ext"] = all_ext
    try:
        df["size"] = [fs.info(key)["Size"] for key in all_assets]
    except Exception:
        df["size"] = [fs.info(key)["size"] for key in all_assets]

    # Add the directory name
    df["dirname"] = dir

    return df


def path_describe_local_dir(dir, minimal=True, log=True):

    # Prepare output table
    features = ["abspath", "isfile", "isdir", "size", "ext",
                "last_access", "last_change", "creation_time",
                "basename", "dirname"]

    if not os.path.isdir(dir):
        raise ValueError(f"{dir} is not a directory.")

    # Get list of things in dir
    base_names = os.listdir(dir)

    df = pd.DataFrame(np.nan, index=base_names, columns=features)

    if not base_names:
        message = f"No files in {dir}, returning empty table"
        report_message(message, log=log, level='warning')
        return df

    # Populate table with useful info
    for base in base_names:

        this_abspath = os.path.join(dir, base)
        df.loc[base, "abspath"] = this_abspath
        df.loc[base, "basename"] = base
        df.loc[base, "ext"] = os.path.splitext(base)[1]
        df.loc[base, "size"] = os.path.getsize(this_abspath)

        if not minimal:
            # These operations take time, only do it if requested
            stats = os.stat(this_abspath)
            df.loc[base, "last_access"] = pd.Timestamp(stats.st_atime, unit='s')
            df.loc[base, "last_change"] = pd.Timestamp(stats.st_mtime, unit='s')
            df.loc[base, "creation_time"] = pd.Timestamp(stats.st_ctime, unit='s')
            df.loc[base, "isfile"] = os.path.isfile(this_abspath)
            df.loc[base, "isdir"] = os.path.isdir(this_abspath)

    # Add the directory name
    df["dirname"] = dir

    return df


def path_splitext(path):
    """
    USes os.path.splitext recursively, if filename has multiple extensions.
    Parameters
    ----------
    path    :   str

    Returns
    -------
    output  :   tuple

    Examples
    --------
    multiple_extension_path = "/ff/c/jnsufhsf.json.tar.gz"

    # See difference between these two
    os.path.splitext(multiple_extension_path)
    path_splitext(multiple_extension_path)
    """
    check_type(path, str)
    ext_parts = []
    new_path = path
    while True:
        split = os.path.splitext(new_path)
        new_path = split[0]
        this_ext = split[1]
        if this_ext:
            ext_parts.append(this_ext)
        else:
            extension = ''.join(ext_parts)
            output = (new_path, extension)
            return output


"""
# --------------------------------------- OBJECTS -------------------------------------
Aims ...
# --------------------------------------------------------------------------------------
"""


def obj_serialize(obj, saveas=None, force_ext=False, verbose=False, log=True):
    """
    Serializes an object and writes ti to disk (.pkl) with pickle.dump().
    Parameters
    ----------
    obj         :   Any (serializable)
                    absolute path of the pickled object to read.
    saveas      :   str, None
                    Absolute path where to save the object (.pkl)
    force_ext   :   bool
                    If True and saveas does not end with '.pkl' raises error.
    verbose     :   bool
                    Increases output verbosity
    log         :   bool
    References
    ----------
    Inspired by https://bit.ly/2IU0Ol7
    """
    check_type(saveas, [str, type(None)])
    check_type(force_ext, bool)
    check_type(verbose, bool)

    if saveas is None:
        message = f'Object not saved.'
        report_message(message, verbose=verbose, log=log, level='info')
        return False

    if force_ext and not saveas.endswith('.pkl'):

        _, ext = os.path.splitext(saveas)
        message = f'Absolute path "saveas" required to be ".pkl", passed {ext}'
        report_message(message, verbose=verbose, log=log, level='info')
        raise ValueError(message)

    # Get filesystem for the path
    fs = filesystem_for_path(saveas)

    if fs:
        # Remote location
        with fs.open(saveas, 'wb') as f:

            pickled_obj = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)

            # Write to bucket
            f.write(pickled_obj)

        message = f'Object serialized and saved remotely at {saveas}.'
        report_message(message, verbose=verbose, log=log, level='info')
        return True

    else:
        # Create local dir if it does not exist
        os.makedirs(os.path.dirname(saveas), exist_ok=True)

        # Write to disk
        with open(saveas, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        message = f'Object serialized and saved at {saveas}.'
        report_message(message, verbose=verbose, log=log, level='info')
        return True


def obj_deserialize(f_abspath, log=True, verbose=False):
    """
    Simple function that streams a pickled object (.pkl) from s3 or disk,
    with pickle.load().
    Parameters
    ----------
    f_abspath :     str
                    absolute path of the pickled object to read.
    Returns
    -------
    loaded :        Any
                    The python object that was serialized.
    References
    ----------
    Inspired by https://bit.ly/2IU0Ol7
    """
    check_type(f_abspath, str)

    # Get filesystem for the path
    fs = filesystem_for_path(f_abspath)

    if fs:
        # Remote destination
        with fs.open(f_abspath, 'rb') as f:

            # Read from bucket
            loaded = pickle.load(f)

        message = f'Object extracted from S3 key {f_abspath}.'
        report_message(message, verbose=verbose, log=log, level='info')

    else:
        # Local destination
        with open(f_abspath, 'rb') as f:
            loaded = pickle.load(f)

        message = f'Object extracted from path {f_abspath}.'
        report_message(message, verbose=verbose, log=log, level='info')

    return loaded


def obj_load(src, dest, log=True, verbose=False, **kwargs):
    """
    Moves an object from src_path to dest_path, taking care of
    setting up the right connection if dest_path is path to a bucket.
    Parameters
    ----------
    src
    dest

    Returns
    -------

    """
    # Make sure source path exists
    path_exists(src)

    # Make sure is a file on the local FileSystem
    if not os.path.isfile(src):
        raise ValueError(f"src not found - {src}")

    # Get filesystem for the path
    fs = filesystem_for_path(dest)

    if fs:
        # Remote destination
        message = f"loading {src} to \n\t{dest}"
        report_message(message, level="info", log=log, verbose=verbose)

        fs.put(src, dest, **kwargs)
        # TODO "content-type": "image/png"
    else:
        # Local destination
        my_logger.warning(f"Making a copy of {src} to {dest}")
        shutil.copyfile(src, dest)

    message = f"asset {src} successfully loaded to {dest}"
    report_message(message, level="info", log=log, verbose=verbose)

    return True


def obj_extract(src, dest, nosrc_ok=True, use_cache=False, log=True):
    """
    Moves an obect from src_path to dest, taking care of
    setting up the right connection if dest is an s3 key.
    Parameters
    ----------
    src             :   str
    dest            :   str
    log             :   bool
    nosrc_ok        :   bool
    use_cache       :   bool

    Returns
    -------

    """
    check_type(src, str)
    check_type(dest, str)
    check_type(log, bool)
    check_type(nosrc_ok, bool)
    check_type(use_cache, bool)

    # Make sure source path exists
    if not path_exists(src):
        if not nosrc_ok:
            raise ValueError(f"Source absolute path {src} not found.")
        else:
            message = f"src {src} does not exist but missing_src_ok is {nosrc_ok}, extraction skipped"
            report_message(message, level='warning', log=log)
            return True

    # Change behavior depending if file exists at destination
    if path_exists(dest):
        if use_cache:
            message = f"File {src} already present at destination {dest}, " \
                      f"\nextraction skipped because use_cache={use_cache}"
            report_message(message, level='warning', log=log)
            return True

    # Make sure a directory exists for the file
    path_consolidate(dest)

    # Get filesystem for the src path
    fs = filesystem_for_path(src)

    if fs:
        # Extract from remote path
        fs.get(src, dest)
    else:
        # Extract from local paths
        my_logger.info(f"Making a local copy of {src} to {dest}")
        shutil.copyfile(src, dest)

    # Report success message
    message = f"File {src} correctly loaded to {dest}"
    report_message(message, log=log, level='info')

    return True



"""
# -------------------------------------- COMPRESSION --------------------------------------
Aims ...
# -----------------------------------------------------------------------------------------
"""


def validate_compression_type(comp):
    check_type(comp, str)

    # Validate compression
    allows_comp_types = ["", "gz", "bz2", "xz"]

    if comp not in allows_comp_types:
        raise ValueError(f"compression must be one of {allows_comp_types}, passed {comp}")
    return True


def tar_and_compress_files(*, src_list, dest, empty_archive=False, compression="bz2", force_ext=True, log=True):
    """
    Archives and compresses a list of files. Note: Compression is a single-thread operation and can
    take a long time on large datasets. If execution speed is important set compression to "".
    Parameters
    ----------
    compression :   str
                    Compression algorithm, must be one of ["","gz", "bz2", "xz"].
    force_ext   :   bool
                    If True, src must end in f".tar.{compression}"
    log         :   bool
                    Improves output verbosity
    src_list    :   list
                    List of files to add to compressed act_raw_archive
    dest        :   str
                    Absolute path of the compressed act_raw_archive
    """
    check_type(src_list, list)
    check_type(dest, str)

    # Validate compression
    validate_compression_type(compression)

    # Check that all files in src_list exist
    _ = [check_file(file) for file in src_list]

    if not os.path.isabs(dest):
        raise ValueError(f"dest {dest} is not a valid path on the host OS. "
                         f"The function os.path.isabs() returned False.")

    if src_list:  # List not empty

        # Make tar file and save it!
        with tarfile.open(dest, f"w:{compression}") as tar:
            for file in src_list:
                tar.add(file, arcname=os.path.split(file)[1])

        message = f"{len(src_list)} files compressed to {dest}."
        report_message(message, verbose=False, log=log, level="info")

    else:
        if empty_archive:

            # List empty, make an empty tar file
            with tarfile.open(dest, f"w:{compression}") as tar:
                pass

            message = f"Passed empty list, empty tar archive created."
            report_message(message, verbose=False, log=log, level="info")
        else:

            # Write an empty file in the parent directory at destination
            file_abspath = os.path.join(os.path.dirname(dest), "empty-file.txt")
            with open(file_abspath, mode='w') as ef:
                ef.write("Empty File")

            with tarfile.open(dest, f"w:{compression}") as tar:
                tar.add(file_abspath, arcname=os.path.split(file_abspath)[1])

            message = f"Passed empty list, tar archive create with a dummy txt file."
            report_message(message, verbose=False, log=log, level="info")

    # Validate filename
    validate_tar_abspath(dest, compression, force_ext)

    return True


def tar_and_compress_path(*, src, dest, compression="bz2", force_ext=True, log=True):
    """
    Takes all files in src_path (must be a directory), and creates an act_raw_archive
    with gzip compression at dest_path. Note: Compression is a single-thread operation and can
    take a long time on large datasets. If execution speed is important set compression to "".
    Parameters
    ----------
    compression :   str
                    Compression algorithm, must be one of ["","gz", "bz2", "xz"].
    force_ext   :   bool
                    If True, dest must end in f".tar.{compression}"
    log         :   bool
                    Improves output verbosity
    src         :   str
                    Path to a local directory to act_raw_archive and compress.
    dest        :   str
                    Absolute path of the compressed act_raw_archive

    References
    ----------
    https://docs.python.org/3/library/tarfile.html
    """
    check_type(src, str)
    check_type(dest, str)

    # Validate compression type
    validate_compression_type(compression)

    # Make sure the source directory is a valid dir
    check_dir(src)

    if not os.path.isabs(dest):
        raise ValueError(f"dest {dest} is not a valid path on the host OS. "
                         f"The function os.path.isabs() returned False.")

    if os.listdir(src):

        # Make tar file and save it!
        with tarfile.open(dest, f"w:{compression}") as tar:
            tar.add(src, arcname='.')

        message = f"All files in directory {src} compressed to {dest}."
        report_message(message, verbose=False, log=log, level="INFO")

    else:
        # Write an empty file in the parent directory at destination
        file_abspath = os.path.join(os.path.dirname(dest), "empty-file.txt")
        with open(file_abspath, mode='w') as ef:
            ef.write("Empty File")

        # Archive and compress
        with tarfile.open(dest, f"w:{compression}") as tar:
            tar.add(file_abspath, arcname=os.path.split(file_abspath)[1])

        message = f"No files found in {src}, tar archive create with a dummy txt file."
        report_message(message, verbose=False, log=log, level="info")

    # Validate filename
    validate_tar_abspath(dest, compression, force_ext)

    return True


def validate_tar_abspath(abspath, compression, force_ext):
    """
    If force_ext=True, abspath must end in either ".tar" or ".tar.{compression}".
    Parameters
    ----------
    abspath
    compression
    force_ext
    """

    if not os.path.isfile(abspath):
        raise ValueError(f"compressed TO_archive {abspath} does not exist.")

    if not tarfile.is_tarfile(abspath):
        raise ValueError(f"{abspath} does not appear to be a tar archive file "
                         f"that the 'tarfile' python module can read.")

    if compression:
        expected_ext = f".tar.{compression}"
    else:
        expected_ext = ".tar"

    if not abspath.endswith(expected_ext) and force_ext:
        raise ValueError(f"force_ext=True, but abspath={abspath} does not end in .tar.{compression}")

    return True


def untar_and_uncompress_path(*, src, dest=None, compression="bz2", force_src_ext=True,
                              nosrc_ok=True, use_cache=True, expected_ext=None,
                              cache_th=100, log=True):
    """
    Takes a gzip-compressed act_raw_archive at src_path, and un-compresses it. If
    dest_path=None will uncompress in the same directory as src_path.
    Parameters
    ----------
    compression     :   str
                        Compression algorithm, must be one of ["gz", "bz2", "xz",""].
    force_src_ext   :   bool
                        If True, src must end in f".tar.{compression}"
    log             :   bool
                        Improves output verbosity
    dest            :   None, str
                        If not None, must be a path to a local directory
    src             :   str
                        Absolute local path to the compressed act_raw_archive.
    nosrc_ok        :   bool
                        If True no error is raised when src does not exist, and function returns
    use_cache       :   bool
                        If more than `cache_th` files are found in dir and `use_cache`==True,
                        extraction is skipped. Caching is more accurate if `expected_ext` is known.
    cache_th        :   int
                        Number of files in dir to determine whether to skip extraction or not.
    expected_ext    :   str, None
                        If not None, only files with this extensions are considered for caching in dest

    References
    ----------
    https://docs.python.org/3/library/tarfile.html
    """
    check_type(src, str)
    check_type(dest, [str, type(None)])
    check_type(expected_ext, [str, type(None)])

    # Validate compression type
    validate_compression_type(compression)

    try:
        # Validate filename
        validate_tar_abspath(src, compression, force_src_ext)

    except Exception as ex:
        report_message(level="warning", log=True,
                       message=f"validate_tar_abspath failed with error {ex} skipping extraction.")
        return False

    # ---- Check source ----

    if not path_exists(src):
        if not nosrc_ok:
            raise ValueError(f"Source absolute path {src} not found.")
        else:
            message = f"src {src} does not exist but missing_src_ok is {nosrc_ok}, extraction skipped"
            report_message(message, level='warning', log=log)
            return True

    # ---- Check destination ----

    if dest is None:
        # Use the same directory as the act_raw_archive
        dest_dir = os.path.dirname(src)
    else:
        # Make sure a directory exists for the file
        path_consolidate(dest)
        dest_dir = dest

    # ---- Check if files already expanded ----

    if use_cache:
        if expected_ext:
            # List all files in dest dir
            expanded_files = [file for file in glob.glob(dest_dir + f"/*{expected_ext}")]
        else:
            # List all files that match expected extension
            expanded_files = [file for file in glob.glob(dest_dir + f"/*")]

        if len(expanded_files) > cache_th:
            # Already uncompressed
            message = f"\nfound {len(expanded_files)} in {dest_dir} directory, " \
                      f"\nfile {src} is likely to have been already expanded." \
                      f"\nSkipping because use_cache={use_cache}"
            report_message(message, level="info", log=log)
            return True

    # If not returned before, expand
    report_message(message=f"Extracting all files from {src}")

    # Untar and uncompress files in destination path
    with tarfile.open(src, f"r:{compression}") as tar:
        tar.extractall(path=dest_dir)

    # Report status
    report_message(f"File {src} extracted in {dest_dir}", level="info", log=log)

    return True


"""
# ---------------------------------- ENVIRONMENT --------------------------------
Aims at creating a single place to validate that all necessary env variables are available
in the current running environment.
# -------------------------------------------------------------------------------
"""

required_variables=['ENV',
                     'BUCKETS',
                     'ETL_HOME_TEMP_OBJECTS',
                     'ETL_HOME_TEMP_LOGS',
                     'AIRFLOW__CORE__REMOTE_BASE_LOG_FOLDER',
                     'AWS_APPLICATION_CREDENTIALS',
                     'AIRFLOW_HOME',
                     'AIRFLOW__CORE__PLUGINS_FOLDER',
                     'AIRFLOW__CORE__BASE_LOG_FOLDER',
                     'AIRFLOW__CORE__DAGS_FOLDER',
                     'AIRFLOW__CORE__EXECUTOR',
                     'AIRFLOW__CORE__LOAD_EXAMPLES',
                     'AIRFLOW__CORE__REMOTE_LOGGING',
                     'AIRFLOW__CORE__DEFAULT_TIMEZONE',
                     'POSTGRES_PASSWORD',
                     'POSTGRES_USER',
                     'POSTGRES_DB',
                     'AIRFLOW__CORE__SQL_ALCHEMY_CONN',
                     'AIRFLOW__CORE__PARALLELISM',
                     'AIRFLOW__CORE__DAG_CONCURRENCY',
                     'AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION',
                     'AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG',
                     'AIRFLOW__CORE__LOGGING_CONFIG_CLASS',
                     'AIRFLOW__CORE__FERNET_KEY',
                     'AIRFLOW__API__AUTH_BACKEND',
                     'AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC',
                     'AIRFLOW__SCHEDULER__CHILD_PROCESS_LOG_DIRECTORY',
                     'APHQ_API_GOAL_KEY',
                     'APHQ_API_GOAL_URL',
                     'CLOUDANT_PASSWORD',
                     'CLOUDANT_URL',
                     'CLOUDANT_USERNAME',
                     'DEV_SUB_LIST',
                     'GOOGLE_SHEET_SUCCESS_SPREADSHEET',
                     'GOOGLE_APPLICATION_CREDENTIALS',
                     'KELP_CONTENT_TYPE',
                     'KELP_EX_ACTIVITIS_URL',
                     'KELP_REPORT_URL',
                     'KELP_X_API_KEY',
                     'PROV_SUB_NAME',
                     'PROV_SUB_PW',
                     'PROV_SUB_URL',
                     'SHOWERHEAD_AUTH_KEY',
                     'SHOWERHEAD_CONTENT_TYPE',
                     'SHOWERHEAD_RESET_CACHE_URL',
                     'SHOWERHEAD_TIME_ZONE_URL',
                     'AIRFLOW__WEBSERVER__DAG_ORIENTATION',
                     'AIRFLOW__WEBSERVER__BASE_URL',
                     'AIRFLOW__WEBSERVER__AUTHENTICATE',
                     'AIRFLOW__WEBSERVER__WEB_SERVER_PORT',
                     'AIRFLOW__WEBSERVER__AUTH_BACKEND',
                     'AIRFLOW_INITIAL_USER_PASSWORD',
                     'AIRFLOW_INITIAL_USER_USERNAME',
                     ]


def set_python_env_from_file(filename):
    """
    Reads key abd values from file filename separated by and '='.
    Parameters
    ----------
    filename

    Returns
    -------

    """
    check_type(filename, str)

    # Make sure file exists
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} not found.")

    set_variables = []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                try:
                    # Split on "=" (only the first equal you find)
                    key, value = line.strip().split(sep='=', maxsplit=1)

                    # Remove any white space in the key and value
                    key = key.strip()
                    value = value.strip()

                    # Add key value pair to environment
                    os.environ[key] = value

                    # Appent to list of keys set
                    set_variables.append(key)

                except Exception as ex:
                    message = f"Error with line {line}"
                    report_message(message, level='error')

    return set_variables


def show_env_values_and_types():

    for key, value in os.environ.items():
        print(f"({type(value)}) - {key}: {value}")

    return True


def cloudant_creds():

    user = env_get("CLOUDANT_USERNAME", required=True)
    passwd = env_get("CLOUDANT_PASSWORD", required=True)
    url = env_get("CLOUDANT_URL", required=True)

    return user, passwd, url


def env_get(env, required=True):

    check_type(env, str)
    check_type(required, bool)

    # Retrieve slack endpoint from env
    value = os.environ.get(env)

    if value is None and required:
        raise ValueError(f"Failed to parse env variable {env} because it does not exist.")

    return value


def etl_local_home():
    """Simply returns the path to the local directory to temporarily save assets"""
    home = env_get("ETL_HOME_TEMP_OBJECTS")
    return home


def available_bucket_ids():
    """
    Returns a list of strings, each containing 'xx://' for each bucket defined in
    env variable BUCKETS, where xx depends on the bucket type.
    Returns
    -------
    output      :   list(str)
    """
    split_on = "://"
    output = []

    # Read from environment
    BUCKETS = env_get("BUCKETS")

    # Extract bucket names from BUCKETS
    bucket_names = [bucket.strip() for bucket in BUCKETS.split(",")]
    for name in bucket_names:
        parts = name.split(split_on)
        if not parts:
            raise ValueError(f"env variable BUCKETS contains invalid bucket names, tye must contain {split_on}")

        # Add this identifier to the list
        output.append(parts[0] + split_on)
    return output


def airflow_bucket_from_id(identifier=None):
    """
    Reads the env variable BUCKETS and instantiates as
    many AirlfowBuckets objects as needed
    Parameters
    ----------
    identifier

    Returns
    -------

    """

    check_type(identifier, [type(None), str])

    # Read from environment
    BUCKETS = env_get("BUCKETS")

    # Extract bucket names from BUCKETS
    bucket_names = [bucket.strip() for bucket in BUCKETS.split(",")]

    # Optionally filter
    if identifier:
        matches = [name for name in bucket_names if name.startswith(identifier)]

        # Prepare output
        if len(matches) == 0:
            # No match found
            return None

        elif len(matches) != 1:
            # Passed identifier but found more than one. Ambiguous.
            ValueError(f"Found more than one bucket name with identifier {identifier}, expected at most one")
        else:
            # Attempt to create an object and return it
            return AirflowBucket(name=matches[0])
    else:
        # Return a list of bucket objects
        output = [AirflowBucket(name=name) for name in bucket_names]
        return output


def load_enfile(filepath, log=True):

    with open(filepath) as envfile:
        for line in envfile:
            line = line.replace("\n", "")
            if line and line[0] != "#":
                parts = line.strip().split("=")
                os.environ[parts[0]]=parts[1]

    message = f"env file {filepath} loaded"
    report_message(message, level="info", log=log)

    return True


def env_get_gcp_project_id():
    with open(env_get("GOOGLE_APPLICATION_CREDENTIALS"), "r") as file:
        gacc_creds = json.load(file)

    output = gacc_creds["project_id"]
    return output


class AirflowBucket:

    def __init__(self, *, name):
        self.name = name
        self.__id = None
        self.__bucket_vendor = None
        self.__products = None
        self.__tests = None
        self.__logs = None
        self.__fs = None
        self.ini_bucket()

    @property
    def id(self):
        return self.__id

    @property
    def bucket_vendor(self):
        return self.__bucket_vendor

    @property
    def products(self):
        return self.__products

    @property
    def tests(self):
        return self.__tests

    @property
    def logs(self):
        return self.__logs

    @property
    def filesystem(self):
        return self.__fs

    def ini_bucket(self):
        if "gs://" in self.name:
            self.__bucket_vendor = "GCP"  # It's a Google Storage bucket
            self.__id = "gs://"
            self.__fs = filesystem_gcs()

        elif "s3://" in self.name:
            self.__bucket_vendor = "AWS"  # It's an Amazon S3 bucket
            self.__id = "s3://"
            self.__fs = filesystem_s3()
        else:
            raise ValueError(f"Only AWS and GCP supported at the moment, passed {self.name}")

        self.__logs = "/".join([self.name, "logs"])
        self.__products = "/".join([self.name, "products"])
        self.__tests = "/".join([self.name, "pytest"])

        return self

    def fs(self):

        if self.name.startswith("gs://"):
            filesystem = filesystem_gcs()
            return filesystem

        elif self.name.startswith("s3://"):
            filesystem = filesystem_s3()
            return filesystem
        else:
            raise ValueError(f"Only AWS and GCP supported at the moment, passed {self.name}")

    def __repr__(self):

        attrs = vars(self)
        string = ''
        for k, v in attrs.items():
            if '__' not in k:
                string = string + f'{k} = {v} \n'
            else:
                this_k = k.split('__')[-1]
                string = string + f'{this_k} = {v} \n'
        return string


"""
--------------------------------------- MISCELLANEOUS ------------------------------------------
-------------------------------------------------------------------------------------------------
"""


def sizeof_fmt(num, precision=3):
    """
    Simple function that displays the size of a binary object in a human raeadable way
    Parameters
    ----------
    num     :   numeric

    Returns
    -------
    output  :   str
    """

    check_numeric(num)
    check_type(precision, int)

    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:

        if abs(num) < 1024.0:
            output = f"{round(num,precision)} {unit}B"
            return output
        num /= 1024.0

    output = f"{round(num,precision)} {unit}B"
    return output


def dump_keys(d, lvl=0, buff_len=20):
    """
    Prints the keys in a actions_dict and the type of the
    corresponding value. If d is a nested index, it continues
    to display keys and value type until it reaches the end.

    Parameters
    ----------
    d : dict

    Returns
    -------
    b : type

    Example
    -------
    # Create a actions_dict with other nested dictionaries:
    d = {'a': {'fist': [2,3,4], 'second': pd.DataFrame()}, 'b':4}

    # Call the method to have a quick peak under the curtain
    dump_keys(d)

    """
    for k, v in d.items():
        key_label = lvl * '  ' + str(k)
        padd_len = buff_len - len(key_label)
        if padd_len < 0:
            raise ValueError('buff_len not enough, specify a longer padding.')
        else:
            padd = ' ' * padd_len
            if isinstance(v, str):
                # Print first 20 characters of value
                printed_v = v[0:30]
                print(key_label + padd + str(type(v)).lstrip() + '    "' + printed_v + '"')

            elif isinstance(v, datetime) | isinstance(v, bool):
                printed_v = str(v)
                print(key_label + padd + str(type(v)).lstrip() + '    "' + printed_v + '"')

            elif isinstance(v, list):
                printed_v = []
                for element in v:
                    printed_v.append(str(type(element)))
                print(key_label + padd + str(type(v)).lstrip() + '    "' + str(printed_v) + '"')

            else:
                # Print only the type
                printed_v = str(type(v)).lstrip() if type(v) != dict else '|'
                print(key_label + padd + printed_v)

            if type(v) == dict:
                dump_keys(v, lvl + 1, buff_len=buff_len)
            else:
                # Try to convert field into a dict and display it.
                try:
                    vdict = dict(v)
                    dump_keys(vdict, lvl + 1, buff_len=buff_len)
                except Exception as ex:
                    pass


def report_message(message, verbose=False, log=True, level='info', logger=None):
    """
    Simple function to log messages.
    Parameters
    ----------
    message     :   str
    verbose     :   bool
    log         :   bool
    level       :   str
    logger      :   logging.Logger

    Example
    -------
    report_message(message, log=True, level='info')
    report_message(message, log=True, level='error')
    """
    check_type(message, str)
    check_type(verbose, bool)
    check_type(log, bool)
    check_type(level, str)
    check_type(logger, [type(None), logging.Logger])

    allowed_levels = ['info', 'debug', 'warning', 'error', 'critical']

    if level.casefold() not in allowed_levels:
        raise ValueError('level=' + str(level) + ' invalid, expected to be in ' + str(allowed_levels) + '.')

    # Update logger if any was specified
    logger = my_logger if logger is None else logger

    if log:
        # Log on the root logger
        if level.casefold() == 'info':
            logger.info(message)

        elif level.casefold() == 'debug':
            logger.debug(message)

        elif level.casefold() == 'warning':
            logger.warning(message)

        elif level.casefold() == 'error':
            logger.error(message)
            raise ValueError(message)

        elif level.casefold() == 'critical':
            logger.critical(message)

    if verbose:
        # print(message, file=sys.stderr)
        print(message)


def param_doc(par, doc):
    """
    Extract documentation line(s) for the parameter par in the docstring doc.

    Parameters
    ----------
    a : type

    Returns
    -------
    b : type
    """
    lines = doc.splitlines()
    # Find line
    pos = [i for i, line in enumerate(lines) if par in line]
    try:
        p = pos[0]
        extracted_docs = lines[p] + '\n' + lines[p + 1]

    except Exception:
        extracted_docs = ' MISSING DOCS '

    return extracted_docs


def list_diff(A, B, verbose=False):
    """
    Check if two lists differ in either number of elements or order.
    Parameters:
    ----------
    A :             list
                    Current List
    B :             list
                    Reference List
    verbose :       bool
                    Increases output verbosity.
    Returns:
    --------

    difference :    list of differences between two lists A and B
    longer     :    True: A longer than B
                    False: B longer than A
                    None: A and B have same length
    exit_code :     1: A shorter than B
                    2: A longer than B
                    3: Same length but different elements
                    4: Same length, same elements but different order
                    5: Same length, same elements same order (identical)
    """

    check_type(A, list)
    check_type(B, list)
    check_type(verbose, bool)

    current_length = len(A)
    target_length = len(B)
    diff = []

    if current_length > target_length:
        diff = list(set(A) - set(B))
        # order the different elements based on their index (from low to high)
        diff_indexes = [A.index(d) for d in diff]
        diff_indexes.sort()
        difference = [A[ind] for ind in diff_indexes]
        longer = True
        exit_code = 2

        # Report status
        message = f'A longer than B: len(A)={len(A)}, len(B)={len(B)}, ' \
                  f'item(s) {diff} is(are) in the A but not in B.'
        report_message(message, verbose=verbose, log=True, level='info')

    elif current_length < target_length:
        diff = list(set(B) - set(A))
        # order the different elements based on their index (from low to high)
        diff_indexes = [B.index(d) for d in diff]
        diff_indexes.sort()
        difference = [B[ind] for ind in diff_indexes]
        longer = False
        exit_code = 1

        # Report status
        message = f'A shorter than B: len(A)={len(A)}, len(B)={len(B)}.' \
                  f' Item(s) {diff} is(are) missing from A.'
        report_message(message, verbose=verbose, log=True, level='info')

    elif current_length == target_length:
        longer = None
        diff = list(set(A) - set(B))

        if len(diff) == 0:
            # same length and same elements. Check if order is also the same order:
            ordered_diff = [A[i] for i in np.arange(len(A)) if A[i] != B[i]]

            if len(ordered_diff) == 0:
                difference = diff
                exit_code = 5

                # Report status
                message = f'Same elements exist in the same order'
                report_message(message, verbose=verbose, log=True, level='info')
            else:
                difference = ordered_diff
                exit_code = 4

                # Report status
                message = f'Same elements exist in different order.'
                report_message(message, verbose=verbose, log=True, level='info')
        else:
            ordered_diff = [A[i] for i in np.arange(len(A)) if A[i] != B[i]]
            difference = diff
            exit_code = 3

            # Report status
            message = f'Same length but different elements.'
            report_message(message, verbose=verbose, log=True, level='info')

    if (len(difference) == 0) and (longer is None):
        ValueError('Impossible situation. If there is no difference the two should have the same length.')

    return difference, longer, exit_code


def table_of_business_weeks(start="2019-01-01", end="2030-12-31"):
    """
    When no specified, beginning is a pd.Timestamp equal to date 2019-04-01
    Parameters
    ----------
    beginning

    Returns
    -------

    """

    # Get year of execution based on machine clock
    todays_year = datetime.now().year
    end_year = pd.Timestamp(end).year

    if todays_year > end_year:
        raise ValueError(f"Today's year looks to be larger than {end_year} which is the end year."
                         f"The end date needs to be bigger.")

    # Get the date range in a week
    W = pd.Timedelta(value=1, unit="W")

    end_dates = pd.date_range(start=start, end=end, freq="W")
    start_dates = end_dates - W

    # Export table
    frame = pd.DataFrame(data=start_dates, index=range(len(start_dates)), columns=["start_date"])
    frame["end_date"] = end_dates
    frame["year"] = frame["end_date"].dt.year
    frame["weekofyear"] = frame["end_date"].dt.weekofyear

    return frame


def week_start_end_dates(*, year, weekofyear):
    """
    Returns a dictionary corresponding to a line of the table from table_of_business_weeks().
    Parameters
    ----------
    year
    weekofyear

    Returns
    -------
    """

    bw = table_of_business_weeks()

    c1 = bw["year"] == year
    c2 = bw["weekofyear"] == weekofyear

    output = bw[c1 & c2].iloc[0].to_dict()

    return output


def exec_date_as_ts(**context):
    """
    Transforms Airflow execution date (pendulum object) to
    pd.Timestamp object which is easier to work with. For more info see
    https://stackoverflow.com/questions/55458891/airflow-error-with-pandas-attributeerror-
    pendulum-object-has-no-attribute-n
    Parameters
    ----------
    context     :   dict
                    Airflow execution context
    Returns
    -------
    output      :   pd.Timestamp
    """
    check_type(context, dict)

    time_pendulum = context["execution_date"]

    # Make sur ethe type is correct
    check_type(time_pendulum, pendulum.pendulum.Pendulum)

    # Extract unix time
    time_unix = time_pendulum.timestamp()

    # Make a datetime object out of it
    time_datetime = datetime.fromtimestamp(time_unix)

    # Transform to pd.Timestamp objecct
    output = pd.Timestamp(time_datetime)

    return output


def validate_past_window(window):

    check_type(window, str)
    allowed_windows = ["90-60d", "60-30d", "30-00d", "90d"]
    if window not in allowed_windows:
        raise ValueError(f"window {window} not allowed. Must be one of {allowed_windows}.")

    return True


def pq_schema(path, print=True):
    """
    Return the schema of a parquet file.
    Parameters
    ----------
    path        :   str
                    Absolute path to local parquet file
    Returns
    -------
    pq_schema   :   fastparquet.schema.SchemaHelper
    """
    pq_schema = ParquetFile(path).schema

    if print:
        for line in pq_schema.text.split("\n"):
            print(line)

    return pq_schema

"""
--------------------------------------- PARALLEL ------------------------------------------
-------------------------------------------------------------------------------------------------
"""


def max_processes():
    """
    Returns the number of available CPUs on the machine.
    Returns
    -------
    cpu_count   :   int
                    Number of available CPUs on the machine
    """
    cpu_count = mp.cpu_count()
    return cpu_count


def random_codes(n, length):
    """
    Returns a list of random strings of length=length
    Parameters
    ----------
    n       :   int
                How many random strings
    length  :   Number of characters in each string
    Returns
    -------
    my_list :   list
    """
    check_numeric(n)
    check_type(length, int)

    if not isinstance(n, int):
        n = int(n)

    if n > 1e6:
        my_logger.warning("n > 1e6 - computation might take some time.")

    alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    np_alphabet = np.array(alphabet)
    np_codes = np.random.choice(np_alphabet, [n, length])
    codes = ["".join(np_codes[i]) for i in range(len(np_codes))]

    return codes


def my_dummy_transform(*, name, number):

    check_type(name, str)
    print(f"Hey {name}, you are {number}")

    return True


def pid_message():
    message = f"Process id {os.getpid()}, children of {os.getppid()}:"
    return message


def validate_execution_map(*, f, execution_map):

    # Validate input types
    check_type(execution_map, pd.DataFrame)

    if not callable(f):
        raise ValueError(f"f should be a callable object.")

    # Check that the arguments of f
    inspection = inspect.getfullargspec(f)

    # Get all args
    all_args = inspection.args + inspection.kwonlyargs

    # With default value
    if inspection.kwonlydefaults:
        args_with_default = list(inspection.kwonlydefaults.keys())
    else:
        args_with_default = []

    # List all args that must be provided
    must_provide = [a for a in all_args if a not in args_with_default]

    # List all the arguments provided with this map
    provided = list(execution_map.columns.values)

    # Check if there is enough information
    missing = [a for a in must_provide if a not in provided]

    if missing:
        raise ValueError(f"column argument {missing} required to call function {f.__name__}.")

    return True


def parallel_exec_map(*, f, execution_map, workers, log=True):
    """
    "f" is executed as many times as there are rows in "execution_map",
    and its kwargs are extracted from the columns of "execution_map".
    V2 of this should use Queues!!
    Parameters
    ----------
    f
    execution_map
    workers

    Returns
    -------

    Example
    -------
    def wt(t):
        time.sleep(t)
        print(f"slept for {t} seconds.")
        return True

    m = pd.DataFrame(data=[1,1,1,1,1,1,1,1,1],index=range(9), columns=["t"])
    f = wt
    perform_execution_map(f=f, execution_map=m, n_cores=9)

    """
    validate_execution_map(f=f, execution_map=execution_map)

    # Spawn multi processes
    chunks = chunk_df(execution_map, workers)
    n_chunks = len(chunks)

    # Print status
    message = f"distributing {len(execution_map)} calls ({n_chunks} chunks) " \
              f"across {workers} workers for function {f.__name__}"
    report_message(message)

    # Init a counter for the chunks.
    chunk_counter = 1
    for submap in chunks:

        # Garbage collect
        gc.collect()
        jobs = []
        for row_index in range(len(submap)):

            # Extract kwargs from this tables raw
            kwargs = kwargs_from_df(df=submap, row=row_index)
            process_name = f"em-subprocess-{row_index}"
            p = mp.Process(name=process_name, target=f, kwargs=kwargs)
            jobs.append(p)
            p.start()

        # Wait for all of them to complete
        for job in jobs:
            job.join()

        # Report on progress
        message = f"completed {chunk_counter}/{n_chunks} chunks for function {f.__name__}"
        report_message(message, level='info', log=True)

        # Increment chunk counter
        chunk_counter += 1

    return True


def kwargs_from_df(*, df, row):
    """
    Converts the raw of a df into a dict, making sure that any numpy dtype
    be converted to python dtypes.
    Parameters
    ----------
    df    :   pd.DataFrame
    row   :   int
    Returns
    -------
    kwargs      :   dict
    """
    check_type(df, pd.DataFrame)
    check_type(row, int)

    # Get a list of the kwargs provided in the execution_map
    kwargs_provided = list(df.columns.values)

    # Extract kwargs from this tables raw
    ka_list = []
    for key in kwargs_provided:
        value = df.iloc[row][key]

        if isinstance(value, np.int64):
            value = int(value)

        elif isinstance(value, np.bool_):
            value = bool(value)

        this_tuple = (key, value)
        ka_list.append(this_tuple)

    kwargs = dict(ka_list)
    return kwargs


def chunk_list(my_list, n):
    """
    Generator who yields the next sub-list of length n, given my_list.
    Parameters
    ----------
    my_list
    n
    Returns
    -------
    output  :   list
    """
    check_type(my_list, list)

    def my_generator(l):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    output = list(my_generator(my_list))
    return output


def chunk_df(df, n):
    """
    Generator who yields the next sub-list of length n, given my_list.
    Parameters
    ----------
    my_list
    n
    """
    check_type(df, pd.DataFrame)

    def my_generator(df):
        for i in range(0, len(df), n):
            yield df.iloc[i:i + n, :]

    output = list(my_generator(df))
    return output


def parallel_my_transform(all_names):

    check_type(all_names, list)

    # 1. Divide input into chunks
    max_parallel = max_processes() - 2
    all_names_chunks = list(chunk_list(all_names, max_parallel))

    # 2. For each chunk and each element in chunk
    counter = 0
    for this_chunk in all_names_chunks:
        for name in this_chunk:

            # 3. APPLY my_dummy_transform()

            keyword_args = {"name": name, "number": np.random.randint(1000)}
            processes = mp.Process(target=my_dummy_transform, kwargs=keyword_args)

            processes.start()
            print("Start done")

            processes.join()
            print("Join done")
            counter += 1

    print(f"Counter={counter}")

    return True


def split_numpy_array(array, n):

    # Takes a numpy array and splits into as many parts
    # as the number of processes you will spawn

    list_of_arrays = np.array_split(array, n)

    return list_of_arrays


def subprocess_cmd(command):
    """
    Subshells and runs the bash command
    Parameters
    ----------
    command :   str
                The bash command you want to run
    Returns
    -------
    output  :   list
                A list of strings each containing the line redirected to stdout
    """
    # Open a process pool in shell mode and pipe stdout to output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    # Grub stdout and remove leading and trailing spaces
    proc_stdout = process.communicate()[0].strip()

    # Decode the binary usign utf-8 and split on new line creating a list of strings
    output = proc_stdout.decode("utf-8").split('\n')

    return output



"""
--------------------------------------- BIG-QUERY ------------------------------------------
-------------------------------------------------------------------------------------------------
"""


def gbq_list_datasets(*, project_id):

    client = bigquery.Client(project=project_id)
    datasets = list(client.list_datasets())
    project = client.project

    all_datasets = [d.dataset_id for d in datasets]

    if all_datasets:
        print(f"project {project} has following datasets {all_datasets}")
    else:
        print(f"project {project} has no datasets")

    return all_datasets


def gbq_create_dataset(*, dataset_id, location, project_id=None):
    """

    Parameters
    ----------
    project_id
    dataset_id
    location

    Returns
    -------

    """
    check_type(project_id, [type(None), str])
    check_type(dataset_id, str)
    check_type(location, str)

    if not project_id:
        project_id = env_get_gcp_project_id()

    # Obtain bigquery client
    client = bigquery.Client(project=project_id)

    # Create a DatasetReference using a chosen dataset ID.
    # The project defaults to the Client's project if not specified.
    dataset_ref = client.dataset(dataset_id)

    # Construct a full Dataset object to send to the API.
    dataset = bigquery.Dataset(dataset_ref)

    # Specify the geographic location where the dataset should reside.
    dataset.location = location

    # Send the dataset to the API for creation.
    # Raises google.api_core.exceptions.AlreadyExists if the Dataset already
    # exists within the project.
    dataset = client.create_dataset(dataset)  # API request

    return dataset


def gbq_list_tables_in_dataset(*, projectId, datasetId, log=True):
    """

    Parameters
    ----------
    projectId   :   str
    datasetId   :   str

    Returns
    -------
    tables      :   list
    """
    check_type(projectId, str)
    check_type(datasetId, str)

    # Get BigQuery Client
    client = bigquery.Client(project=projectId)

    # Get all tables in dataset
    tables = client.list_tables(datasetId)

    # Get all table full names in dataset
    tables = [f"{table.project}.{table.dataset_id}.{table.table_id}" for table in tables]

    message = f"Found {len(tables)} tables in dataset {datasetId}"
    report_message(message, level="info", log=log)

    return tables


def gbq_table_exist_in_dataset(*, projectId, datasetId, tableId):

    check_type(projectId, str)
    check_type(datasetId, str)
    check_type(tableId, str)

    all_tables = gbq_list_tables_in_dataset(projectId=projectId, datasetId=datasetId)

    if f"{projectId}.{datasetId}.{tableId}" in all_tables:
        return True
    else:
        return False


def gbq_ingestion_schema_for_partitioned_table(df, partition_column_name="date"):

    schema = generate_bq_schema(df)
    schema_table = pd.DataFrame(schema['fields'])
    schema_table.loc[schema_table["name"] == partition_column_name, "type"] = "DATE"

    return schema_table


def gbq_load_parquet_from_gcs(*, gcs_path, project_id, dataset_id, table_id, part_column="date"):
    """
    Loads a .parquet asset from GCS to BigQuery. `part_column` is the column on which BigQuery table `table_id`
    is partitioned and should be a timestamp type. It is sufficient to make sure that the pandas table used to
    write the parquet file with fastparquet has a column `part_column` of type pd.DateTimeIndex.
    Parameters
    ----------
    gcs_path        :   str
    project_id      :   str
    dataset_id      :   str
    table_id        :   str
    part_column     :   str, None
    """

    check_type(gcs_path, str)
    check_type(project_id, str)
    check_type(dataset_id, str)
    check_type(table_id, str)
    check_type(part_column, [str, type(None)])

    if os.path.splitext(gcs_path)[-1] != ".parquet":
        raise ValueError(f"gcs_path is expected to be a file with extension .parquet, passed {gcs_path}")

    if not gcs_path.startswith("gs://"):
        raise ValueError(f"gcs_path should be a GCS uri starting with gs://, passed {gcs_path}")

    # Open client
    client = bigquery.Client(project=project_id)

    # Get dataset reference
    dataset_ref = client.dataset(dataset_id)

    # Configure transfer job
    job_config = bigquery.LoadJobConfig()

    # Set source format as parquet
    job_config.source_format = bigquery.SourceFormat.PARQUET

    if part_column:

        # Configure partition
        daily_partition = bigquery.TimePartitioningType.DAY
        job_config.time_partitioning = bigquery.TimePartitioning(type_=daily_partition, field=part_column)

        # Report status
        report_message(level="info", log=True,
                       message=f"gcp project {project_id} - ingesting {gcs_path} "
                               f"\n\tto {dataset_id}.{table_id} (table partitioned "
                               f"by {part_column})")
    else:
        # Report status
        report_message(level="info", log=True,
                       message=f"gcp project {project_id} - ingesting {gcs_path} "
                               f"\n\tto {dataset_id}.{table_id} (non-partitioned table)")

    # Prepare load job
    load_job = client.load_table_from_uri(gcs_path, dataset_ref.table(table_id), job_config=job_config)  # API request
    print(f"Starting job {load_job.job_id}")

    # Waits for table load to complete.
    load_job.result()
    print('Job finished.')

    destination_table = client.get_table(dataset_ref.table(table_id))
    print(f"Loaded {destination_table.num_rows} rows.")

    return True


def gbq_table_consolidate(*, view):
    """
    If exists it updates it. If id does not exists, it creates it.
    Parameters
    ----------
    view        :   bigquery.table.Table

    """
    check_type(view, bigquery.table.Table)

    gcp_project = view.project
    gcp_dataset = view.dataset_id
    view_name = view.table_id

    # Open client
    gbq_client = bigquery.Client(project=gcp_project)

    # Check if view exists
    if gbq_table_exist_in_dataset(projectId=gcp_project, datasetId=gcp_dataset, tableId=view_name):

        # Update view
        _ = gbq_client.update_table(view, ["view_query"])

        # Save state
        message = f"View for {gcp_project}.{gcp_dataset}.{view_name} updated" \
                  f"because table name found in dataset."
    else:
        # Create view
        _ = gbq_client.create_table(view)

        # Save state
        message = f"View for {gcp_project}.{gcp_dataset}.{view_name} created"

    # Report status
    report_message(log=True, level="info", message=message)

    return True


def gbq_instantiate_table_object(*, name, is_view=False):
    """

    Parameters
    ----------
    name        :   str
    is_view     :   bool
    Returns
    -------
    table       :   bigquery.table.Table
    """
    check_type(name, str)

    # Enforce naming convention if is a view
    if is_view:
        if name[:2] != "v_":
            raise ValueError(f"By convention the name of a view should start with v_, passed {name}")

    # Extract env variables for GCP project, dataset and location
    gcp_project = env_get_gcp_project_id()
    gcp_dataset = env_get("GBQ_DATASET_NAME")
    gcp_location = env_get("GBQ_DATASET_LOCATION")

    # Get BigQuery Client
    client = bigquery.Client(project=gcp_project)

    # Get Reference to Dataset
    dataset_ref = client.dataset(gcp_dataset)

    # Get Reference to table
    table_ref = dataset_ref.table(name)

    # Prepare table associated to the table
    table = bigquery.Table(table_ref)

    return table
