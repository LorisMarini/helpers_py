from helpers.utils import *
from helpers.imports import *


class ETLDefaults:

    def __init__(self):

        self.exist_ok = True
        self.log = True


class ETLObject:

    def __init__(self, bucket_id="gs://", consolidate_temp=True, *, insight, stage, version, date_time, name, ext):

        check_type(consolidate_temp, bool)
        check_type(insight, str)
        check_type(stage, str)
        check_type(version, str)
        check_type(name, str)
        check_type(ext, str)

        self.bucket_id = bucket_id
        self.date_time = date_time
        self.insight = insight
        self.stage = stage
        self.version = version
        self.name = name
        self.ext = ext
        self.log = ETLDefaults().log
        self.exist_ok = ETLDefaults().exist_ok
        self.missing_src_ok = True

        self.__date_string = None
        self.__consolidate_temp = consolidate_temp
        self.__base_name = None
        self.__temp_home = None
        self.__temp_dir = None
        self.__temp_path = None
        self.__prod_home = None
        self.__prod_dir = None
        self.__prod_path = None

        # Initialize object
        self.initialize_object()

    @property
    def date_time(self):
        return self._date_time

    @date_time.setter
    def date_time(self, value):

        check_type(value, [str, datetime, type(pendulum.date(2019, 3, 1))])

        if isinstance(value, datetime):
            # Both pd.Timestamp and pendulum objects are children of datetime type
            self._date_time = value.date()

        else:
            # Passed string
            self._date_time = value

    @property
    def date_string(self):
        return self.__date_string

    @property
    def consolidate_temp(self):
        return self.__consolidate_temp

    @property
    def base_name(self):
        return self.__base_name

    @property
    def temp_home(self):
        return self.__temp_home

    @property
    def temp_dir(self):
        return self.__temp_dir

    @property
    def temp_path(self):
        return self.__temp_path

    @property
    def prod_home(self):
        return self.__prod_home

    @property
    def prod_dir(self):
        return self.__prod_dir

    @property
    def prod_path(self):
        return self.__prod_path

    def __str__(self):

        attrs = vars(self)
        string = ''
        for k, v in attrs.items():

            string = string + f'{k} = {v} \n'
        return string

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

    def initialize_object(self):

        # Validate input
        self.validate_etlid()

        if isinstance(self.date_time, datetime) or isinstance(self.date_time, type(pendulum.date(2019, 3, 1))):

            # Both pd.Timestamp and pendulum objects are children of datetime type
            self.__date_string = self.date_time.strftime('%Y-%m-%d')

        elif isinstance(self.date_time, str):
            # Pass datetime as is (already a string)
            self.__date_string = self.date_time
        else:
            raise ValueError(
                f"expected string or datetime found {type(self.date_time)}")
        # Build temp and prod absolute paths
        self.path_build(path_type="temp")
        self.path_build(path_type="prod")

        # Make sure there is a directory on disk if __consolidate_temp = True
        if self.__consolidate_temp:
            self.temp_consolidate()

        return self

    def update_attributes(self, **kwargs):

        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

        self.initialize_object()

        return self

    def validate_etlid(self):
        """
        """
        no_underscore = ["insight", "stage",
                         "version", "name"]

        for attribute in no_underscore:
            if "_" in self.__dict__[attribute]:

                # Prepare error message
                message = f"'_' found in attribute {attribute}, invalid."
                raise ValueError(message)

        if not self.ext.startswith('.'):

            # Prepare error message
            message = f"ext must start with a dot, passed {self.ext}"
            raise ValueError(message)

    def path_build(self, path_type="temp"):
        """
        """
        rel_dir_values = [self.insight, self.stage,
                          self.version, self.date_string]
        abs_dir_values = rel_dir_values + [self.name]

        self.__base_name = "_".join(abs_dir_values) + self.ext

        # Get home directory
        if path_type == "temp":

            self.__temp_home = etl_local_home()
            # Get object directory
            self.__temp_dir = os.path.join(
                self.__temp_home, "/".join(rel_dir_values))
            # Get object absolute temp path
            self.__temp_path = os.path.join(self.__temp_dir, self.__base_name)

        elif path_type == "prod":

            bucket = airflow_bucket_from_id(self.bucket_id)
            self.__prod_home = bucket.products
            # Get object directory
            self.__prod_dir = os.path.join(
                self.__prod_home, "/".join(rel_dir_values))
            # Get object absolute temp path
            self.__prod_path = os.path.join(self.__prod_dir, self.__base_name)

        else:
            message = f"path_type can only be one of {['temp', 'prod']}"
            raise ValueError(message)

    def temp_consolidate(self):
        """"""
        path_consolidate(self.__temp_path)

    def temp_clean(self):
        """If self.__temp_dir exists it removes it"""
        if path_exists(self.__temp_dir):

            shutil.rmtree(self.__temp_dir)

            message = f"Temp directory {self.__temp_dir} removed"
            report_message(message, log=True, level='info')

    def clean_older_objects(self, days_keep=100):
        """
        Looks inside the parent directory of this object's temp directory. If
        a directory is found with a date
        """

        oldest_date_on_disk = pd.Timestamp(
            self.date_time) - pd.Timedelta(days_keep, unit='d')
        object_parent_temp_dir = os.path.dirname(self.temp_dir)

        # Describe parent temp dir
        on_disk = pd.DataFrame(
            {"dirs": glob.glob(f"{object_parent_temp_dir}/*")})
        on_disk["dates_str_on_disk"] = on_disk["dirs"].apply(
            lambda x: os.path.basename(x))

        try:
            on_disk["dates_on_disk"] = pd.DatetimeIndex(
                on_disk["dates_str_on_disk"])
        except ValueError as ex:
            message = f"could not transform directory basenames into pd.DatetimeIndex becasue of error {ex}." \
                      f"Make sure that parent dir of sel.temp_dir contains dirs named after a date string"
            report_message(message, level='error', log=self.log)

        # Filter only those older then oldest_date_on_disk
        to_remove = on_disk[on_disk["dates_on_disk"] < oldest_date_on_disk]

        message = f"recursively removig {len(to_remove)} directories on disk"
        report_message(message, level='info', log=self.log)

        # Remove recursively all dirs to remove
        to_remove["dirs"].apply(lambda x: shutil.rmtree(x))

        return True

    def make_collection(self, *, days, direction):
        """
        Returns a list of objects of the same class, for which the date_time attribute and
         all attributes correlated are updated according to the number of days
         in the projection and the direction.
        Parameters
        ----------
        days        :   int
        direction   :   str
                        Must be one of ["past", "future"]
        Returns
        -------
        output      :   ETLObjectsCollection
                        List of ETLObjects
        """
        check_type(days, int)
        check_type(direction, str)

        if days <= 1:
            raise ValueError("days must be a positive integer, grater than 1.")

        allowed_directions = ["past", "future"]
        if direction not in allowed_directions:
            raise ValueError(
                f"direction {direction} invalid. Should be one of {allowed_directions}")

        # Get date range (first/last day)
        date_a = pd.Timestamp(self.date_time)
        time_window = pd.Timedelta(value=days - 1, unit='d')

        if direction == "past":
            date_b = date_a - time_window
            date_range = pd.date_range(start=date_b, end=date_a, freq='d')

        else:
            # direction = "future"
            date_b = date_a + time_window
            date_range = pd.date_range(start=date_a, end=date_b, freq='d')

        object_list = []
        for this_date in date_range:
            this_copy = copy.deepcopy(self)
            this_date_string = this_date.strftime('%Y-%m-%d')
            object_list.append(this_copy.update_attributes(
                date_time=this_date_string))

        # Put into an ELT collection object and return it
        output = ETLObjectCollection(etl_objects=object_list)

        return output

    def extract(self, nosrc_ok=True, use_cache=False):

        src = self.prod_path
        dest = self.temp_path
        obj_extract(src, dest, nosrc_ok=nosrc_ok,
                    use_cache=use_cache, log=self.log)

        return True

    def load(self, **kw):

        src = self.temp_path
        dest = self.prod_path
        obj_load(src, dest, **kw)

        return True

    def expand(self, dest=None, compression="bz2", nosrc_ok=True,
               use_cache=True, expected_ext=None, force_src_ext=True):

        src = self.temp_path
        # Untar and uncompress archive_temp
        untar_and_uncompress_path(src=src, dest=dest,
                                  compression=compression, force_src_ext=force_src_ext,
                                  log=self.log, nosrc_ok=nosrc_ok,
                                  use_cache=use_cache, expected_ext=expected_ext)
        return True

    def archive_temp(self, *, dest, files=None):
        "Archives the self.temp_dir to destination absolute path"

        check_type(files, [type(None), list])

        if files is None:

            # Compress all files in temp directory to dest path
            tar_and_compress_path(src=self.temp_dir, dest=dest)
        else:
            # Compress only selected files to dest path
            tar_and_compress_files(
                src_list=files, dest=dest, compression="bz2")

        return True

    def poke_prod_path(self, interval=10):

        # Get path to poke
        to_poke = self.prod_path
        while not path_exists(to_poke):
            time.sleep(interval)
        return True

    def get_object_url(self, bucket_location="ap-southeast-2"):

        if path_exists(self.prod_path):

            # Determine the base url based on the region
            aws_base_url = f"https://s3-{bucket_location}.amazonaws.com/"

            # Construct the url by replacing the S3:// part
            image_url = self.prod_path.replace("s3://", aws_base_url)

            return image_url
        else:
            raise ValueError(
                f"Object does not exist at remote location {self.prod_path}")


class ETLObjectCollection:

    def __init__(self, etl_objects):

        # Check input type
        check_type(etl_objects, list)

        # All elements of the list must be instances of ETLObject
        for obj in etl_objects:
            check_type(obj, ETLObject)

        # Set attributes
        self.objects = etl_objects
        self.size = len(etl_objects)
        self.log = ETLDefaults().log

    def extract(self):

        dates_left = self.size
        for obj in self.objects:

            obj.extract()

            # Report remaining dates
            dates_left -= 1
            message = f"{dates_left} remaining dates"
            report_message(message, log=self.log, level='info')

        message = "Objects extracted. Done."
        report_message(message, log=self.log, level='info')
        return True

    def expand(self):

        dates_left = self.size
        for obj in self.objects:

            obj.expand()

            # Report remaining dates
            dates_left -= 1
            message = f"{dates_left} remaining dates"
            report_message(message, log=self.log, level='info')

        message = "expand done."
        report_message(message, log=self.log, level='info')

        return True

    def temp_clean(self):

        for obj in self.objects:
            obj.temp_clean()

        return True


def past_contexts(*, n_days, log=True, **context):
    """
    Generates a list of dicts, containing keys "ds" and "date" for the n_days past dates.
    "ds" is the date in format "%Y-%m-%d" while "date" is the date as a datetime object.
    Parameters
    ----------
    n_days
    context

    Returns
    -------
    context_list    :   list
                        of dicts
    """
    check_type(n_days, int)

    if n_days <= 1:
        raise ValueError("n_days must be a positive integer, grater than 1.")

    # Get date range (first/last day)
    last_date = pd.Timestamp(context["ds"])
    time_window = pd.Timedelta(value=n_days - 1, unit='d')

    # Get first date
    first_date = last_date - time_window

    # Build Series of dates with daily frequency
    date_range = pd.date_range(start=first_date, end=last_date, freq='d')

    # Initialize the list
    context_list = []

    # Append a new context for each day
    for this_date in date_range:

        # Format date as YYYY-MM-DD
        this_ds = this_date.date().strftime('%Y-%m-%d')

        # Append to list of contexts
        context_list.append({"ds": this_ds,
                             "date": this_date.date()})

    message = f"Returned a total of {len(context_list)} contexts."
    report_message(message, log=log, level='info')

    return context_list


# ------------------ INTEGRATE SLACK ALERTS------------------


def task_fail_slack_alert(context):
    """
    Define a general function to send the alert messags to Slack channel
    when a task in a DAG is failed
    """
    message = f"""
                :circleci-fail: Data pipeline task failed.
                *Task*: {context.get('task_instance').task_id}
                *Dag*: {context.get('task_instance').dag_id}
                *Execution Time*: {context.get('execution_date')}
                *Log Url*: {context.get('task_instance').log_url}
                """

    slack_post_kwargs = {"message": message,
                         "channel_type": "ALERTS",
                         "text_type": "markdown"}

    f = post_message_to_slack
    Operator = PythonOperator(task_id=f.__name__, python_callable=f, trigger_rule="all_success",
                              op_kwargs=slack_post_kwargs, provide_context=True)

    return Operator.execute(context=context)


# ------------------ Resolve name of slack channel ----------------

def slack_notifications_channel():

    # Get a list of valid env names
    valid_env_names = ["DEV", "STG", "PROD"]

    # Extract current environment name
    this_env = os.environ.get("ENV")

    if this_env == "DEV":
        slack_channel = '#data-pipeline-dev'

    elif this_env == "PROD":
        slack_channel = '#data-pipeline-alerts'

    elif this_env == "STG":
        slack_channel = '#data-pipeline-stg'
    else:
        raise ValueError(
            f"ENV name invalid. Expected one of {valid_env_names}, passed {this_env}.")

    return slack_channel


def s3_bucket_location():

    # Get a list of valid env names
    valid_env_names = ["DEV", "STG", "PROD"]

    # Extract current environment name
    this_env = os.environ.get("ENV")

    if this_env == "DEV":
        # Dev S3 bucket location
        bucket_location = "ap-southeast-2"

    elif this_env == "PROD":
        # Prod S3 bucket location
        bucket_location = "us-east-2"

    elif this_env == "STG":
        # Staging S3 bucket location
        bucket_location = "us-east-2"

    else:
        raise ValueError(
            f"ENV name invalid. Expected one of {valid_env_names}, passed {this_env}.")

    return bucket_location


def post_message_to_slack(*, message, text_type="plain_text", channel_type="ALERTS", **kwargs):
    """
    CAREFUL with refactoring arguments, as this might be called in an Airflow Python Operator
    and kwargs passed not be refactored.
    Parameters
    ----------
    message         :   str
    text_type       :   str
    channel_type    :   str
                        Must satisfy validate_slack_channel_type()
    """
    check_type(message, str)
    check_type(channel_type, str)

    # Validate channel type and text type
    validate_slack_channel_type(channel_type)
    validate_slack_text_type(text_type)

    # Get the url to post in slack (including of token)
    slack_url = env_get(f"SLACK_{channel_type}_URL", required=True)

    # Ge the slack channel for this token
    slack_channel = env_get(f"SLACK_{channel_type}_CHANNEL", required=True)

    # Build a block to display in slack
    message_block = {"type": "section", "text": {
        "type": "plain_text", "text": message}}

    # Prepare body
    request_body = {"channel": slack_channel, "blocks": [message_block]}

    # encode body
    enc_body = json.dumps(request_body).encode('utf-8')

    # Post
    http = urllib3.PoolManager()
    response = http.request('POST', slack_url, headers={
                            'Content-Type': "application/json"}, body=enc_body)

    # Get response
    body_content = response.data.decode('utf-8')
    error_code = response.status

    if error_code > 200:
        message = f"POST request with token: {slack_url} and body: {request_body} " \
                  f"failed with error code: {error_code}. Returned message: {body_content}"
        raise ValueError(message)

    return True


def post_image_to_slack(etlid, channel_type="REPORTS"):
    """

    Parameters
    ----------
    etlid

    Returns
    -------

    """
    check_type(etlid, ETLObject)

    # Validate channel type
    validate_slack_channel_type(channel_type)

    # Get the url to post in slack (including of token)
    slack_url = env_get(f"SLACK_{channel_type}_URL", required=True)

    # Ge the slack channel for this token
    slack_channel = env_get(f"SLACK_{channel_type}_CHANNEL", required=True)

    # Get url from object id
    image_url = etlid.get_object_url(bucket_location=s3_bucket_location())

    if path_exists(etlid.prod_path):

        # Validate file extension
        (fn, file_extension) = path_splitext(image_url)

        if file_extension != ".png":
            raise ValueError(
                f"Only .png images can be posted on slack, passed {file_extension}")

        # Get file base name to be used as image name
        image_name = os.path.basename(image_url)

        # Prepare title payload
        title_dict = {"type": "plain_text", "text": image_name}

        # Build a block to display in slack
        message_block = {"type": "image", "image_url": image_url,
                         "alt_text": image_name, "title": title_dict}

        # Prepare body
        request_body = {"channel": slack_channel,
                        "blocks": [message_block]}
        # encode body
        encoded_body = json.dumps(request_body).encode('utf-8')

        # Post
        http = urllib3.PoolManager()
        response = http.request('POST', slack_url, headers={'Content-Type': "application/json"},
                                body=encoded_body)

        # Get response
        body_content = response.data.decode('utf-8')
        error_code = response.status

        if error_code > 200:
            message = f"POST request with token: {slack_url} and body: {request_body} " \
                      f"failed with error code: {error_code}. Returned message: {body_content}"
            raise ValueError(message)

    else:
        m = "no image found in bucket. Posting to Slack skipped."
        report_message(m, level='warning', log=True)

    return True


def less_instances_if_dev(all_instances):
    """
    If ENV="DEV" return a list of instances as indicated in DEV_SUB_LIST.
    Otherwise return all_instances.
    Parameters
    ----------
    all_instances   :   list

    Returns
    -------
    output       :   list
    """
    check_type(all_instances, list)

    # Reduce for DEV
    if os.environ.get("ENV") == "DEV":
        # Get list of instances from DEV_SUB_LIST
        output = os.environ.get("DEV_SUB_LIST").split(",")
    else:
        output = all_instances

    return output


def validate_slack_text_type(text_type):
    """
    See https://api.slack.com/reference/messaging/composition-objects#text
    Parameters
    ----------
    message_type

    Returns
    -------

    """
    check_type(text_type, str)

    allowed_types = ["plain_text", "markdown"]

    if text_type not in allowed_types:
        raise ValueError(
            f"The type fo Slack 'text' objects can only be one of {allowed_types}")

    return True


def validate_slack_channel_type(channel_type):
    """
    Ensures that channel_type is one of a strict range of values. To allow for another type "newtype",
    make sure that the following environment variables exist at runtime. If they don't an error is raised.
        SLACK_{newtype}_URL
        SLACK_{newtype}_CHANNEL

    Parameters
    ----------
    channel_type    :   str
                        Uppercase keyword one of ["ALERTS", "REPORTS"]

    """
    check_type(channel_type, str)

    allowed_types = ["ALERTS", "REPORTS"]
    if channel_type not in allowed_types:
        raise ValueError(
            f"The type of slack channel can only be one of {allowed_types}")

    return True


def pq_merge_files(*, paths, etlid_out, use_cache=True, instance=None,
                   drop_index=False, attribute=False, verbose=False, log=True):
    """
    Creates a larger parquet file {saveas} starting from a list of smaller files in {paths}.

    Parameters
    ----------
    paths
    etlid_out   :   ETLObject
    use_cache   :   bool
    instance    :   str
    drop_index  :   bool
    verbose     :   bool
    log         :   bool

    Returns
    -------

    """
    check_type(paths, list)
    check_type(etlid_out, ETLObject)
    check_type(use_cache, bool)
    check_type(instance, [type(None), str])
    check_type(log, bool)

    # Extension of all files must be `.parquet`
    _ = [check_file(pq_part, ext=".parquet") for pq_part in paths]

    # Temporary directory of output
    output_temp = etlid_out.temp_path

    if path_exists(output_temp) and use_cache:

        message = f"found file {output_temp}, merge skipped as use_cache={use_cache}"
        report_message(message, level="info", log=True)

        # Load record to bucket
        etlid_out.load()
        return True

    # Initialize counters
    n_empty, n_not_empty = 0, 0

    for i, pq_part in enumerate(paths):

        # Load part in RAM
        df_part = df_from_parquet(
            path=pq_part, verbose=False, log=False, get_size=False)

        if df_part.empty:
            n_empty += 1  # Increment counter of empty files
        else:
            n_not_empty += 1  # Increment counter of non empty files

            # Do not append for first (non empty) file
            append = False if (n_not_empty == 1) else True

            if drop_index:
                df_part = df_part.reset_index(drop=True)

            if attribute:
                instance_name = get_instance_name_from_path(pq_part)
                df_part["instance"] = instance_name

            # Write to file
            fastparquet.write(
                output_temp, df_part, file_scheme='simple', compression="snappy", append=append)

    # Report
    message = f"{n_not_empty}/{len(paths)} parquet files " \
              f"merged for instance {instance}, ({n_empty} were empty files)"
    report_message(message, level="info", log=log, verbose=verbose)

    # It is not given that file exists... if it does load it
    if path_exists(output_temp):
        etlid_out.load()

    return True


def sqlalchemy_url_pgdb(*, username, password, host, database=None, test_connection=False, verbose=False):
    """
    Returns the SQLAlchemy string needed to talk to the PostgreSQL database.
    Parameters
    ----------
    username    :   str
    password    :   str
    host        :   str
    database    :   str, None
    verbose     :   bool
                    Print string if True
    Returns
    -------
    SQLAlchemy_string   :   str
                            Optionally returns it if the connection to the DB is successful.
    """
    check_type(username, str)
    check_type(password, str)
    check_type(host, str)
    check_type(database, [str, type(None)])

    # Name of the driver to use to talk to the database
    drivername = 'postgres'

    if database:
        database = database
    else:
        # Default value for db name if user does not specify anything
        database = 'postgres'

    postgres_db = {'drivername': drivername,
                   'username': username,
                   'password': password,
                   'host': host,
                   'port': 5432,
                   'database': database}

    SQLAlchemy_string = sqlalchemy_url(**postgres_db)
    db = sqlalchemy.create_engine(SQLAlchemy_string)

    if test_connection:

        # Attempt connection to database
        db.connect()

    # If connection successful print the SQLAlchemy string for this DB
    print(SQLAlchemy_string) if verbose else None

    return SQLAlchemy_string


def write_error_file(*, directory, fn, m):
    """

    Parameters
    ----------
    directory   :   str
    fn          :   str
    m           :   str

    Returns
    -------
    file_path   :   str
                    The absolute path to the error file
    """
    check_dir(directory)
    check_type(fn, str)
    check_type(m, str)

    file_path = os.path.join(directory, f"error-file-{fn}.txt")

    with open(file_path, 'w') as error_file:
        error_file.write(m)

    return file_path


def df_in_case_of_error(message):

    df = pd.DataFrame(message, columns=["error"], index=[0])

    return df
