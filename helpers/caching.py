from skepsi.utils.utils import *
from skepsi.utils.imports import *

http = urllib3.PoolManager()

"""
Define utility functions to reach Showerhead endpoint and
communicate that it is time to cache the results.
"""


class ShowerheadResetCacheError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


def validate_insight_name(name):

    check_type(name, str)

    all_insights = ['email-anomaly', 'goal-performance', 'smart-delay',
                    'smart-content', 'smart-interval']

    if name not in all_insights:
        raise ValueError(f"Insight name expected to be one "
                         f"of {all_insights}, passed {name}")
    else:
        return True


def reset_redis_cache(*, insight, **context):
    """
    Posts to http endpoint to set the Redis cache used by
    Torpedo and other Autopilot services.
    Parameters
    ----------
    insight     :   str
    context     :   dict
    """

    # Get execution date in string format YYYY-MM-DD
    # See https://airflow.apache.org/code.html#macros
    date_to_reset = context["ds"]
    check_type(date_to_reset, str)

    # Make sure the insight name passed is valid
    validate_insight_name(insight)

    # If dev env do not set any cache (no showerhead dev available)
    this_env = os.environ.get("ENV")

    if this_env == "DEV":
        message = f"Redis cache not reset because detected environment is {this_env}"
        report_message(message, level='warning', log=True)
        pass
    else:
        try:
            # Parse cache info from env
            cache_data = read_cache_env_variables()
            url = cache_data["sh_reset_cache_url"]

            headers = {'Content-Type': cache_data["sh_content_type"],
                       'Authorization': "Bearer " + cache_data["sh_auth_key"]}

            body = {"invalidDate": date_to_reset,
                    "insight": insight}

            body = json.dumps(body).encode('utf-8')

            message = f"POST request to url={url}, header={headers}, body={body}"
            report_message(message, level='info', log=True)

            # Get response from endpoint
            response = http.request('POST', url, headers=headers, body=body)
            status = response.status

        except Exception as ex:

                message = f"POST request to url={url}, header={headers}, " \
                          f"body={body} returned status {status} and error {ex}"

                report_message(message, level="info", log=True)
                raise ShowerheadResetCacheError(message)

    return True


def read_cache_env_variables():

    # Read environment variables
    env_variables = {"sh_reset_cache_url": os.environ.get("SHOWERHEAD_RESET_CACHE_URL", None),
                     "sh_content_type": os.environ.get("SHOWERHEAD_CONTENT_TYPE", None),
                     "sh_auth_key": os.environ.get("SHOWERHEAD_AUTH_KEY", None)}

    # Validate env variables
    try:
        for key, value in env_variables.items():
            check_type(value, str)

    except Exception as ex:
        my_logger.error(f"Invalid credentials for {key}, "
                        f"Check config file. Error: {ex}")

    return env_variables
