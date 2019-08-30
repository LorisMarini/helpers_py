from helpers.utils import *
from helpers.imports import *


def test_environment(log=False):
    """Makes sure that all required env variables are set properly"""

    for name in required_variables:
        _ = env_get(name, required=True)

        # Log if required
        message = f"env variable {name} properly set."
        report_message(message, level='info', log=log)

    return True


# TODO json_env_variable_to_file(env_name, filename=None)

# TODO set_python_env_from_file(filename)

# TODO show_env_values_and_types()

# TODO cloudant_creds()

# TODO env_get(env, required=True)

# TODO etl_remote_home(bucket_name="GS")

# TODO etl_local_home()
