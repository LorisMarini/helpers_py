from skepsi.utils.utils import *
from skepsi.utils.imports import *


def test_environment(log=False):
    """Makes sure that all required env variables are set properly"""

    for name in required_variables:
        _ = env_get(name, required=True)

        # Log if required
        message = f"env variable {name} properly set."
        report_message(message, level='info', log=log)

    return True


# json_env_variable_to_file(env_name, filename=None)

# set_python_env_from_file(filename)

# show_env_values_and_types()

# cloudant_creds()

# env_get(env, required=True)

# etl_remote_home(bucket_name="GS")

# etl_local_home()
