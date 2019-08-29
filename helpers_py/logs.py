from skepsi.utils.utils import check_type
from skepsi.utils.imports import *


def initialize_logger(name='root', level='INFO', file_handler=None, console_level=None):
    """
    Instantiates an my_logger.my_logger object
    Parameters
    ----------
    name            :   str
                        Name of the my_logger. This is used in the formatting of the log messages, and
                        helps parsing different logs.
    level           :   str
                        Must be one of DEBUG,  INFO, WARNING, ERROR, CRITICAL (see my_logger library)
    file_handler    :   dict
                        A actions_dict with 'abspath' and 'level' to specify the type of logs saved
                        to file. If this is specified, then a my_logger.FileHandler object is created accordingly
                        and passed to the my_logger object so that logs are dumped to file directly.
    console_level   :
    Returns
    -------

    """
    check_type(name, str)
    check_type(level, str)
    check_type(file_handler, [dict, type(None)])
    check_type(console_level, [str, type(None)])

    # create standard formatter (the way logged messages are displayed)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create my_logger (either root or personalized one)
    logger = logging.getLogger('') if name is 'root' else logging.getLogger(name)

    # Change its my_logger level to 'level'
    change_logging_level(level, logger=logger)

    # If user specified a file_handler, take
    # - 'abspath' which is the location on disk where the log file will be saved
    # - 'level' which is the my_logger level (degree of severity that gets recorded in log file)

    if file_handler:

        logger = set_logging_handler(logger, handler_dict=file_handler, formatter=formatter)

    if console_level:

        # Initialize default StreamHandler for console
        ch = logging.StreamHandler()

        # Update its level as specified
        change_logging_level(console_level, logger=ch)

        # set format of log message
        ch.setFormatter(formatter)

        # add the handler to the my_logger
        logger.addHandler(ch)

    return logger


def change_logging_level(new_level, logger=None):
    """
    If a logger other than root is specified it changes its logging level,
    otherwise it changes the level of the root logger.
    Parameters
    ----------
    new_level   :   str
                    Must be one of DEBUG,  INFO, WARNING, ERROR, CRITICAL (see my_logger library)
    logger      :   logging.Logger
                    Any object of type my_logger (my_logger.FileHandler, my_logger.root ...)
    Returns
    -------
    output      :   logging.Logger
                    The my_logger with the updated my_logger level.
    Example
    -------
    new_level = 'INFO'
    change_logging_level(new_level)
    """

    check_type(new_level, str)
    check_type(logger, [type(None), logging.Logger, logging.FileHandler, logging.StreamHandler])
    validate_logging_level(new_level)

    if logger is None:
        this_logger = logging.root
    else:
        this_logger = logger

    new_level_string = 'logging.' + new_level
    this_logger.setLevel(eval(new_level_string))

    return this_logger


def set_logging_handler(logger, *, handler_dict, formatter=None):
    """
    Setups a handler for a my_logger by passing a actions_dict with two keys: ['abspath', 'level']. This
    is useful when the events need to be logged into a file with a level that might differ from that
    of the main my_logger.
    Parameters
    ----------
    logger          :   logging.Logger
                        The my_logger whose handler need be set up
    handler_dict    :   dict

    Returns
    -------
    my_logger  :
    """
    check_type(logger, logging.Logger)
    check_type(handler_dict, dict)

    # Make sure handler actions_dict contains the correct keys (['abspath', 'level'])
    if any(key not in handler_dict for key in ['abspath', 'level']):
        error_message = f"file_handler is expected to have " \
                        f"two keys: 'abspath' and 'level'"
        raise ValueError(error_message)

    # Make sure that the level indicated in the actions_dict is valid
    handler_logging_level = handler_dict.get('level')
    validate_logging_level(handler_logging_level)

    # Remove all handlers from my_logger if any
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Create a my_logger.handler object that points to 'abspath'
    fh = logging.FileHandler(handler_dict.get('abspath'))

    # Change its level as specified in the handler
    change_logging_level(handler_dict.get('level'), logger=fh)

    # Set formatting of log message
    fh.setFormatter(formatter) if formatter else None

    # Add Handler to my_logger
    logger.addHandler(fh)

    return logger


def allowed_logging_levels():

    allowed = ['DEBUG',
               'INFO',
               'WARNING',
               'ERROR',
               'CRITICAL']

    return allowed


def validate_logging_level(level):

    check_type(level, str)

    allowed_level_strings = allowed_logging_levels()

    if level not in allowed_level_strings:
        raise ValueError(f"level {level} not allowed. Allowed values are {allowed_level_strings}")

    return True
