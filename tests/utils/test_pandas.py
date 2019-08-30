from helpers.utils import *
from helpers.imports import *

def test_validate_timezone():

    assert pd_validate_timezone('UTC')
    with pytest.raises(pytz.exceptions.UnknownTimeZoneError):
        pd_validate_timezone('America')
