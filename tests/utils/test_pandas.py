from skepsi.utils.utils import *
from skepsi.utils.imports import *

def test_validate_timezone():

    assert pd_validate_timezone('UTC')
    with pytest.raises(pytz.exceptions.UnknownTimeZoneError):
        pd_validate_timezone('America')
