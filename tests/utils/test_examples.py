from skepsi.utils.utils import *
from skepsi.utils.imports import *

# Get test data
testdata = [(datetime(2001, 12, 12), datetime(2001, 12, 11), timedelta(1)),
            (datetime(2001, 12, 11), datetime(2001, 12, 12), timedelta(-1))]


@pytest.mark.parametrize("a,b,expected", testdata, ids=lambda x: str(x))
def test_timedistance_v1(a, b, expected):
    diff = a - b
    assert diff == expected
