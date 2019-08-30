import pytest
from skepsi.emails.find_outliers import *


@pytest.fixture(scope="session")
def sample_evaluation_threshold_dfs(tmpdir_factory):
    """
    Generate a dataframe with correct structure to be the
    output of an evaluation step, as well as valid CDF thresholds.

    Returns
    -------

    """

    # Get default thresholds
    thresholds = default_cdf_thresholds()

    # Build Multi-Index df with a "eval_cdf" column containing other columns
    level_0 = list(thresholds.index)
    level_1 = ['eval_cdf'] * len(level_0)

    # Define the names of the columns of the dataframe
    column_arrays = [np.array(level_1), np.array(level_0)]

    # Get some test data (2D matrix)
    test_data = np.reshape(np.linspace(0, 1, 3 * len(level_0)), [3, len(level_0)])

    # Build multi-index DataFrame with column 'eval_cdf'
    evaluation_df = pd.DataFrame(data=test_data, columns=column_arrays)

    return evaluation_df, thresholds


@pytest.fixture(scope="session")
def sample_email_aggregation(tmpdir_factory):

    sample_df = pd.DataFrame()
    return sample_df


@pytest.fixture(scope="session")
def sample_probabilities_df(tmpdir_factory):

    sample_df = pd.DataFrame()
    return sample_df


def test_validate_email_aggregation(sample_email_aggregation):

    # TODO    Write a test for test_accept_email_aggregation.
    df = sample_email_aggregation

    with pytest.raises(InvalidSchemaEmailAggregation):

        validate_schema_email_aggregation(df)

    return True


def test_validate_probabilities_df(sample_probabilities_df):

    # TODO    Write a Test for test_accept_probabilities_df

    df = sample_probabilities_df

    with pytest.raises(InvalidProbabilitiesDf):
        validate_probabilities_df(df)

    return True


def test_validate_evaluation_df(sample_evaluation_threshold_dfs):

    # Valid evaluation dataframe
    df, thresholds = sample_evaluation_threshold_dfs
    validate_evaluation_df(df)

    with pytest.raises(InvalidEvaluationDf):

        # Construct an invalid dataframe
        df = pd.DataFrame(data={'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        validate_evaluation_df(df)

    # --------- Types ---------

    with pytest.raises(TypeError):
        # Test type
        validate_evaluation_df(None)


def test_validate_thresholds_df(sample_evaluation_threshold_dfs):

    # Sample thresholds and df should be valid
    df, thresholds = sample_evaluation_threshold_dfs
    validate_thresholds_df(df, thresholds, verbose=True)

    with pytest.raises(InvalidEvaluationDf):

        # --------------- Invalid df --------------

        df2 = df.rename(columns={'eval_cdf': 'loris'})
        validate_thresholds_df(df2, thresholds, verbose=True)

    with pytest.raises(InvalidThresholdsDf):

        thresholds = thresholds.iloc[0:4, :]
        validate_thresholds_df(df, thresholds, verbose=True)

    with pytest.raises(TypeError):

        validate_thresholds_df(df, np.random.randn(1)[0], verbose=True)
        validate_thresholds_df(np.random.randn(1)[0], thresholds, verbose=True)
        validate_thresholds_df(df, thresholds, verbose=np.random.randn(1)[0])


def test_validate_df_to_fit():

    # Create a valid pandas DataFrame (numeric)
    df = pd.DataFrame(np.random.randn(12, 4))

    # Confirm is valid
    validate_df_to_fit(df, mss=10, max_cols=None, verbose=True)

    # Test for errors:
    with pytest.raises(InvalidDfToFit):

        # With minimum sample size mss=13, a df with 10 rows should be in valid
        validate_df_to_fit(df, mss=13, max_cols=None, verbose=True)

    with pytest.raises(InvalidDfToFit):
        # With a column number of 2, a df with 4 columns should be in valid
        validate_df_to_fit(df, mss=10, max_cols=2, verbose=True)

    with pytest.raises(InvalidDfToFit):
        # A MultiIndex df should be invalid
        df_multindex = df_group_columns(df, 'additional_level')
        validate_df_to_fit(df_multindex, mss=10, max_cols=None, verbose=True)

    # --------- Types ---------

    with pytest.raises(TypeError):
        validate_df_to_fit(df, mss=10.1, max_cols=None, verbose=True)

    with pytest.raises(TypeError):
        validate_df_to_fit(None, mss=10, max_cols=None, verbose=True)

    with pytest.raises(TypeError):
        validate_df_to_fit(df, mss=10, max_cols=pd.DataFrame, verbose=True)

    with pytest.raises(TypeError):
        validate_df_to_fit(df, mss=10, max_cols=None, verbose='test')
