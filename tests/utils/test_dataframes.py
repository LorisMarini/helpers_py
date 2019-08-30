from helpers.etl import *


def etl_object_1(*, bid, filename):
    """
    Wrapper around ETLObject() to quickly create ELTObjects for testing
    Parameters
    ----------
    bid         :   str
                    one of ["gs://", "s3://"] or other supported bucket
    filename    :   str
                    `filename.ext`
    """
    parts = os.path.splitext(filename)

    obj = ETLObject(bucket_id=bid,
                    insight="test",
                    stage="miscellaneous",
                    version="v1",
                    date_time="2000-01-01",
                    name=parts[0],
                    ext=f"{parts[1]}")
    return obj


def test_df_to_text(dataframe_1, tmpdir):

    # Local
    df_to_text(df=dataframe_1, force_ext=False, orient='index',
               saveas=tmpdir.join("dataframe-1.json").strpath)

    df_to_text(df=dataframe_1, force_ext=False, orient='index',
               saveas=tmpdir.join("dataframe-1.csv").strpath)

    df_to_text(df=dataframe_1, force_ext=False, orient='index',
               saveas=tmpdir.join("dataframe-1.agg").strpath)

    # Remote (all buckets)
    bids = available_bucket_ids()
    for bid in bids:
        obj = etl_object_1(bid=bid, filename="dataframe-1.json")
        df_to_text(df=dataframe_1, saveas=obj.prod_path, force_ext=False, orient='index')


def test_df_to_parquet(dataframe_2, tmpdir):

    # Local
    lp = tmpdir.join("dataframe-2.parquet").strpath

    # TO parquet
    df_to_parquet(df=dataframe_2, path=lp, file_scheme='simple', compression="snappy", get_size=True)
    df_to_parquet(df=dataframe_2, path=None, file_scheme='simple', compression="snappy", get_size=False)

    # FROM parquet
    df_from_parquet(path=lp, columns=None, verbose=False, get_size=True, log=True)

    # Remote (all buckets)
    bids = available_bucket_ids()

    for bid in bids:
        # Get the ETLObject id for this test
        obj = etl_object_1(bid=bid, filename="dataframe-2.parquet")

        # TO bucket
        df_to_parquet(df=dataframe_2, path=obj.prod_path)
        # FROM bucket
        _ = df_from_parquet(path=obj.prod_path)


def test_df_column_check_type(dataframe_1):

    df_column_check_type(dataframe_1, 'integers', int)
    df_column_check_type(dataframe_1, 'strings', str)
    df_column_check_type(dataframe_1, 'floats', float)
    df_column_check_type(dataframe_1, 'dates', pd.datetime)

    with pytest.raises(TypeError):
        df_column_check_type(dataframe_1, 'integers', float)
        df_column_check_type(dataframe_1, 'strings', float)
        df_column_check_type(3, 'strings', float)


# df_to_google_sheet(*, df, name, sfn, page)

# df_from_google_sheet(*, name, sfn, page=0)

# df_types_table(df)

# df_fill_missing_values(df, filler=-1)

# df_remove_nan(df, how='any', axis=0, inplace=False, verbose=False)

# df_list_unique(df, column='instanceId', verbose=False, mask=None)

# df_largest_unique(df, column='action', dropna=True, verbose=False)

# df_ecdf(*, df, column, percent=True)

# df_group_columns(df, groupname='new_group', column_names=None, copy=True)

# df_histogram(*, df, columns, bins, percent=True, apply_fun=None)

# df_timedelta_to_seconds(df, column, inplace=False, verbose=False)

# df_timestamp_to_iso(df, column, inplace=False, tz='utc', verbose=False)

# df_select(*, df, mode="all", **kwargs)

# df_columns_cat_codes(*, df, newcol, columns=["actorId", "batchCampaignId"])

# df_most_frequent(*, df, what, sort_col=None, out="sorted_df", n_max=None, ascending=False)

# df_parse_dates(*, df, columns, errors='coerce')

# df_lower_case(*, df, columns)

# df_percent_missing(df, width=400, height=250)

# get_instance_name_from_path(abspath, must_exist=True)

# df_action_columns(df, log=True)

# validate_comp_fastparquet(compression)

# df_to_smart_dashboard_api(*, df, title, saveas=None, legend=None, ylabel=None, xlabel=None,
#                               xmin=None, xmax=None, ymin=None, ymax=None)

