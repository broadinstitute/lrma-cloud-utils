from lrmaCU.terra.submission.submission_utils import verify_before_submit
from lrmaCU.terra.table_utils import *

"""
This module provides some higher-level, highly specialized utilities.
They are driven by targeted use cases, so please verify carefully before using any of them.
"""

logger = logging.getLogger(__name__)


def get_entities_with_QC_results(input_table: pd.DataFrame,
                                 output_table_uuid_colname: str,
                                 qc_result_columns: List[str],
                                 other_columns_to_carryover: List[str]) -> pd.DataFrame or None:
    """
    Get entities in a (parsed) Terra table that have QC results ready.
    Assumes the first column in the input_table holds the entity ids.
    :param input_table:
    :param output_table_uuid_colname:
    :param qc_result_columns:
    :param other_columns_to_carryover:
    :return: the formatted DataFrame, or None if no entity is ready
    """
    idx = (~input_table.iloc[:,0].isna())  # hacky way to initialize to all True
    for attr in qc_result_columns:
        if attr not in input_table.columns:
            return None  # a QC column isn't available at all, nothing is ready
        else:
            idx &= (~attribute_is_not_available(input_table, attr))

    etype = input_table.columns[0]
    output_col = [etype]
    output_col.extend(qc_result_columns)
    output_col.extend(other_columns_to_carryover)

    return input_table.loc[idx, output_col]\
            .reset_index(drop=True)\
            .rename({etype:output_table_uuid_colname}, axis=1)


def get_entities_ready_for_QC(input_table: pd.DataFrame,
                              upstream_indicator_columns: List[str],
                              downstream_indicator_columns: List[str]) -> pd.DataFrame or None:
    """
    Get entities that are ready for QC, i.e. those without QC results yet.
    Assumes the first column in the input_table holds the entity ids.
    Entities ready for QC workflows must meet the following criteria:
        upstream_indicator_columns are available, and
        downstream_indicator_columns are unavailable,
    :param input_table:
    :param upstream_indicator_columns:
    :param downstream_indicator_columns:
    :return: the subset table holding rows for entities ready for QC, or None if nothing is ready
    """
    # upstream are ready
    idx = (~input_table.iloc[:,0].isna())  # hacky way to initialize to all True
    for attr in upstream_indicator_columns:
        if attr not in input_table.columns:  # a column isn't available, nothing is ready
            idx = input_table.iloc[:,0].isna()
            break
        idx &= (~attribute_is_not_available(input_table, attr))
    if all(~idx):
        return None

    # downstream aren't ready
    idy = input_table.iloc[:,0].isna()  # hacky way to initialize to all False
    for attr in downstream_indicator_columns:
        if attr not in input_table.columns:  # a column isn't available, every entity is ready
            idy = (~input_table.iloc[:,0].isna())
            break
        idy |= attribute_is_not_available(input_table, attr)
    if all(~idy):
        return None

    return input_table.loc[(idx & idy), ].reset_index(drop=True)


def launch_QC_workflows(ns:str, ws:str,
                        input_table: pd.DataFrame,
                        upstream_indicator_columns: List[str],
                        downstream_indicator_columns: List[str],
                        qc_workflows: List[str],
                        **wdl_launch_kwargs) -> dict or None:
    """
    For a (parsed) Terra table, launch QC workflows on entities that
    are not yet QCed, but are ready to be QCed.
    Entities ready for QC workflows must meet the following criteria:
        upstream_indicator_columns are available, and
        downstream_indicator_columns are unavailable.
    :param wdl_launch_kwargs: check verify_before_submit for which kwargs are recognized.
    """

    qc_ready_table = get_entities_ready_for_QC(input_table,
                                               upstream_indicator_columns,
                                               downstream_indicator_columns)

    if qc_ready_table is None or 0 == len(qc_ready_table):
        logger.warning("Nothing to QC in this batch.")
        return

    logger.info(f"{len(qc_ready_table)} entities to QC in this batch.")
    etype = input_table.columns[0]
    failures = dict()
    for wdl_name in qc_workflows:
        try:
            f = \
            verify_before_submit(ns, ws,
                                 method_name=wdl_name,
                                 etype=etype,
                                 enames=qc_ready_table.iloc[:,0].tolist(),
                                 batch_type_name=f"zzBatch_{wdl_name.replace('.','_')}",
                                 expression=f'this.{etype}s',
                                 days_back=7, count=3,
                                 **wdl_launch_kwargs)
            if f is not None and 0 != len(f):
                failures.update(f)
        except Exception as e:
            logger.error(f"Failed to launch workflow {wdl_name} for {etype}")
            logger.error("I'll continue to the following, but humans, please check.")
            logger.error(repr(e))
            failures.update({wdl_name: f"Exception encountered for {etype}. Please check the following:\n" + repr(e)})
    return failures
