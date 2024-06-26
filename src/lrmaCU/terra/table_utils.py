import copy
import datetime
import re
from enum import Enum

import numpy as np
import pandas as pd
from dateutil import parser
from firecloud.errors import FireCloudServerError
from google.cloud import storage

from lrmaCU.gcs_utils import GcsPath
from lrmaCU.utils import *

logger = logging.getLogger(__name__)

ROOT_LEVEL_TABLE = 'The table in a workspace that represents the smallest analyzable unit of data, e.g. a flowcell.'


########################################################################################################################
def fetch_existing_root_table(ns: str, ws: str, etype: str,
                              list_type_attributes: List[str] = None,
                              max_attempts: int = 2) \
        -> pd.DataFrame:
    """
    Getting the ROOT_LEVEL_TABLE.

    :param ns:
    :param ws:
    :param etype: e.g. 'flowcell'
    :param list_type_attributes: a list of attribute names,
                                 where each of them is assumed to be an attribute that holds a list as value;
                                 Missing values will be parsed into an empty list.
                                 Note that this cannot be list of references to other entities in the workspace,
                                 i.e. members for a set type, for that, use `fetch_and_format_existing_set_table`
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return: DataFrame where the first column is named as what you see as the table name on Terra
    """
    response = retry_fiss_api_call('get_entities', max_attempts,
                                   ns, ws, etype=etype)
    if not response.ok:
        logger.error(f"Table {etype} doesn't seem to exist in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    if 0 == len(response.json()):
        raise KeyError(f"The entity type you requested ({etype}) doesn't exist in {ns}/{ws}")

    # get columns 2+
    attributes = pd.DataFrame([e.get('attributes') for e in response.json()]).sort_index(axis=1)
    if list_type_attributes is not None:
        attributes = _format_list_type_column(attributes, list_type_attributes)

    # get the ID column right
    entities = [e.get('name') for e in response.json()]
    entity_type = [e.get('entityType') for e in response.json()][0]
    attributes.insert(0, column=entity_type, value=entities)
    return attributes


def upload_root_table(ns: str, ws: str, table: pd.DataFrame,
                      max_attempts: int = 2) -> None:
    """
    Upload a ROOT_LEVEL_TABLE to Terra ns/ws. Most useful when initializing a workspace.

    Special note on the format of the input `table:
    The pandas DataFrame is assumed to be
        1) either correctly formatted,
            i.e. the 1st column's name conforms to the naming convention of 'entity:{blah}_id'.
        2) or 1st column's name will be the name of the table,
            i.e. if the 1st column's 'blah', the table will appear as 'blah' on Terra
    """
    n = table.columns.tolist()[0]
    if not (n.startswith('entity:') and n.endswith('_id')):
        logger.warning(f"Input table's 1st column name doesn't follow Terra's requirements: {n}.\n"
                       f"We'll assume the 1st column ({table.columns[0]}) is how you want to name the table.\n"
                       f"This means, if you intend to append to an existing table, whose name isn't the same as the 1st"
                       f"column of this input table, it WILL NOT be appended. "
                       f"It may be appened to the wrong table by accident, or create another table.", )
    response = retry_fiss_api_call('upload_entities', max_attempts,
                                   namespace=ns,
                                   workspace=ws,
                                   entity_data=table.to_csv(sep='\t', index=False),
                                   model='flexible')
    if not response.ok:
        logger.error(f"Failed to upload root level table {n} to workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)


def correct_simple_map_attribute_format(ns: str, ws: str, table_name: str,
                                        map_attributes: List[str],
                                        max_attempts: int = 2) -> None:
    """
    To correct the formatting issue on Terra tables' Map attributes:
    sometimes they are formatted as string as opposed to JSON.
    """
    def _correct_format(row: pd.Series):
        # etype = re.sub('_id$', '', re.sub('^entity:', '', row.index[0]))
        ename = row[0]
        for attr in map_attributes:
            dd = row[attr]
            if not isinstance(dd, dict):
                raise ValueError(str(row))
            reformated_dd = dd.copy()
            for k, v in reformated_dd.items():
                if isinstance(v, float) and str(v) == 'nan':
                    reformated_dd[k] = 'na'
            new_or_overwrite_attribute(ns, ws,
                                       etype=table_name, ename=ename,
                                       attribute_name=attr, attribute_value=reformated_dd)
    terra_table = fetch_existing_root_table(ns, ws, table_name, max_attempts=max_attempts)
    terra_table.apply(_correct_format, axis=1)


########################################################################################################################
class MembersOperationType(Enum):
    RESET = 1  # remove old members and fill with new members
    MERGE = 2  # just add in new members that weren't there


def upload_set_table(ns: str, ws: str, table: pd.DataFrame,
                     current_set_type_name: str, desired_set_type_name: str,
                     current_membership_col_name: str, desired_membership_col_name: str,
                     operation: MembersOperationType,
                     max_attempts: int = 2) -> None:
    """
    Upload set level table to Terra ns/ws.

    Table is will be formatted automatically, if the column names are given correctly.
    :param ns:
    :param ws:
    :param table:
    :param current_set_type_name:
    :param desired_set_type_name:
    :param current_membership_col_name:
    :param desired_membership_col_name:
    :param operation: whether old members list (if any) needs to be reset, or just add new ones.
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return:
    """

    formatted_set_table, members_for_each_set = \
        format_set_table_ready_for_upload(copy.deepcopy(table), current_set_type_name, desired_set_type_name,
                                          current_membership_col_name)

    # upload set table, except membership column
    response = retry_fiss_api_call('upload_entities', max_attempts,
                                   namespace=ns, workspace=ws,
                                   entity_data=formatted_set_table.to_csv(sep='\t', index=False),
                                   model='flexible')
    if not response.ok:
        logger.error(f"Failed to upload set level table {desired_set_type_name} to workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)
    logger.info("uploaded set level table, next fill-in members...")

    # update each set with its members
    member_entity_type = _resolve_member_type(desired_membership_col_name)
    for i in range(len(members_for_each_set)):
        set_uuid = formatted_set_table.iloc[i, 0]
        members = members_for_each_set[i]
        try:
            fill_in_entity_members(ns, ws, etype=desired_set_type_name, ename=set_uuid,
                                   member_entity_type=member_entity_type, members=members, operation=operation,
                                   max_attempts=max_attempts)
        except FireCloudServerError:
            logger.error(f"Failed to upload membership information for {set_uuid}")
            raise


def format_set_table_ready_for_upload(set_table: pd.DataFrame,
                                      current_set_type_name: str, desired_set_type_name: str,
                                      membership_col_name: str) -> (pd.DataFrame, List[List[str]]):
    """
    Given an un-formatted set table, format it in a way that's ready to be accepted by Terra API.

    :param set_table: to-be-formatted table
    :param current_set_type_name:
    :param desired_set_type_name: desired name of the table, i.e. its 0th column will be f"entity:{desired_set_type_name}_id"
    :param membership_col_name: name of the column holding the members
    :return: a formatted table that is ready to be uploaded to Terra via API calls sans the membership column,
             which is returned as the 2nd value in the returned tuple
    """
    col = set_table.pop(current_set_type_name)
    formatted_set_table = set_table.copy(deep=True)
    formatted_set_table.insert(0, f"entity:{desired_set_type_name}_id", col)

    members = formatted_set_table[membership_col_name].tolist()

    formatted_set_table.drop([membership_col_name], axis=1, inplace=True)

    return formatted_set_table, members


def fill_in_entity_members(ns: str, ws: str,
                           etype: str, ename: str,
                           member_entity_type: str, members: List[str],
                           operation: MembersOperationType,
                           member_col_name_override: str = None,
                           max_attempts: int = 2) -> None:
    """
    For a given entity set identified by etype and ename, fill-in it's members

    Critical assumption: the set itself and the member entities already exist on Terra.
    :param ns: namespace
    :param ws: workspace
    :param etype:
    :param ename:
    :param member_entity_type:
    :param members: list of member uuids
    :param operation: whether to override or append to existing membership list
    :param member_col_name_override: on Terra, the name of the column holding the references to members
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return:
    """

    response = retry_fiss_api_call('get_entity', max_attempts,
                                   ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Error occurred while trying to fill in entity members to {etype} {ename}. Make sure it exists.")
        raise FireCloudServerError(response.status_code, response.text)

    member_col_name = f'{member_entity_type}s' if member_col_name_override is None else member_col_name_override

    existing_attributes = response.json().get('attributes')

    operations = list()
    if member_col_name not in existing_attributes:
        operations.append({
            "op": "CreateAttributeEntityReferenceList",
            "attributeListName": member_col_name
        })
        members_to_upload = members
    else:
        old_members = [e['entityName'] for e in existing_attributes[member_col_name]['items']]
        if operation == MembersOperationType.MERGE:
            members_to_upload = list(set(members) - set(old_members))
        else:
            for member_id in old_members:
                operations.append({
                    "op": "RemoveListMember",
                    "attributeListName": member_col_name,
                    "removeMember": {"entityType": f"{member_entity_type}",
                                     "entityName": f"{member_id}"}
                })
            members_to_upload = members

    for member_id in members_to_upload:
        operations.append({
            "op": "AddListMember",
            "attributeListName": member_col_name,
            "newMember": {"entityType": f"{member_entity_type}",
                          "entityName": f"{member_id}"}
        })
    logger.debug(operations)

    response = retry_fiss_api_call('update_entity', max_attempts,
                                   ns, ws,
                                   etype=etype,
                                   ename=ename,
                                   updates=operations)
    if not response.ok:
        logger.error(f"Error occurred while trying to fill in entity members to {etype} {ename}."
                     f"Tentative {member_entity_type} members: {members}")
        raise FireCloudServerError(response.status_code, response.text)


def add_one_set(ns: str, ws: str,
                etype: str, ename: str,
                member_type: str, members: List[str],
                attributes: dict or None,
                max_attempts: int = 2) -> None:
    """
    To support adding a new set.
    :param ns: namespace
    :param ws: workspace
    :param etype: type of the set, must exist
    :param ename: entity name of the set, must NOT exist
    :param member_type: members' type, must exist
    :param members: list of members, must exist
    :param attributes: attributes to add for this set
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return:
    """

    one_row_bare_bone = pd.DataFrame.from_dict({etype: ename, member_type: members}, orient='index').transpose()
    upload_set_table(ns, ws, one_row_bare_bone, etype, etype, member_type, member_type, MembersOperationType.RESET,
                     max_attempts)

    if attributes:
        for k, v in attributes.items():
            new_or_overwrite_attribute(ns, ws, etype, ename, attribute_name=k, attribute_value=v,
                                       max_attempts=max_attempts)


def fetch_and_format_existing_set_table(ns: str, ws: str, etype: str, member_column_name: str,
                                        list_type_attributes: List[str] = None,
                                        max_attempts: int = 2) -> pd.DataFrame:
    """
    Intended to be used when some columns of an existing set level table are to be edited.
    See add_or_drop_columns_to_existing_set_table() for example
    :param ns:
    :param ws:
    :param etype:
    :param member_column_name:
    :param list_type_attributes: a list of attribute names,
                                 where each of them is assumed to be an attribute that holds a list as value;
                                 Missing values will be parsed into an empty list.
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return:
    """
    # fetch and keep all attributes in original table
    response = retry_fiss_api_call('get_entities', max_attempts,
                                   ns, ws, etype=etype)
    if not response.ok:
        logger.error(f"Error occurred while fetching table {etype} from {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    entities = pd.Series([e.get('name') for e in response.json()], name=etype)
    attributes = pd.DataFrame([e.get('attributes') for e in response.json()]).sort_index(axis=1)

    # re-format the columns where values are simple lists
    if list_type_attributes is not None:
        attributes = _format_list_type_column(attributes, list_type_attributes)

    # re-format the membership column, otherwise uploading will cause problems
    x = attributes[member_column_name].apply(lambda d: [e.get('entityName') for e in d.get('items')])
    attributes[member_column_name] = x

    return pd.concat([entities, attributes], axis=1)


def _add_or_drop_columns_to_existing_set_table(ns: str, ws: str, etype: str, member_column_name: str,
                                               max_attempts: int = 2) -> None:
    """
    An example (so please don't run) scenario to use fetch_and_format_existing_set_table.
    :param ns:
    :param ws:
    :param etype:
    :param member_column_name:
    :return:
    """

    formatted_original_table = fetch_and_format_existing_set_table(ns, ws, etype, member_column_name,
                                                                   max_attempts=max_attempts)

    # an example: do something here, add, drop, batch-modify existing columns
    identities = formatted_original_table.iloc[:, 1].apply(lambda s: s)
    identities.name = 'identical_copy_of_col_2'
    updated_table = pd.concat([formatted_original_table, identities], axis=1)

    # and upload
    upload_set_table(ns, ws, updated_table,
                     current_set_type_name=etype, desired_set_type_name=etype,
                     current_membership_col_name=member_column_name, desired_membership_col_name=member_column_name,
                     operation=MembersOperationType.RESET,
                     max_attempts=max_attempts)


def _resolve_member_type(membership_col_name: str) -> str:
    """
    An attempt to, given plural form like 'samples', 'families', return its singular form for member type.
    :param membership_col_name:
    :return: 'samples' -> 'sample', 'families' -> 'family'
    """
    # best I could do here; if we see more exotic names, consider using a lib
    return re.sub("ies$", "y", re.sub("s$", "", membership_col_name))


########################################################################################################################
def transfer_set_table(namespace: str,
                       original_workspace: str, new_workspace: str,
                       original_set_type: str, membership_col_name: str,
                       desired_new_set_type_name: str,
                       columns_to_keep: List[str] = None,
                       columns_that_are_lists: List[str] = None,
                       max_attempts: int = 2) -> None:
    """
    Transfer set-level table from one workspace to another workspace.

    Assuming
      * the two workspaces live under the same namespace
      * the membership column are exactly the same
      * all the member entities are already in the target workspace

    todo: relax the assumptions above

    :param namespace:
    :param original_workspace:
    :param new_workspace:
    :param original_set_type:
    :param membership_col_name:
    :param desired_new_set_type_name:
    :param columns_to_keep:
    :param columns_that_are_lists: name of the columns where entries are lists rather than simple types
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return:
    """

    if columns_to_keep is not None and len(columns_to_keep) == 0:
        logger.warning("Only membership column will be transferred")

    original_table = fetch_and_format_existing_set_table(namespace, original_workspace,
                                                         original_set_type, membership_col_name,
                                                         list_type_attributes=columns_that_are_lists,
                                                         max_attempts=max_attempts)

    columns_to_transfer = [original_set_type, membership_col_name]
    if columns_to_keep is not None and len(columns_to_keep) > 0:
        columns_to_transfer.extend(columns_to_keep)
    table_to_transfer = original_table[columns_to_transfer]

    upload_set_table(namespace, new_workspace, table_to_transfer,
                     original_set_type, desired_new_set_type_name,
                     membership_col_name, membership_col_name, MembersOperationType.RESET,
                     max_attempts)


########################################################################################################################
def new_or_overwrite_attribute(ns: str, ws: str, etype: str, ename: str,
                               attribute_name: str, attribute_value,
                               max_attempts: int = 2,
                               dry_run: bool = False) -> None:
    """
    Add a new, or overwrite existing value of an attribute to a given entity, with the given value.

    :param ns: namespace
    :param ws: workspace
    :param etype: entity type
    :param ename: entity uuid
    :param attribute_name:
    :param attribute_value:
    :param max_attempts: for retrying when seeing connection reset by peer error
    :param dry_run: safe measure, you may want to see the command before actually committing the action.
    """
    if attribute_value is None:
        raise ValueError("Attribute value is none")

    response = retry_fiss_api_call('get_entity', max_attempts,
                                   ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Are you sure {etype} {ename} exists in {ns}/{ws}?")
        raise FireCloudServerError(response.status_code, response.text)

    cov = {"op": "AddUpdateAttribute",
           "attributeName": attribute_name,
           "addUpdateAttribute": attribute_value}
    operations = [cov]
    if dry_run:
        print(operations)
        return

    response = retry_fiss_api_call('update_entity', max_attempts,
                                   ns, ws, etype=etype, ename=ename, updates=operations)
    if not response.ok:
        logger.error(f"Failed to update attribute {attribute_name} to {attribute_value}, for {etype} {ename}.")
        raise FireCloudServerError(response.status_code, response.text)

def update_list_attribute(ns: str, ws: str, etype: str, ename: str,
                          attribute_name: str,
                          list_entries: list,
                          reset_list: bool,
                          max_attempts: int = 2,
                          dry_run: bool = False) -> None:
    """
    Create/reset/append to a list-like attribute of an entity.

    :param ns: namespace
    :param ws: workspace
    :param etype: entity type
    :param ename: entity uuid
    :param attribute_name: if attribute does not exist, will create
    :param list_entries: entries in the list to be appended/used
    :param reset_list: if true, will reset the existing list with provided list, otherwise will append;
                       no effect if the entity doesn't already have the attribute (the attr will be created)
    :param max_attempts: for retrying when seeing connection reset by peer error
    :param dry_run: safe measure, you may want to see the command before actually committing the action.
    """
    if list_entries is None:
        raise ValueError("Attribute value is none. If you want to remove this attribute, use delete_attribute instead.")
    if 0 == len(list_entries):
        logger.warning("Provided list entry is empty.")
        if not reset_list:
            raise ValueError("Provided an empty list while not demanding reset. Most likely this is an error. Check.")

    response = retry_fiss_api_call('get_entity', max_attempts,
                                   ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Are you sure {etype} {ename} exists in {ns}/{ws}?")
        raise FireCloudServerError(response.status_code, response.text)

    try:
        ldf = fetch_existing_root_table(ns, ws, etype,
                                        list_type_attributes=[attribute_name])
    except KeyError:  # the attribute doesn't exist
        ldf = fetch_existing_root_table(ns, ws, etype)

    operations = list()
    if attribute_name in ldf.columns:
        if reset_list:
            op_remove = {"op": "RemoveAttribute",
                         "attributeName": attribute_name}
            operations.append(op_remove)
            op_create_list = {"op": "CreateAttributeValueList",
                              "attributeName": attribute_name,}
            operations.append(op_create_list)
    else:
        op_create_list = {"op": "CreateAttributeValueList",
                          "attributeName": attribute_name,}
        operations.append(op_create_list)

    for e in list_entries:
        op_append = {"op": "AddListMember",
                     "attributeListName": attribute_name,
                     "newMember": e}
        operations.append(op_append)
    if dry_run:
        print(operations)
        return

    response = retry_fiss_api_call('update_entity', max_attempts,
                                   ns, ws, etype=etype, ename=ename, updates=operations)
    if not response.ok:
        logger.error(f"Failed to update list-like attribute {attribute_name} with {list_entries}, for {etype} {ename}.")
        raise FireCloudServerError(response.status_code, response.text)


def delete_attribute(ns: str, ws: str, etype: str, ename: str,
                     attribute_name: str,
                     max_attempts: int = 2,
                     dry_run: bool = False) -> None:
    """
    Delete a requested attribute of the requested entity.

    :param ns: namespace
    :param ws: workspace
    :param etype: entity type
    :param ename: entity uuid
    :param attribute_name: name of the attribute to delete
    :param max_attempts: for retrying when seeing connection reset by peer error
    :param dry_run: safe measure, you may want to see the command before actually committing the action.
    """

    response = retry_fiss_api_call('get_entity', max_attempts,
                                   ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Are you sure {etype} {ename} exists in {ns}/{ws}?")
        raise FireCloudServerError(response.status_code, response.text)

    action = {"op": "RemoveAttribute",
              "attributeName": attribute_name}
    operations = [action]
    if dry_run:
        print(operations)
        return
    response = retry_fiss_api_call('update_entity', max_attempts,
                                   ns, ws, etype=etype, ename=ename, updates=operations)
    if not response.ok:
        logger.error(f"Failed to remove attribute {attribute_name} from {etype} {ename}.")
        raise FireCloudServerError(response.status_code, response.text)


def delete_attribute_and_remove_file(ns: str, ws: str, etype: str, ename: str,
                                     attribute_name: str,
                                     storage_client: storage.Client,
                                     error_on_nonexistent: bool = True,
                                     max_attempts: int = 2,
                                     dry_run: bool = False) -> None:
    """
    Delete a requested attribute of the requested entity. And if it's a cloud storage, delete the associate file.

    THIS IS A DANGEROUS OPERATION. DO IT VERY, VERY CAREFULLY.

    Assumes attribute is a plain string and starts with gs://.

    :param ns: namespace
    :param ws: workspace
    :param etype: entity type
    :param ename: entity uuid
    :param attribute_name: name of the attribute to delete
    :param storage_client:
    :param error_on_nonexistent: if the file pointed to by the value in the table doesn't exist, should error or not
    :param max_attempts: for retrying when seeing connection reset by peer error
    :param dry_run: safe measure, you may want to see the command before actually committing the action.
    """

    # it doesn't matter if it's a root type or not
    df = fetch_existing_root_table(ns, ws, etype, max_attempts=max_attempts)
    row = df.loc[df.iloc[:, 0] == ename, :]
    assert len(row) == 1, f"Requested entity {ename} of type {etype} either doesn't exist, or has multiple entries"

    attr = str(df.loc[df.iloc[:, 0] == ename, attribute_name])
    assert isinstance(attr, str), f"The requested attribute ({attribute_name}) is not a path-type."
    assert attr.startswith('gs://'), f"The requested attribute ({attribute_name}) is not a storage path."

    delete_attribute(ns, ws, etype, ename, attribute_name, max_attempts, dry_run)
    if not dry_run:
        GcsPath(attr).delete(storage_client, recursive=True, error_on_nonexistent=error_on_nonexistent)


def attribute_is_not_available(loaded_terra_table: pd.DataFrame,
                               attribute: str) -> pd.Series:
    """
    For checking which entities doesn't have the attribute-of-interest yet.

    Throws KeyError if the attribute-of-interest isn't in the table's columns at all.
    """
    column: pd.Series = loaded_terra_table[attribute]

    idx_1 = column.isna()

    def more_nuanced(v) -> bool:
        if isinstance(v, float):
            return np.isnan(v)
        else:
            return str(v).lower() in ('', 'nan', 'na')
    idx_2 = column.apply(more_nuanced)

    return idx_1 | idx_2

########################################################################################################################
def _format_list_type_column(table: pd.DataFrame, list_type_attributes: List[str]) -> pd.DataFrame:
    """
    For a dataframe retrieved from Terra, format the attributes where values for an entity is a list, accordingly.

    :param table: table just retrieved from Terra
    :param list_type_attributes: list of such attribute names
    :return: corrected formatted table
    """
    formatted = table.copy(deep=True)
    for attr in list_type_attributes:
        formatted[attr] = \
            table[attr].apply(lambda d: list() if isinstance(d, float) or pd.isna(d) else d['items'])
    return formatted


def _convert_to_float(e) -> float or None:
    if e:
        if pd.isna(e):
            return np.nan
        elif e.lower() in ['nan', 'none']:
            return np.nan
        else:
            try:
                return float(e)
            except TypeError:
                logger.error(f"{e} cannot be casted to a floating point number.")
                raise
    else:
        return np.nan


def _convert_to_int(e) -> int:
    f = _convert_to_float(e)
    return np.nan if np.isnan(f) else round(f)


def _convert_date_time(s, timezone):
    try:
        t = parser.isoparse(s).astimezone(tz=timezone)
        return pd.to_datetime(t)
    except (ValueError, pd.errors.OutOfBoundsDatetime):
        logger.warning(f"Error when parsing {s} as datetime with {timezone}. "
                       f"Formatting it to {pd.Timestamp.min}.")
        return pd.Timestamp.min


def format_table_to_appropriate_type(raw_table: pd.DataFrame,
                                     boolean_columns: List[str] = None,
                                     categorical_columns: List[str] = None,
                                     string_type_columns: List[str] = None,
                                     int_type_columns: List[str] = None,
                                     float_type_columns: List[str] = None,
                                     date_time_columns: List[str] = None,
                                     timezone: datetime.timezone = None) -> pd.DataFrame:
    """
    Perform type conversion for a raw DataFrame.

    :param raw_table:
    :param boolean_columns: note that missing values be converted to False
    :param categorical_columns:
    :param string_type_columns:
    :param int_type_columns:
    :param float_type_columns:
    :param date_time_columns:
    :param timezone: must be provided when date_time_columns is provided
    :return:
    """

    if date_time_columns:
        if not timezone:
            raise ValueError("Please provide timezone when formatting datetime columns")

    res = raw_table.copy(deep=True)
    for n in boolean_columns:
        if res[n].dtype == np.bool_:
            continue
        res[n] = res[n].apply(lambda s: pd.notna(s) and str(s).lower() == 'true').astype('bool')

    for n in categorical_columns:
        res[n] = res[n].astype('category')

    for n in string_type_columns:
        res[n] = res[n].astype('str')

    for n in float_type_columns:
        try:
            res[n] = res[n].astype(str).apply(_convert_to_float).astype('float64')
        except TypeError:
            logger.error(f"Error when casting for column {n}.")
            raise

    for n in int_type_columns:
        try:
            res[n] = res[n].astype(str).apply(_convert_to_int).astype('Int64')
        except TypeError:
            logger.error(f"Error when casting for column {n}.")
            raise

    for n in date_time_columns:
        res[n] = res[n].astype(str).apply(lambda s: _convert_date_time(s, timezone))

    return res
