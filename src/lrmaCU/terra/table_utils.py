import datetime
import re
from enum import Enum

import numpy as np
import pandas as pd
from dateutil import parser
from firecloud import api as fapi
from firecloud.errors import FireCloudServerError

from ..utils import *

logger = logging.getLogger(__name__)

ROOT_LEVEL_TABLE = 'The table in a workspace that represents the smallest analyzable unit of data, e.g. a flowcell.'


########################################################################################################################
def fetch_existing_root_table(ns: str, ws: str, etype: str,
                              list_type_attributes: List[str] = None, list_attribute_compact_str_delimiter: str = ',') \
        -> pd.DataFrame:
    """
    Getting the ROOT_LEVEL_TABLE.

    :param ns:
    :param ws:
    :param etype: e.g. 'flowcell`
    :param list_type_attributes: a list of attribute names,
                                 where each of them is assumed to be an attribute that holds a list as value;
                                 each list of will be compacted into a string, using the provided delimiter.
                                 Missing values will be parsed into 'nan'.
                                 Note that this cannot be list of references to other entities in the workspace,
                                 i.e. members for a set type, for that, use `fetch_and_format_existing_set_table`
    :param list_attribute_compact_str_delimiter: for an attribute that is a list, the list is compacted into a string,
                                 delimited by the provided delimiter
    :return: DataFrame where the first column is named as what you see as the table name on Terra
    """
    response = fapi.get_entities(ns, ws, etype=etype)
    if not response.ok:
        logger.error(f"Table {etype} doesn't seem to exist in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    if 0 == len(response.json()):
        raise KeyError(f"The entity type you requested ({etype}) doesn't exist in {ns}/{ws}")

    attributes = pd.DataFrame([e.get('attributes') for e in response.json()]).sort_index(axis=1)
    delim = list_attribute_compact_str_delimiter
    if list_type_attributes is not None:
        for attr in list_type_attributes:
            attributes[attr] = \
                attributes[attr].apply(lambda x:
                                       'nan' if pd.isna(x) else delim.join([str(e) for e in x['items']]))

    entities = [e.get('name') for e in response.json()]
    entity_type = [e.get('entityType') for e in response.json()][0]
    attributes.insert(0, column=entity_type, value=entities)
    return attributes


def upload_root_table(ns: str, ws: str, table: pd.DataFrame) -> None:
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
                       f"It may be appened to the wrong table by accident, or create another table.",)
    response = fapi.upload_entities(namespace=ns,
                                    workspace=ws,
                                    entity_data=table.to_csv(sep='\t', index=False),
                                    model='flexible')
    if not response.ok:
        logger.error(f"Failed to upload root level table {n} to workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)


########################################################################################################################
class MembersOperationType(Enum):
    RESET = 1  # remove old members and fill with with new members
    MERGE = 2  # just add in new members that weren't there


def upload_set_table(ns: str, ws: str, table: pd.DataFrame,
                     current_set_type_name: str, desired_set_type_name: str,
                     current_membership_col_name: str, desired_membership_col_name: str,
                     operation: MembersOperationType) -> None:
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
    :return:
    """

    formatted_set_table, members_for_each_set = \
        format_set_table_ready_for_upload(table, current_set_type_name, desired_set_type_name,
                                          current_membership_col_name)

    # upload set table, except membership column
    response = fapi.upload_entities(namespace=ns, workspace=ws,
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
                                   member_entity_type=member_entity_type, members=members, operation=operation)
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
                           operation: MembersOperationType) -> None:
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
    :return:
    """

    operations = list()
    response = fapi.get_entity(ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Error occurred while trying to fill in entity members to {etype} {ename}. Make sure it exists.")
        raise FireCloudServerError(response.status_code, response.text)

    attributes = response.json().get('attributes')
    if f'{member_entity_type}s' not in attributes:
        operations.append({
            "op": "CreateAttributeEntityReferenceList",
            "attributeListName": f"{member_entity_type}s"
        })
        members_to_upload = members
    else:
        old_members = [e['entityName'] for e in attributes[f'{member_entity_type}s']['items']]
        if operation == MembersOperationType.MERGE:
            members_to_upload = list(set(members) - set(old_members))
        else:
            for member_id in old_members:
                operations.append({
                    "op": "RemoveListMember",
                    "attributeListName": f"{member_entity_type}s",
                    "removeMember": {"entityType":f"{member_entity_type}",
                                     "entityName":f"{member_id}"}
                })
            members_to_upload = members

    for member_id in members_to_upload:
        operations.append({
            "op": "AddListMember",
            "attributeListName": f"{member_entity_type}s",
            "newMember": {"entityType":f"{member_entity_type}",
                          "entityName":f"{member_id}"}
        })
    logger.debug(operations)

    response = fapi.update_entity(ns, ws,
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
                attributes: dict or None) -> None:
    """
    To support adding a new set.
    :param ns: namespace
    :param ws: workspace
    :param etype: type of the set, must exist
    :param ename: entity name of the set, must NOT exist
    :param member_type: members' type, must exist
    :param members: list of members, must exist
    :param attributes: attributes to add for this set
    :return:
    """

    one_row_bare_bone = pd.DataFrame.from_dict({etype: ename, member_type: members}, orient='index').transpose()
    upload_set_table(ns, ws, one_row_bare_bone, etype, etype, member_type, member_type, MembersOperationType.RESET)

    if attributes:
        for k, v in attributes.items():
            new_or_overwrite_attribute(ns, ws, etype, ename, attribute_name=k, attribute_value=v)


def fetch_and_format_existing_set_table(ns: str, ws: str, etype: str, member_column_name: str) -> pd.DataFrame:
    """
    Intended to be used when some columns of an existing set level table are to be edited.
    See add_or_drop_columns_to_existing_set_table() for example
    :param ns:
    :param ws:
    :param etype:
    :param member_column_name:
    :return:
    """
    # fetch and keep all attributes in original table
    response = fapi.get_entities(ns, ws, etype=etype)

    entities = pd.Series([e.get('name') for e in response.json()], name=etype)
    attributes = pd.DataFrame([e.get('attributes') for e in response.json()]).sort_index(axis=1)

    # re-format the membership column, otherwise uploading will cause problems
    x = attributes[member_column_name].apply(lambda d: [e.get('entityName') for e in d.get('items')])
    attributes[member_column_name] = x

    return pd.concat([entities, attributes], axis=1)


def _add_or_drop_columns_to_existing_set_table(ns: str, ws: str, etype: str, member_column_name: str) -> None:
    """
    An example (so please don't run) scenario to use fetch_and_format_existing_set_table.
    :param ns:
    :param ws:
    :param etype:
    :param member_column_name:
    :return:
    """

    formatted_original_table = fetch_and_format_existing_set_table(ns, ws, etype, member_column_name)

    # an example: do something here, add, drop, batch-modify existing columns
    identities = formatted_original_table.iloc[:, 1].apply(lambda s: s)
    identities.name = 'identical_copy_of_col_2'
    updated_table = pd.concat([formatted_original_table, identities], axis=1)

    # and upload
    upload_set_table(ns, ws, updated_table,
                     current_set_type_name=etype, desired_set_type_name=etype,
                     current_membership_col_name=member_column_name, desired_membership_col_name=member_column_name,
                     operation=MembersOperationType.RESET)


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
                       columns_that_are_lists: List[str] = None) -> None:
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
    :return:
    """

    if columns_to_keep is not None and len(columns_to_keep) == 0:
        logger.warning("Only membership column will be transferred")

    original_table = fetch_and_format_existing_set_table(namespace, original_workspace,
                                                         original_set_type, membership_col_name)

    columns_to_transfer = [original_set_type, membership_col_name]
    if columns_to_keep is not None and len(columns_to_keep) > 0:
        columns_to_transfer.extend(columns_to_keep)
    table_to_transfer = original_table[columns_to_transfer]

    upload_set_table(namespace, new_workspace, table_to_transfer,
                     original_set_type, desired_new_set_type_name,
                     membership_col_name, membership_col_name, MembersOperationType.RESET)


########################################################################################################################
def new_or_overwrite_attribute(ns: str, ws: str, etype: str, ename: str,
                               attribute_name: str, attribute_value,
                               dry_run: bool = False) -> None:
    """
    Add a new, or overwrite existing value of an attribute to a given entity, with the given value.

    :param ns: namespace
    :param ws: workspace
    :param etype: entity type
    :param ename: entity uuid
    :param attribute_name:
    :param attribute_value:
    :param dry_run: safe measure, you may want to see the command before actually committing the action.
    """
    if attribute_value is None:
        raise ValueError("Attribute value is none")

    response = fapi.get_entity(ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Are you sure {etype} {ename} exists in {ns}/{ws}?")
        raise FireCloudServerError(response.status_code, response.text)

    cov = {"op":                 "AddUpdateAttribute",
           "attributeName":      attribute_name,
           "addUpdateAttribute": attribute_value}
    operations = [cov]
    if dry_run:
        print(operations)
        return

    response = fapi.update_entity(ns, ws,
                                  etype=etype,
                                  ename=ename,
                                  updates=operations)
    if not response.ok:
        logger.error(f"Failed to update attribute {attribute_name} to {attribute_value}, for {etype} {ename}.")
        raise FireCloudServerError(response.status_code, response.text)


def delete_attribute(ns: str, ws: str, etype: str, ename: str,
                     attribute_name: str,
                     dry_run: bool = False) -> None:
    """
    Delete a requested attribute of the requested entity.

    :param ns: namespace
    :param ws: workspace
    :param etype: entity type
    :param ename: entity uuid
    :param attribute_name: name of the attribute to delete
    :param dry_run: safe measure, you may want to see the command before actually committing the action.
    """

    response = fapi.get_entity(ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Are you sure {etype} {ename} exists in {ns}/{ws}?")
        raise FireCloudServerError(response.status_code, response.text)

    action = {"op":                 "RemoveAttribute",
              "attributeName":      attribute_name}
    operations = [action]
    if dry_run:
        print(operations)
        return

    response = fapi.update_entity(ns, ws,
                                  etype=etype,
                                  ename=ename,
                                  updates=operations)
    if not response.ok:
        logger.error(f"Failed to remove attribute {attribute_name} from {etype} {ename}.")
        raise FireCloudServerError(response.status_code, response.text)


def update_one_list_attribute(ns: str, ws: str,
                              etype: str, ename: str,
                              attribute_name: str,
                              attribute_values: List[str],
                              operation: MembersOperationType) -> None:
    """
    To create an attribute, which must be a list of reference to something else, of the requested entity.

    Example of reference:
        1) reference to member entities
        2) reference to member entities' attribute
    Whatever the list elements refer to, the targets must exist.

    :param ns: namespace
    :param ws: workspace
    :param etype: entity type
    :param ename: entity uuid
    :param attribute_name: name the the attribute
    :param attribute_values: a list of target to reference to
    :param operation:
    :return:
    """
    operations = list()
    response = fapi.get_entity(ns, ws, etype, ename)
    if not response.ok:
        logger.error(f"Entity {etype} {ename} doesn't seem to exist in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    attributes = response.json().get('attributes')
    if attribute_name not in attributes:  # attribute need to be created
        operations.append({
            "op": "CreateAttributeValueList",
            "attributeName": attribute_name
        })
        values_to_upload = attribute_values
    else:
        existing_values = [v for v in attributes[attribute_name]['items']]
        logger.debug(existing_values)
        if operation == MembersOperationType.MERGE:
            values_to_upload = list(set(attribute_values) - set(existing_values))
        else:
            for val in existing_values:
                operations.append({
                    "op": "RemoveListMember",
                    "attributeListName": attribute_name,
                    "removeMember": val
                })
            values_to_upload = attribute_values

    for val in values_to_upload:
        operations.append({
            "op": "AddListMember",
            "attributeListName": attribute_name,
            "newMember": val
        })
        logger.debug(operations)

    response = fapi.update_entity(ns, ws,
                                  etype=etype,
                                  ename=ename,
                                  updates=operations)
    if not response.ok:
        logger.error(f"Failed to update a list of references for {etype} {ename}:\n"
                     f" attribute {attribute_name},\n"
                     f" attribute values {attribute_values}")
        raise FireCloudServerError(response.status_code, response.text)


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
        formatted[attr] = table[attr].apply(lambda d: d['items'])
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
        res[n] = res[n].apply(lambda s: pd.notna(s) and s.lower() == 'true').astype('bool')

    for n in categorical_columns:
        res[n] = res[n].astype('category')

    for n in string_type_columns:
        res[n] = res[n].astype('str')

    for n in float_type_columns:
        try:
            res[n] = res[n].apply(_convert_to_float).astype('float64')
        except TypeError:
            logger.error(f"Error when casting for column {n}.")
            raise

    for n in int_type_columns:
        try:
            res[n] = res[n].apply(_convert_to_int).astype('Int64')
        except TypeError:
            logger.error(f"Error when casting for column {n}.")
            raise

    for n in date_time_columns:
        res[n] = res[n].apply(lambda s: _convert_date_time(s, timezone))

    return res
