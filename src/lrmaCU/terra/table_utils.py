import re
from enum import Enum

import pandas as pd
from firecloud import api as fapi
from firecloud.errors import FireCloudServerError

from ..utils import *

logger = logging.getLogger(__name__)

ROOT_LEVEL_TABLE = 'The table in a workspace that represents the smallest analyzable unit of data, e.g. a flowcell.'


########################################################################################################################
def fetch_existing_root_table(ns: str, ws: str, etype: str) -> pd.DataFrame:
    """
    Getting the ROOT_LEVEL_TABLE.

    :param ns:
    :param ws:
    :param etype: e.g. 'flowcell`
    :return: DataFrame where the first column is named as what you see as the table name on Terra
    """
    response = fapi.get_entities(ns, ws, etype=etype)
    if not response.ok:
        logger.error(f"Table {etype} doesn't seem to exist in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    entities = [e.get('name') for e in response.json()]
    entity_type = [e.get('entityType') for e in response.json()][0]
    attributes = pd.DataFrame.from_dict([e.get('attributes') for e in response.json()]).sort_index(axis=1).astype('str')
    attributes.insert(0, column=entity_type, value=entities)
    return attributes.copy(deep=True)


def upload_root_table(ns: str, ws: str, table: pd.DataFrame) -> None:
    """
    Upload a ROOT_LEVEL_TABLE to Terra ns/ws. Most useful when initializing a workspace.

    The pandas DataFrame is assumed to be correctly formatted,
    i.e. the 1st column's name conforms to the naming convention of 'entity:{blah}_id'.
    """
    n = table.columns.tolist()[0]
    if not (n.startswith('entity:') and n.endswith('_id')):
        raise ValueError(f"Input table's 1st column name doesn't follow Terra's requirements: {n}")
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
            _fill_in_entity_members(ns, ws, etype=desired_set_type_name, ename=set_uuid,
                                    member_entity_type=member_entity_type, members=members, operation=operation)
        except FireCloudServerError:
            logger.error(f"Failed to upload membership information for {set_uuid}")
            raise


def format_set_table_ready_for_upload(set_table: pd.DataFrame,
                                      current_set_type_name: str, desired_set_type_name: str,
                                      membership_col_name: str) \
        -> (pd.DataFrame, List[List[str]]):
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


def _fill_in_entity_members(ns: str, ws: str,
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

    entities = pd.Series([e.get('name') for e in response.json()], name=f"entity:{etype}_id")
    attributes = pd.DataFrame.from_dict([e.get('attributes') for e in response.json()])

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
                       desired_new_set_type_name: str) -> None:
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
    :return:
    """

    response = fapi.get_entities(namespace, original_workspace, etype=original_set_type)
    if not response.ok:
        logger.error(f"Failed to retrieve set entities {original_set_type} from workspace"
                     f" {namespace}/{original_workspace}.")
        raise FireCloudServerError(response.status_code, response.text)
    logger.info(f"Original set table {original_set_type} fetched")

    # format
    uuids = [e.get('name') for e in response.json()]
    attributes_table = pd.DataFrame.from_dict([e.get('attributes') for e in response.json()])
    attributes_table.insert(0, f'entity:{original_set_type}_id', uuids)
    original_table = attributes_table.copy(deep=True)

    ready_for_upload_table, members_list = format_set_table_ready_for_upload(
        original_table, current_set_type_name=original_set_type,
        desired_set_type_name=desired_new_set_type_name, membership_col_name=membership_col_name)

    # everything except membership
    response = fapi.upload_entities(namespace, new_workspace,
                                    entity_data=ready_for_upload_table.to_csv(sep='\t', index=False),
                                    model='flexible')
    if not response.ok:
        logger.error(f"Failed to copy over set-level entities {desired_new_set_type_name},"
                     f" even before member entities are filled in.")
        raise FireCloudServerError(response.status_code, response.text)
    logger.info("uploaded set level table, next fill-in members...")

    # update each set with its members
    flat_text_membership = list(map(lambda dl: [d.get('entityName') for d in dl.get('items')], members_list))
    member_entity_type = _resolve_member_type(membership_col_name)
    for i in range(len(flat_text_membership)):
        set_uuid = ready_for_upload_table.iloc[i, 0]
        members = flat_text_membership[i]
        try:
            _fill_in_entity_members(namespace, new_workspace,
                                    etype=desired_new_set_type_name, ename=set_uuid,
                                    member_entity_type=member_entity_type, members=members, operation=MembersOperationType.RESET)
        except FireCloudServerError:
            logger.error(f"Failed to upload membership information for {set_uuid}")
            raise


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
