from firecloud import api as fapi
from firecloud.errors import FireCloudServerError

from ..utils import *

logger = logging.getLogger(__name__)


def get_workspace_bucket(ns: str, ws: str) -> str:
    """
    Get bucket (starting with fc-) attached to the workspace

    :param ns:
    :param ws:
    :return:
    """
    return _query_workspace(ns, ws)['bucketName']


def get_workspace_attribute(ns: str, ws: str, attribute_name: str):
    """
    Get value of an existing attribute.

    :param ns:
    :param ws:
    :param attribute_name:
    :return:
    """
    attributes = _query_workspace(ns, ws)['attributes']
    if attribute_name not in attributes:
        msg = f"Queried attribute {attribute_name} not set up yet in workspace {ns}/{ws}."
        logger.error(msg)
        raise KeyError(msg)

    return attributes[attribute_name]


def update_workspace_attribute(ns: str, ws: str, attribute_name: str, attribute_value: str) -> None:
    """
    For updating, or making new attribute of a workspace.

    Most common usage is to add workspace level data.
    :param ns:
    :param ws:
    :param attribute_name:
    :param attribute_value:
    :return:
    """

    response = fapi.update_workspace_attributes(ns, ws,
                                                [
                                                    {
                                                        "op": "AddUpdateAttribute",
                                                        "attributeName": attribute_name,
                                                        "addUpdateAttribute": attribute_value
                                                    }
                                                ])
    if not response.ok:
        logger.error(f"Failed to add/update attribute {attribute_name} for workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)


def remove_workspace_attribute(ns: str, ws: str, attribute_name: str) -> None:
    """
    Remove workspace attribute.

    :param ns:
    :param ws:
    :param attribute_name:
    :return:
    """
    response = fapi.update_workspace_attributes(ns, ws,
                                                [
                                                    {
                                                        "op": "RemoveAttribute",
                                                        "attributeName": attribute_name
                                                    }
                                                ])
    if not response.ok:
        logger.error(f"Failed to add/update attribute {attribute_name} for workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)


########################################################################################################################
def _query_workspace(ns, ws):
    response = fapi.get_workspace(ns, ws)
    if not response.ok:
        logger.error(f"Failed to query workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)
    return response.json()['workspace']
