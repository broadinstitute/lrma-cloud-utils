import datetime

import gcsfs
from dateutil import parser
from firecloud.errors import FireCloudServerError

from lrmaCU.utils import *

logger = logging.getLogger(__name__)


def get_workspace_bucket(ns: str, ws: str, max_attempts: int = 2) -> str:
    """
    Get bucket (starting with fc-) attached to the workspace

    :param ns:
    :param ws:
    :param max_attempts:
    :return:
    """
    return _query_workspace(ns, ws, max_attempts)['bucketName']


def get_workspace_attribute(ns: str, ws: str, attribute_name: str, max_attempts: int = 2):
    """
    Get value of an existing attribute.

    :param ns:
    :param ws:
    :param attribute_name:
    :param max_attempts:
    :return:
    """
    attributes = _query_workspace(ns, ws, max_attempts)['attributes']
    if attribute_name not in attributes:
        msg = f"Queried attribute {attribute_name} not set up yet in workspace {ns}/{ws}."
        logger.error(msg)
        raise KeyError(msg)

    return attributes[attribute_name]


def update_workspace_attribute(ns: str, ws: str, attribute_name: str, attribute_value: str,
                               max_attempts: int = 2) -> None:
    """
    For updating, or making new attribute of a workspace.

    Most common usage is to add workspace level data.
    :param ns:
    :param ws:
    :param attribute_name:
    :param attribute_value:
    :param max_attempts:
    :return:
    """
    response = retry_fiss_api_call('update_workspace_attributes', max_attempts,
                                   ns, ws,
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


def remove_workspace_attribute(ns: str, ws: str, attribute_name: str,
                               max_attempts: int = 2) -> None:
    """
    Remove workspace attribute.

    :param ns:
    :param ws:
    :param attribute_name:
    :param max_attempts:
    :return:
    """
    response = retry_fiss_api_call('update_workspace_attributes', max_attempts,
                                   ns, ws,
                                   [
                                       {
                                           "op": "RemoveAttribute",
                                           "attributeName": attribute_name
                                       }
                                   ])
    if not response.ok:
        logger.error(f"Failed to remove attribute {attribute_name} for workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)


########################################################################################################################
def _query_workspace(ns, ws, max_attempts: int = 2):
    response = retry_fiss_api_call('get_workspace', max_attempts, ns, ws)
    if not response.ok:
        logger.error(f"Failed to query workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)
    return response.json()['workspace']


########################################################################################################################
def get_workspace_submissions_old_enough(ns: str, ws: str, days_back: int,
                                         max_attempts: int = 2) -> list:
    """
    Get submissions in a workspace that's old enough (older than the days_back arg)
    :param ns:
    :param ws:
    :param days_back: workflows submitted earlier than this date will be included in the returned submission IDs
    :param max_attempts:
    :return: a list of submission ids
    """

    response = retry_fiss_api_call('list_submissions', max_attempts,
                                   ns, ws)
    if not response.ok:
        raise FireCloudServerError(response.status_code, response.text)

    cut_off = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=days_back)

    def is_old(dd: dict):
        return parser.parse(dd['submissionDate']) < cut_off and 'Done' == dd['status']
    old_enough = filter(is_old, response.json())

    return sorted(list(old_enough), key=lambda dd: dd['submissionDate'])


def get_submission_ids_to_delete(ns: str, ws: str, days_back: int,
                                 submissions_to_skip: List[str],
                                 max_attempts: int = 2) -> List[str]:
    """
    Get a list of submissions IDs in a workspace whose submission folders can be deleted ,
    where the submission was created over "days_back" days ago, and not included in "submissions_to_skip".

    :param ns:
    :param ws:
    :param days_back:
    :param submissions_to_skip:
    :param max_attempts:
    :return:
    """

    workspace_bucket = get_workspace_bucket(ns, ws, max_attempts)

    # get existing folders
    terra_bucket_fs = gcsfs.GCSFileSystem()
    existing_folders = [fd.split('/')[-1] for fd in terra_bucket_fs.ls(f"{workspace_bucket}/submissions/")]

    # get submissions old enough
    x = get_workspace_submissions_old_enough(ns, ws, days_back=days_back)
    y = list(map(lambda d: d['submissionId'], x))

    # intermediate folder still exists && old enough && not on whitelist
    z = [sub for sub in y
         if (sub in existing_folders) and (sub not in submissions_to_skip)]
    return z
