import datetime

import gcsfs
from dateutil import parser
from firecloud.errors import FireCloudServerError
from tqdm.auto import tqdm

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


def get_workspace_attributes(ns: str, ws: str, max_attempts: int = 2) -> dict[str, str]:
    """
    Get names and values of all attributes in a workspace.

    :param ns:
    :param ws:
    :param max_attempts:
    :return:
    """
    return _query_workspace(ns, ws, max_attempts)['attributes']


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


########################################################################################################################
def compute_cost_this_month(ns: str, ws: str, user_account: str = None) -> float:
    month_beg = parser.parse(f"{datetime.datetime.now().year}-{datetime.datetime.now().month}-01T00:00:00.000Z")
    return compute_cost_between_dates(ns, ws,
                                      month_beg, datetime.datetime.now(tz=datetime.timezone.utc),
                                      user_account)


def compute_cost_this_year(ns: str, ws: str, user_account: str = None) -> float:
    year_beg = parser.parse(f"{datetime.datetime.now().year}-01-01T00:00:00.000Z")
    return compute_cost_between_dates(ns, ws,
                                      year_beg, datetime.datetime.now(tz=datetime.timezone.utc),
                                      user_account)


def compute_cost_between_dates(ns: str, ws: str,
                               earlier_cutoff: datetime.datetime,
                               later_cutoff: datetime.datetime,
                               user_account: str = None) -> float:
    """
    Return WDL-related compute cost accrued for a workspace during the specified cutoff timestamps.

    This uses the cost reported on Terra, but has intrinsic inaccuracies.
    For example: because workflows take time to run, workflows launched at the end of last month—assuming
    you're interested in the cost for the current month—will not be included, even though they'll
    be billed in the current month by cloud service provider.
    Another example is for workflows finished shortly before this query.
    Because it takes up to 48 hours before all costs are reported back by the cloud service provided,
    these workflows' cost will not be included.

    Hence, consider the number returned as a reasonably good estimate, but not 100% accurate.
    """

    response = retry_fiss_api_call('list_submissions', 2,
                                   namespace=ns, workspace=ws)
    if not response.ok:
        logger.error(f"Failed to retrieve list of submissions in {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    sorted_subs = sorted(response.json(), key=lambda sub: parser.parse(sub['submissionDate']))

    # filter by user account, if requested
    if user_account:
        relevant_subs = _filter_sub_by_submitter(sorted_subs, user_account)
    else:
        relevant_subs = sorted_subs

    # filter by timestamps
    subs_filtered_by_timestamp = _filter_sub_by_timestamp(relevant_subs, earlier_cutoff, later_cutoff)

    tqdm.pandas()
    cost = 0.0
    for sub in tqdm(subs_filtered_by_timestamp):
        response = retry_fiss_api_call('get_submission', 2,
                                       namespace=ns, workspace=ws,
                                       submission_id=sub['submissionId'])
        if not response.ok:
            logger.error(f"FISS request for submission {sub['submissionId']} failed repeatedly.")
            raise FireCloudServerError(response.status_code, response.text)
        cost += response.json()['cost']

    logger.info(f"For the month {datetime.datetime.now().strftime('%B')} of {datetime.datetime.now().year},"
                f" a total of {len(relevant_subs)} WDL submissions have costed you ${cost} so far for"
                f" {ns}/{ws} ")
    return cost


def _filter_sub_by_submitter(subs: List[dict], submitter_account: str) -> list:
    return [sub for sub in subs if submitter_account == sub['submitter']]


def _filter_sub_by_timestamp(submissions: List[dict],
                             earlier_cutoff: datetime.datetime,
                             later_cutoff: datetime.datetime) -> list:

    """
    Filter submissions in the provided list of submissions by submission date.
    Note that UTC time is used on Terra for submission timestamps.
    If the cutoff timestamps provided are not in UTC, provide timestamp_tz.
    :param submissions: list of submissions to filter on
    :param earlier_cutoff: (inclusive) submissions no earlier than this timestamp will be retained
    :param later_cutoff: (inclusive) submissions no later than this timestamp will be retained
    :return:
    """
    return [sub for sub in submissions
            if earlier_cutoff <= parser.parse(sub['submissionDate']) <= later_cutoff]

def backup_workspace(ns: str, ws: str, backup_bucket: str, max_attempts: int = 2) -> None:
    """
    Backup a workspace by copying all tables, notebooks, workflow information, and job history to a given google bucket.

    :param ns: Namespace of the workspace to back up.
    :param ws: Name of the workspace to back up.
    :param backup_bucket: Destination bucket into which to save the backup.
    :param max_attempts: Maximum number of attempts to make for each API call.
    :return:
    """

    # Track start and end time so we can report how long the backup took:
    start_time = time.time()

    # Remove leading `gs://` if present:
    backup_bucket = backup_bucket.replace("gs://", "")

    # Get our entity types so we know what to dump:
    entity_types = fapi.list_entity_types(ns, ws).json()

    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    workspace_bucket = fapi.get_workspace(ns, ws).json()["workspace"]["bucketName"]
    backup_folder_path = f"{timestamp}"

    # Create our timestamped backup bucket:
    storage_client = storage.Client()
    bucket = storage_client.bucket(backup_bucket)

    # Backup the entities:
    # Iterate over entity types and dump each one to a separate TSV:
    print(f"Writing workspace entities to backup dir:")
    print(f"gs://{backup_bucket}/{backup_folder_path}/tables")
    for et in entity_types:
        print(f"\t{et}")
        tbl, _ = load_table(ns, ws, et)
        tbl = fix_nans(tbl)
        table_name = f"{timestamp}_{ns}_{ws}_{et}.tsv.gz"

        # Write our table to our bucket:
        blob = bucket.blob(f"{backup_folder_path}/tables/{table_name}")

        with io.StringIO() as buf:
            tbl.to_csv(buf, sep="\t", index=False)
            with blob.open('wb') as f:
                f.write(gzip.compress(bytes(buf.getvalue(), 'utf-8')))
    print('Done.')

    # Now backup the notebooks:
    print("Writing notebooks to backup dir:")
    print(f"gs://{backup_bucket}/{backup_folder_path}/notebooks")
    for notebook_blob in storage_client.list_blobs(workspace_bucket, prefix='notebooks'):
        original_name = notebook_blob.name[notebook_blob.name.find("/") + 1:]
        print(f"\t{original_name}")
        notebook_name = f"{timestamp}_{ns}_{ws}_{original_name}"
        bucket.copy_blob(notebook_blob, bucket, new_name=f"{backup_folder_path}/notebooks/{notebook_name}")
    print("Done.")

    # Now store workspace attributes:
    ws_dict = lrma_workspace_utils._query_workspace(ns, ws, 2)
    ws_attributes_dict = ws_dict['attributes']

    # Write our dict to our bucket:
    print(f"Writing workspace attributes to backup dir:")
    print(f"gs://{backup_bucket}/{backup_folder_path}")
    _write_json_file_to_bucket(f"{backup_folder_path}", bucket, timestamp, ns, ws, ws_attributes_dict,
                               f"workspace_attributes")

    print(f"Done.")
    print()

    # and store the workspace metadata:
    del ws_dict['attributes']

    print(f"Writing workspace metadata to backup dir:")
    print(f"gs://{backup_bucket}/{backup_folder_path}")
    _write_json_file_to_bucket(f"{backup_folder_path}", bucket, timestamp, ns, ws, ws_dict,
                               f"workspace_metadata")
    print(f"Done.")

    # Now store workspace method (i.e. workflow) information:
    response = lrma_workspace_utils.retry_fiss_api_call('list_workspace_configs', 2, ns, ws, True)
    workflow_dict = response.json()

    # Sort our workflows in alphabetical order:
    workflow_dict = sorted(workflow_dict, key=lambda k: k['name'])

    print(f"Writing workflow high-level information to backup dir:")
    print(f"gs://{backup_bucket}/{backup_folder_path}")

    _write_json_file_to_bucket(f"{backup_folder_path}", bucket, timestamp, ns, ws, workflow_dict,
                               f"workflows")
    print(f'Done.')

    # Get the metadata, default inputs, and default outputs for each workflow:
    print(f"Writing workflow metadata, inputs, and outputs to backup dir:")
    print(f"gs://{backup_bucket}/{backup_folder_path}/workflows")

    for workflow_name in [w["name"] for w in workflow_dict]:
        print(f"\t{workflow_name}")

        response = lrma_workspace_utils.retry_fiss_api_call('get_workspace_config', 2, ns, ws, ns,
                                                            workflow_name)
        metadata = response.json()
        inputs = metadata['inputs']
        outputs = metadata['outputs']
        del metadata['inputs']
        del metadata['outputs']

        _write_json_file_to_bucket(f"{backup_folder_path}/workflows", bucket, timestamp, ns, ws, inputs,
                                   f"{workflow_name}_workflow_inputs")

        _write_json_file_to_bucket(f"{backup_folder_path}/workflows", bucket, timestamp, ns, ws, outputs,
                                   f"{workflow_name}_workflow_outputs")

        _write_json_file_to_bucket(f"{backup_folder_path}/workflows", bucket, timestamp, ns, ws, outputs,
                                   f"{workflow_name}_workflow_metadata")
    print(f"Done")

    # Finally store submissions:
    response = lrma_workspace_utils.retry_fiss_api_call('list_submissions', 2, ns, ws)
    submissions_dict = response.json()

    # Sort our submissions from most recent to least recent:
    submissions_dict = sorted(submissions_dict, key=lambda k: k['submissionDate'], reverse=True)

    print(f"Writing workspace job history to backup dir:")
    print(f"gs://{backup_bucket}/{backup_folder_path}")
    _write_json_file_to_bucket(backup_folder_path, bucket, timestamp, ns, ws, submissions_dict, "workspace_job_history")
    print(f"Done.")

    # Track end time so we can calculate elapsed time for backup:
    end_time = time.time()

    now_utc = datetime.datetime.utcnow()
    timezone = pytz.timezone('America/New_York')
    now_et = now_utc.astimezone(timezone)
    time_string = now_et.strftime("%A %B %d at %H:%M:%S ET")
    print(f"Backup completed on {time_string}")
    print(f"Backup location: gs://{backup_bucket}/{backup_folder_path}")
    print(f"Elapsed time: {end_time - start_time:2.2f}s")


def _write_json_file_to_bucket(backup_folder_path, bucket, timestamp, namespace, workspace, json_object, file_base_name):
    """
    Write a JSON object to a GCS bucket.
    :param backup_folder_path: The folder path in the bucket to write to.
    :param bucket: The bucket to write to.
    :param timestamp: The timestamp of the backup.
    :param namespace: The namespace of the workspace.
    :param workspace: The name of the workspace.
    :param json_object: The JSON object to write.
    :param file_base_name: The base name of the file to write.
    """
    # Write our dict to our bucket:
    blob = bucket.blob(f"{backup_folder_path}/{timestamp}_{namespace}_{workspace}_{file_base_name}.json.gz")
    with blob.open('wb') as f:
        f.write(gzip.compress(bytes(json.dumps(json_object, indent=4), 'utf-8')))