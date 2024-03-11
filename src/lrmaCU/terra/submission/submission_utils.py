import collections.abc
import copy
import datetime
import logging
import os
from enum import Enum
from typing import List, Dict, Tuple

import pytz
from dateutil import parser
from firecloud.errors import FireCloudServerError

from lrmaCU.terra.concepts_help import WORKFLOW_CONFIG_EXAMPLE_JSON, WORKFLOW_CONCEPT_EXPLAINER
from lrmaCU.terra.table_utils import add_one_set
from lrmaCU.utils import retry_fiss_api_call

########################################################################################################################

logger = logging.getLogger(__name__)

local_tz = pytz.timezone(os.environ.get('TZ', 'UTC'))

PRACTICAL_DAYS_LOOKBACK = 7  # made an implicit assumption: 7 days back is the max


# POST-like ############################################################################################################
def change_config_of_method(ns: str, ws: str, method_name: str,
                            new_root_entity_type: str = None,
                            new_input_names_and_values: dict = None,
                            existing_input_names_but_new_values: dict = None,
                            new_branch: str = None,
                            max_attempts: int = 2,
                            restore_config_on_error: bool = True) -> dict:
    """
    Supporting common—but currently limited—scenarios where one wants to update a config of a method.

    Note this does NOT make any efforts in making sure the new configurations make sense.
    That is, one could potentially mismatch the method with wrong root entities,
    and/or providing non-existent branches.
    It's the user's responsibility to make sure the intended config is correct.

    In particular, there are special rules for
        * new_input_names_and_values
        * existing_input_names_but_new_values
    on keys:
       they must follow the same naming rules as you see in the Terra GUI.
       That is, prefix with the variable name with the value in the first column of the corresponding row, and a ".".
    on values:
       array-like values must be pre-formatted into a single string in a way similar to the following:
         '[\"' + '\",\"'.join(array) + '\"]'

    The old config is returned, in case this is a one-time change and one wants to immediately revert back
    once something is done with the new config (e.g. run a one-off analysis).
    If this indeed is the case, checkout restore_method_config(...) in this module.
    :param ns:
    :param ws:
    :param method_name:
    :param new_root_entity_type: when one wants to re-configure a method's root entity
    :param new_input_names_and_values: when one wants to re-configure some input values, and/or add new input values
    :param existing_input_names_but_new_values: input names exist, but take on new value
    :param new_branch: when one wants to switch to a different branch, where supposedly the method is updated.
    :param restore_config_on_error: when the new config is invalid, restore to the old config or not
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return: current config before the update
    """
    if new_root_entity_type is None \
            and new_input_names_and_values is None \
            and new_branch is None \
            and existing_input_names_but_new_values is None:
        raise ValueError(f"Requesting to change config of method: {method_name}, but not changing anything?!")

    logger.debug(WORKFLOW_CONFIG_EXAMPLE_JSON)
    logger.debug(WORKFLOW_CONCEPT_EXPLAINER)

    response = retry_fiss_api_call('get_workspace_config', max_attempts,
                                   ns, ws, ns, method_name)
    if not response.ok:
        logger.error(f"Failed to retrieve current config for method {ns}/{ws}:{method_name}.")
        raise FireCloudServerError(response.status_code, response.text)
    current_config = copy.deepcopy(response.json())

    updated = copy.deepcopy(current_config)
    if new_root_entity_type is not None:
        updated = _update_config_dict(updated, {'rootEntityType': new_root_entity_type})

    if new_branch is not None:
        updated_wdl_version = copy.deepcopy(updated['methodRepoMethod'])
        updated_wdl_version['methodVersion'] = new_branch
        updated_wdl_version['methodUri'] = '/'.join(updated_wdl_version['methodUri'].split('/')[:-1]) + '/' + new_branch
        updated = _update_config_dict(updated, {'methodRepoMethod': updated_wdl_version})

    if new_input_names_and_values is not None:
        updated_inputs = copy.deepcopy(updated['inputs'])
        updated_inputs.update(new_input_names_and_values)
        updated = _update_config_dict(updated, {'inputs': updated_inputs})
    if existing_input_names_but_new_values is not None:
        updated_inputs = copy.deepcopy(updated['inputs'])
        updated_inputs.update(existing_input_names_but_new_values)
        updated = _update_config_dict(updated, {'inputs': updated_inputs})

    updated['methodConfigVersion'] = updated['methodConfigVersion'] + 1  # don't forget this

    response = retry_fiss_api_call('update_workspace_config', max_attempts,
                                   ns, ws, ns,
                                   configname=method_name, body=updated)
    if not response.ok:
        logger.error(f"Failed to update config for method:\n  {ns}/{ws}:{method_name}.")
        raise FireCloudServerError(response.status_code, response.text)

    # validate, but unsure how reliable this is
    if not is_method_config_valid(ns, ws, method_name, max_attempts):
        if restore_config_on_error:
            previous_config = current_config
            restore_method_config(ns, ws, method_name, previous_config, max_attempts=2)
            logger.error(f"The config for the method {ns}/{ws}:{method_name} is updated but doesn't validate."
                         f"  So we've reverted it back. Manual intervention needed.")
        raise ValueError(f"After updating the config for {method_name} in {ns}/{ws}, it's config is no longer valid.")

    return current_config


def is_method_config_valid(ns: str, ws: str, method_name: str,
                           max_attempts: int = 2) -> bool:
    """
    Basic check on if a method's config on Terra is valid or not.
    :param ns:
    :param ws:
    :param method_name:
    :param max_attempts:
    :return:
    """

    logger.debug(WORKFLOW_CONFIG_EXAMPLE_JSON)
    logger.debug(WORKFLOW_CONCEPT_EXPLAINER)

    response = retry_fiss_api_call('validate_config', max_attempts,
                                   ns, ws, ns, method_name)
    if not response.ok:
        raise FireCloudServerError(response.status_code, response.text)

    config_is_valid = True
    jes = response.json()
    for k in ['extraInputs', 'invalidInputs', 'invalidOutputs', 'missingInputs']:
        if 0 != len(jes[k]):
            config_is_valid = False
            break
    return config_is_valid


def restore_method_config(ns: str, ws: str, method_name: str, old_config: dict,
                          max_attempts: int = 2) -> None:
    """
    Restore a config of the method to an old value, presumably validated.

    :param ns:
    :param ws:
    :param method_name:
    :param old_config:
    :param max_attempts:
    :return:
    """

    logger.debug(WORKFLOW_CONFIG_EXAMPLE_JSON)
    logger.debug(WORKFLOW_CONCEPT_EXPLAINER)

    to_upload = copy.deepcopy(old_config)
    response = retry_fiss_api_call('get_workspace_config', max_attempts,
                                   ns, ws, ns, method_name)
    if not response.ok:
        logger.error(f"Failed to retrieve current config for method {ns}/{ws}:{method_name}.")
        raise FireCloudServerError(response.status_code, response.text)
    to_upload['methodConfigVersion'] = response.json()['methodConfigVersion'] + 1

    response = retry_fiss_api_call('update_workspace_config', max_attempts,
                                   ns, ws, ns,
                                   configname=method_name, body=to_upload)
    if not response.ok:
        logger.error(f"Failed to restore config for {ns}/{ws}:{method_name}.")
        raise FireCloudServerError(response.status_code, response.text)

    response = retry_fiss_api_call('validate_config', max_attempts,
                                   ns, ws, ns, method_name)
    if not response.ok:
        logger.error(f"The config for the method {ns}/{ws}:{method_name} is restored to doesn't validate."
                     f" Manual intervention needed.")
        raise FireCloudServerError(response.status_code, response.text)


def _update_config_dict(old_config: dict, new_config: dict) -> dict:
    """
    DFS updating a (nested) dict, modeling the config of a method.

    :param old_config:
    :param new_config:
    :return:
    """
    for k, v in new_config.items():
        if isinstance(v, collections.abc.Mapping):
            old_config[k] = _update_config_dict(old_config.get(k, {}), v)
        else:
            old_config[k] = v
    return old_config


def verify_before_submit(ns: str, ws: str,
                         method_name: str,
                         etype: str, enames: List[str],
                         batch_type_name: str = None, expression: str = None,
                         days_back: int = None, count: int = None,
                         force: bool = False,
                         max_attempts: int = 2,
                         use_callcache: bool=True,
                         delete_intermediate_output_files: bool=False,
                         use_reference_disks: bool=False,
                         memory_retry_multiplier: float=0,
                         workflow_failure_mode:str="", user_comment: str=""
                         ) -> dict or None:
    """
    For a list of entities, conditionally submit a job: if the entity isn't being analyzed already.

    We strongly recommend using keyword argument when calling this function.
    (Because when Terra/FISS provides new interfaces, we may decide to update this function's interface too).

    One can also specify, for entities that fail to be analyzed with the requested method repeatedly,
    whether to go ahead or not, as one may want to manually checkout what's wrong there.
    By not providing the two arguments, you are signaling this isn't necessary.
    Check get_repeatedly_failed_entities(...) for more information.

    When there are multiple entities given in enames, one can specify an expression for batch submission.
    For example, say etype is 'sample', and enames are samples to be analyzed with method BLAH.
    BLAH is configured in a way such that its root entity is 'sample', i.e. same as etype.
    In this case, "expression" can simply be "this.samples".
    This is intuitive if one has configured workflows on Terra-GUI whose root entity is "sample_set",
    but some inputs takes attributes of individual 'sample's.
    :param ns:
    :param ws:
    :param method_name:
    :param etype:
    :param enames:
    :param batch_type_name: type name of the resulting set, when batch submission mode is turned on
    :param expression: if not None, will submit all entities given in enames in one batch
                       Note that this will create a dummy set, for the purpose of batch submission.
    :param days_back: how many day back to check for repeated failures
    :param count: repeated failure threshold, >= which it won't be re-submitted.
    :param force: if True, forcefully launch analysis on every entity provided except those being analyzed at the moment
                  by skipping the check if the entity has been successfully analyzed, or caused repeated failures
    :param max_attempts: for retrying when seeing connection reset by peer error
    :param use_callcache: args like this are forwarded to FISS
    :return: failures as a dictionary, where the keys are entities that failed to launch methods on, and
                                             the values are responses from FireCloud API
    """

    if not is_method_config_valid(ns, ws, method_name, max_attempts):
        raise ValueError(f"Validation of config for {method_name} in {ns}/{ws} failed.")

    logger.info(WORKFLOW_CONCEPT_EXPLAINER)

    to_submit_by_batch = not (1 == len(enames) or expression is None)
    failures = dict()
    if not to_submit_by_batch:
        for e in _analyzable_entities(ns, ws, method_name, etype, enames, days_back, count, force, max_attempts):
            response = retry_fiss_api_call('create_submission', max_attempts,
                                           wnamespace=ns, workspace=ws, cnamespace=ns,
                                           config=method_name,
                                           entity=e,
                                           etype=etype,
                                           use_callcache=use_callcache,
                                           delete_intermediate_output_files=delete_intermediate_output_files,
                                           use_reference_disks=use_reference_disks,
                                           memory_retry_multiplier=memory_retry_multiplier,
                                           workflow_failure_mode=workflow_failure_mode,
                                           user_comment=user_comment)
            if response.ok:
                logger.info(f"Submitted {etype} {e} for analysis with {method_name}.")
            else:
                failures[e] = response.json()
                logger.warning(f"Failed to submit {etype} {e} for analysis with {method_name} due to"
                               f" \n {response.json()}")
        if failures:
            import pprint
            logger.error(f"Failed to submit WDL {method_name} for the following entities:\n"
                         f"{pprint.pformat(failures)}")
    else:
        if batch_type_name is None:
            raise ValueError("When submitting in batching mode, batch_type_name must be specified")

        analyzable_entities = \
            _analyzable_entities(ns, ws, method_name, etype, enames, days_back, count, force, max_attempts)
        if 0 == len(analyzable_entities):
            logger.warning(f"No analyzable entities in\n  {enames}")
            return

        now_str = datetime.datetime.now(tz=local_tz).strftime("%Y-%m-%dT%H-%M-%S")
        dummy_set_name_following_terra_convention = f'{method_name}_{now_str}_lrmaCU'
        # this is the magic to create batch submissions
        add_one_set(ns, ws,
                    etype=batch_type_name,
                    ename=dummy_set_name_following_terra_convention,
                    member_type=etype,
                    members=analyzable_entities,
                    attributes=None,
                    max_attempts=max_attempts)
        response = retry_fiss_api_call('create_submission', max_attempts,
                                       wnamespace=ns, workspace=ws, cnamespace=ns,
                                       config=method_name,
                                       entity=dummy_set_name_following_terra_convention,
                                       etype=batch_type_name,
                                       expression=expression,
                                       use_callcache=use_callcache,
                                       delete_intermediate_output_files=delete_intermediate_output_files,
                                       use_reference_disks=use_reference_disks,
                                       memory_retry_multiplier=memory_retry_multiplier,
                                       workflow_failure_mode=workflow_failure_mode,
                                       user_comment=user_comment)
        if not response.ok:
            logger.error(f"Failed to submit batch job using batch {dummy_set_name_following_terra_convention}"
                         f" to workspace {ns}/{ws} with method {method_name}.")
            failures[dummy_set_name_following_terra_convention] = response.json()
        logger.info(f"Submitted {etype}s {enames} for analysis with {method_name} in a batch.")

    return failures


def custom_submission_without_root_entity(ns: str, ws: str,
                                          method_name: str,
                                          use_callcache: bool,
                                          delete_intermediate_output_files: bool = False,
                                          use_reference_disks: bool = False,
                                          memory_retry_multiplier: float = 0,
                                          workflow_failure_mode:str = "",
                                          user_comment: str = "",
                                          max_attempts: int = 2):

    response = retry_fiss_api_call('create_submission', max_attempts,
                                   wnamespace=ns, workspace=ws, cnamespace=ns, config=method_name,
                                   entity=None, etype=None, expression=None,
                                   use_callcache=use_callcache,
                                   delete_intermediate_output_files=delete_intermediate_output_files,
                                   use_reference_disks=use_reference_disks,
                                   memory_retry_multiplier=memory_retry_multiplier,
                                   workflow_failure_mode=workflow_failure_mode,
                                   user_comment=user_comment)
    if not response.ok:
        raise FireCloudServerError(response.status_code,
                                   f"Failed to submit for a method {method_name} that doesn't have root-entity.")


def delete_workspace_submission_folders(ns: str, ws: str,
                                        cleanup_method_name: str,
                                        submissions_id_to_delete: List[str],
                                        use_callcache: bool,
                                        delete_intermediate_output_files: bool = False,
                                        use_reference_disks: bool = False,
                                        memory_retry_multiplier: float = 0,
                                        workflow_failure_mode:str = "",
                                        user_comment: str = "",
                                        max_attempts: int = 2) -> None:

    """
    Delete a workspace's submissions sub-folders for the provided submission IDs.

    :param ns:
    :param ws:
    :param cleanup_method_name: the name of the cleanup method in the workspace,
            e.g. "CleanupIntermediate" at https://tinyurl.com/pyh8mmuy
    :param submissions_id_to_delete:
    :param use_callcache:
    :param delete_intermediate_output_files:
    :param use_reference_disks:
    :param memory_retry_multiplier:
    :param workflow_failure_mode:
    :param user_comment:
    :param max_attempts:
    :return:
    """

    # this re-formatting is critical for Terra to understand
    to_delete = "[\"" + '\",\"'.join(submissions_id_to_delete) + "\"]"

    change_config_of_method(ns, ws,
                            method_name=cleanup_method_name,
                            existing_input_names_but_new_values={f'{cleanup_method_name}.submissionIDs': to_delete},
                            max_attempts=max_attempts)

    custom_submission_without_root_entity(ns, ws,
                                          method_name=cleanup_method_name,
                                          use_callcache=use_callcache,
                                          delete_intermediate_output_files=delete_intermediate_output_files,
                                          use_reference_disks=use_reference_disks,
                                          memory_retry_multiplier=memory_retry_multiplier,
                                          workflow_failure_mode=workflow_failure_mode,
                                          user_comment=user_comment,
                                          max_attempts=max_attempts)


# GET-like #############################################################################################################
# type alias for the json parsing response of a particular submission, where
# the response is generated by a Firecloud API (or FISS) call: list_submissions()
SUBMISSION_INFO = Dict[str, object]


def get_submissions_for_method_config(ns: str, ws: str, method_config: str, days_back: int,
                                      max_attempts: int = 2) -> List[SUBMISSION_INFO]:
    """
    Get submissions information for a particular method_config, up to a certain datetime back.

    Again, a reminder here that when a method's config is changed,
    its name in submissions prior to the change are updated automatically by Terra
    with a postfix random string.
    Since those submissions are technically for a different method_config, they will not be counted here.
    (using a regex to match may not be what you want either).

    :param ns:
    :param ws:
    :param method_config:
    :param days_back:
    :param max_attempts: for retrying when seeing connection reset by peer error

    :return: a list of response Json rendering (dict) of the relevant submissions
             Note that the level of details are not comparable to that retrieved with fapi.get_submission(...)
    """

    logger.debug(WORKFLOW_CONCEPT_EXPLAINER)

    response = retry_fiss_api_call('list_submissions', max_attempts,
                                   ns, ws)
    if not response.ok:
        logger.error(f"Failed to list submissions in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    # first get all submissions
    all_submissions = sorted(response.json(), key=lambda sub: parser.parse(sub['submissionDate']))

    # then filter down by timestamp (submissions too old aren't interesting, most likely)
    utc_cut_off_date = pytz.utc.localize(datetime.datetime.utcnow()) - datetime.timedelta(days=days_back)

    return [sub for sub in all_submissions
            if sub['methodConfigurationName'] == method_config
            and parser.parse(sub['submissionDate']) > utc_cut_off_date]


class WorkflowExeStatus(str, Enum):
    """
    Modeling the status of a particular workflow execution.
    """
    FAIL = 'Failed'
    SUCC = 'Succeeded'
    RUNN = 'Running'
    ABORTED = 'Aborted'
    ABORTING = 'Aborting'


class EntityStatuses:
    """
    Modeling the number of times an entity has been
    successfully/unsuccessfully processed
    by a particular method_config,
    the latest status, and the time of that.

    Again, note that this specific to a method_config.
    """

    def __init__(self, status: str, timing: datetime.datetime, method_config: str, ename: str, etype: str):
        self.etype = etype
        self.ename = ename
        self.method_config = method_config

        self.latest_timestamp = None
        self.latest_status = None

        self.succ_cnt = 0
        self.fail_cnt = 0

        self.update_latest_status_and_timing(status, timing)

    def __str__(self):
        return (f"{self.etype} {self.ename} has been analyzed with {self.method_config}: "
                f"    successfully   {self.succ_cnt} times, "
                f"    unsuccessfully {self.fail_cnt} times."
                f" The latest status is {self.latest_status} at {self.latest_timestamp}."
                )

    def _bump_succ(self):
        self.succ_cnt += 1

    def _bump_fail(self):
        self.fail_cnt += 1

    def update_latest_status_and_timing(self, status: str or WorkflowExeStatus, timing: datetime.datetime):
        input_st = WorkflowExeStatus(status) if isinstance(status, str) else status
        # status count need to be bumped regardless
        if WorkflowExeStatus.SUCC == input_st:
            self._bump_succ()
        elif WorkflowExeStatus.FAIL == input_st:
            self._bump_fail()

        if self.latest_timestamp is None or timing > self.latest_timestamp:
            self.latest_timestamp = timing
            self.latest_status = input_st


def get_entities_in_a_submission(ns: str, ws: str, submission_id: str,
                                 max_attempts: int = 2) \
        -> Dict[WorkflowExeStatus, List[Tuple[str, datetime.datetime]]]:
    """
    Get entities in a submission, together with time when it was last updated.
    Results are grouped by the status of their respective workflows.

    :param ns:
    :param ws:
    :param submission_id: id
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return: Terra uuid for the (success, failed, running) entities in that batch submission,
             and time when it's last updated
    """
    response = retry_fiss_api_call('get_submission', max_attempts,
                                   ns, ws, submission_id)
    if not response.ok:
        logger.error(f"Failed to get submission {submission_id} in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    batch_submission_json = response.json()
    success = list()
    failure = list()
    aborted = list()
    aborting = list()
    running = list()
    for w in batch_submission_json['workflows']:
        e = w['workflowEntity']['entityName']
        t = parser.parse(w['statusLastChangedDate'])
        st = w['status']
        if WorkflowExeStatus.FAIL.value == st:
            failure.append((e, t))
        elif WorkflowExeStatus.SUCC.value == st:
            success.append((e, t))
        elif WorkflowExeStatus.ABORTED.value == st:
            aborted.append((e, t))
        elif WorkflowExeStatus.ABORTING.value == st:
            aborting.append((e, t))
        elif WorkflowExeStatus.RUNN.value == st:
            running.append((e, t))
        else:
            raise ValueError(f"Seeing workflow execution status not seen before: {st}. lrmaCUX needs to be updated.")

    return {WorkflowExeStatus.SUCC: success,
            WorkflowExeStatus.FAIL: failure,
            WorkflowExeStatus.ABORTED: aborted,
            WorkflowExeStatus.ABORTING: aborting,
            WorkflowExeStatus.RUNN: running}


# todo: this gather is quite slow, anyway to speed it up?
def get_entities_analyzed_by_method_config(ns: str, ws: str, method_config: str, days_back: int, etype: str,
                                           max_attempts: int = 2) \
        -> List[EntityStatuses]:
    """
    Get entities of the requested type, that have been analyzed by a method_config.

    :param ns:
    :param ws:
    :param method_config:
    :param days_back:
    :param etype:
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return:
    """

    relevant_submissions = get_submissions_for_method_config(ns, ws, method_config, days_back, max_attempts)
    return _collect_entities_and_statuses(ns, ws, method_config, etype, relevant_submissions, max_attempts)


def get_repeatedly_failed_entities(ns: str, ws: str, method_config: str, etype: str, days_back: int, count: int,
                                   max_attempts: int = 2,
                                   entity_statuses_under_the_methconf: List[EntityStatuses] = None) \
        -> List[EntityStatuses]:
    """
    Get entities that **repeatedly** failed to be processed by a particular method_config,
    up to a certain datetime back.

    :param ns:
    :param ws:
    :param method_config:
    :param etype:
    :param days_back:
    :param count: entities that failed to be processed, >= this number of times, will be reported
    :param max_attempts: for retrying when seeing connection reset by peer error
    :param entity_statuses_under_the_methconf: if you've collected the entities' statuses previously, happy to use that
    :return: a dict {entity_name: failure_count}, within the days_back limit
    """

    entity_statuses = get_entities_analyzed_by_method_config(ns, ws, method_config, days_back, etype, max_attempts) \
        if entity_statuses_under_the_methconf is None else entity_statuses_under_the_methconf

    return [e for e in entity_statuses
            if WorkflowExeStatus.FAIL == e.latest_status and e.fail_cnt >= count]


def _collect_entities_and_statuses(ns: str, ws: str, method_config: str, etype: str, relevant_submissions: List[dict],
                                   max_attempts: int = 2) \
        -> List[EntityStatuses]:
    """
    For given submissions of a method_config acting on a specific type of entities,
    collect the entities' analysis statuses.

    Basically, this is a parser utility for get_entities_in_a_submission():
    While get_entities_in_a_submission() groups entities in that particular submission by execution status,
    this helper gathers that information across different submissions, and organizes by entity.
    :param ns:
    :param ws:
    :param method_config:
    :param etype:
    :param relevant_submissions:
    :return:
    """
    entity_statuses: Dict[str, EntityStatuses] = dict()  # aux bookkeeping container

    def update(dd: Dict[WorkflowExeStatus, List[Tuple[str, datetime.datetime]]],
               st: WorkflowExeStatus):
        for ename, timing in dd[st]:
            if ename in entity_statuses:
                ess = entity_statuses[ename]
                ess.update_latest_status_and_timing(st, timing)
                entity_statuses[ename] = ess
            else:
                entity_statuses[ename] = \
                    EntityStatuses(st, timing, method_config, ename, etype)

    for sub in relevant_submissions:
        dd = get_entities_in_a_submission(ns, ws, sub['submissionId'], max_attempts)
        update(dd, WorkflowExeStatus.RUNN)
        update(dd, WorkflowExeStatus.ABORTING)
        update(dd, WorkflowExeStatus.SUCC)
        update(dd, WorkflowExeStatus.FAIL)
        update(dd, WorkflowExeStatus.ABORTED)

    return list( entity_statuses.values() )


def _analyzable_entities(ns: str, ws: str, method_config: str, etype: str, enames: List[str],
                         days_back: int or None, count: int or None, force: bool = False,
                         max_attempts: int = 2) -> List[str]:
    """
    Given a homogeneous (in terms of etype) list of entities, return a sub-list of them who are analyzable now.

    One can also specify, for entities that fail to be analyzed with the requested method_config repeatedly,
    whether to go ahead or not, as one may want to manually checkout what's wrong there.
    By not providing the two arguments, you are signaling this isn't necessary.
    Check get_repeatedly_failed_entities(...)

    Analyzable is defined as:
       * hasn't been analyzed by the method_config yet, or
       * isn't being analyzed actively by the method_config at the moment, or
       * has been analyzed, but no success yet, and optionally
       * has NOT been marked as repeated failure

    You may also force launch the job.
    In this case, we only filter away those entities that are
    being analyzed actively by the method_config at the moment.

    :param ns: namespace
    :param ws: workspace
    :param method_config: name of the method_config
    :param etype: entity type
    :param enames: list of entity names (assumed to have the same etype)
    :param days_back
    :param count
    :param force: if True, forcefully launch analysis on every entity provided except those being analyzed at the moment
                  by skipping the check if the entity has been successfully analyzed, or caused repeated failures
    :param max_attempts: for retrying when seeing connection reset by peer error
    :return: list of running jobs (as dict's) optionally filtered
    """

    """
    THIS IS OVERALL A FILTER OPERATION
    
        first,  get all the entities touched by that workflow, 
        second, take away that those that are currently being analyzed
        third,  if user requests force analysis, early return
        fourth, keep input entities that have not been touched by the workflow at all
        last,   take away those that are repeatedly failed to be analyzed by this
    
    It relays to functions for doing its job.
    
        get_entities_analyzed_by_method_config() -> [get_submissions_for_method_config(), 
                                                     _collect_entities_and_statuses()]
            return a list of EntityStatuses 
            (length equals the number of entities touched by the method_config)
            it simply relays to two other functions
            
                get_submissions_for_method_config()
                    leaf function:
                    returns a list of Json rendering of Terra responses,
                    length of list is the number of relevant submissions (not workflow)
            
                _collect_entities_and_statuses() -> get_entities_in_a_submission()
                    iterates through each submission (returned by get_submissions_for_method_config())
                    and for each submission, gather/update an entity's status
                    returns a list of EntityStatuses
                    (length equals the number of entities touched by the method_config)
                
                        get_entities_in_a_submission()
                            leaf function:
                            returns each entity's name, and the time its status was last updated
                            the entities are grouped by their respective workflow status
        
        get_repeatedly_failed_entities() -> get_entities_analyzed_by_method_config()
            return entities that **repeatedly** failed to be processed by a particular workflow, 
            up to a certain time-delta back
    """

    entity_statuses = get_entities_analyzed_by_method_config(ns, ws, method_config, PRACTICAL_DAYS_LOOKBACK, etype,
                                                             max_attempts)

    # get the entities that are being actively analyzed at the moment, and remove from pool
    running = {entity_stat.ename for entity_stat in entity_statuses
               if WorkflowExeStatus.RUNN == entity_stat.latest_status}
    candidates = set(enames) - running

    # if user explicitly want to force it (basically redo previously successful runs)
    if force:
        return list(candidates)

    # entities who haven't been touched at all by the method_config; note this filters away entities that were "success"
    entities_touched_by_methconf = set([entity_stat.ename for entity_stat in entity_statuses])
    fresh = candidates - entities_touched_by_methconf

    # entities whose latest status with the method_config is fail, i.e. should redo
    unsuccessful = {entity_stat.ename for entity_stat in entity_statuses
                    if WorkflowExeStatus.SUCC != entity_stat.latest_status}
    redo = candidates.intersection(unsuccessful)

    # but drop those that saw repeated failures
    if days_back is not None and count is not None:
        repeated_offenders = [e.ename for e in
                              get_repeatedly_failed_entities(ns, ws, method_config, etype, days_back, count,
                                                             max_attempts, entity_statuses)]
        redo = redo - set(repeated_offenders)

    # untouched + redo
    return list(fresh.union(redo))
