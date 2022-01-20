import collections.abc
import copy
import datetime
import logging
from typing import List, Dict, Tuple

import pytz
from dateutil import parser
from firecloud import api as fapi
from firecloud.errors import FireCloudServerError

from ..table_utils import add_one_set

########################################################################################################################

logger = logging.getLogger(__name__)


local_tz = pytz.timezone('US/Eastern')

PRACTICAL_DAYS_LOOKBACK = 7  # made an implicit assumption: 7 days back is the max

"""
Example workflow config.
{'deleted': False,
 'inputs': {'Dummy.bai': 'this.aligned_bai', 'Dummy.bam': 'this.aligned_bam'},
 'methodConfigVersion': 3,
 'methodRepoMethod': {'methodUri': 'dockstore://github.com%2Fbroadinstitute%2Flong-read-pipelines%2FDummy/sh_dummy',
                      'sourceRepo': 'dockstore',
                      'methodPath': 'github.com/broadinstitute/long-read-pipelines/Dummy',
                      'methodVersion': 'sh_dummy'},
 'name': 'Dummy',
 'namespace': 'broad-firecloud-dsde-methods',
 'outputs': {},
 'prerequisites': {},
 'rootEntityType': 'clr-flowcell'}
"""


# POST-like ############################################################################################################
def change_workflow_config(ns: str, ws: str, workflow_name: str,
                           new_root_entity_type: str = None,
                           new_input_names_and_values: dict = None,
                           new_branch: str = None) -> dict:
    """
    Supporting common—but currently limited—scenarios where one wants to update a config of a workflow.

    Note that does NOT make any efforts in making sure the new configurations make sense.
    That is, one could potentially mismatch the workflow with wrong root entities,
    and/or providing non-existent branches.
    It's the user's responsibility to make sure the input values are correct.

    The old config is returned, in case this is a one-time change and one wants to immediately revert back
    once something is done with the new config (e.g. run an one-off analysis).
    If this indeed is the case, checkout restore_workflow_config(...) in this module.
    :param ns:
    :param ws:
    :param workflow_name:
    :param new_root_entity_type: when one wants to re-configure a workflow's root entity
    :param new_input_names_and_values: when one wants to re-configure some input values, and/or add new input values
    :param new_branch: when one wants to switch to a different branch, where supposedly the workflow is updated.
    :return: current config before the update
    """
    if new_root_entity_type is None \
            and new_input_names_and_values is None \
            and new_branch is None:
        raise ValueError(f"Requesting to change config of workflow: {workflow_name}, but not changing anything.")

    response = fapi.get_workspace_config(ns, ws, ns, workflow_name)
    if not response.ok:
        logger.error(f"Failed to retrieve current config for workflow {ns}/{ws}:{workflow_name}.")
        raise FireCloudServerError(response.status_code, response.text)
    current_config = copy.deepcopy(response.json())

    updated = copy.deepcopy(current_config)
    if new_root_entity_type is not None:
        updated = _update_config(updated, {'rootEntityType': new_root_entity_type})
    if new_input_names_and_values is not None:
        updated_inputs = copy.deepcopy(updated['inputs'])
        updated_inputs.update(new_input_names_and_values)
        updated = _update_config(updated, {'inputs': updated_inputs})
    if new_branch is not None:
        updated_wdl_version = copy.deepcopy(updated['methodRepoMethod'])
        updated_wdl_version['methodVersion'] = new_branch
        updated_wdl_version['methodUri'] = '/'.join(updated_wdl_version['methodUri'].split('/')[:-1]) + '/' + new_branch
        updated = _update_config(updated, {'methodRepoMethod': updated_wdl_version})
    updated['methodConfigVersion'] = updated['methodConfigVersion'] + 1  # don't forget this

    response = fapi.update_workspace_config(ns, ws, ns,
                                            configname=workflow_name, body=updated)
    if not response.ok:
        logger.error(f"Failed to update workflow config {ns}/{ws}:{workflow_name}.")
        raise FireCloudServerError(response.status_code, response.text)

    # validate, but unsure how reliable this is
    response = fapi.validate_config(ns, ws, ns, workflow_name)
    if not response.ok:
        logger.error(f"The config for the workflow {ns}/{ws}:{workflow_name} is updated to doesn't validate."
                     f" Manual intervention needed.")
        raise FireCloudServerError(response.status_code, response.text)

    return current_config


def restore_workflow_config(ns: str, ws: str, workflow_name: str, old_config: dict) -> None:
    """
    Restore a config of the workflow to an old value, presumably validated.

    :param ns:
    :param ws:
    :param workflow_name:
    :param old_config:
    :return:
    """

    to_upload = copy.deepcopy(old_config)
    response = fapi.get_workspace_config(ns, ws, ns, workflow_name)
    if not response.ok:
        logger.error(f"Failed to retrieve current config for workflow {ns}/{ws}:{workflow_name}.")
        raise FireCloudServerError(response.status_code, response.text)
    to_upload['methodConfigVersion'] = response.json()['methodConfigVersion'] + 1

    response = fapi.update_workspace_config(ns, ws, ns, configname=workflow_name, body=to_upload)
    if not response.ok:
        logger.error(f"Failed to restore workflow config {ns}/{ws}:{workflow_name}.")
        raise FireCloudServerError(response.status_code, response.text)
    response = fapi.validate_config(ns, ws, ns, workflow_name)
    if not response.ok:
        logger.error(f"The config for the workflow {ns}/{ws}:{workflow_name} is restored to doesn't validate."
                     f" Manual intervention needed.")
        raise FireCloudServerError(response.status_code, response.text)


def _update_config(old_config: dict, new_config: dict) -> dict:
    """
    DFS updating a (nested) dict, modeling the config of a workflow.

    :param old_config:
    :param new_config:
    :return:
    """
    for k, v in new_config.items():
        if isinstance(v, collections.abc.Mapping):
            old_config[k] = _update_config(old_config.get(k, {}), v)
        else:
            old_config[k] = v
    return old_config


def verify_before_submit(ns: str, ws: str, workflow_name: str, etype: str, enames: List[str], use_callcache: bool,
                         batch_type_name: str = None, expression: str = None,
                         days_back: int = None, count: int = None) -> None:
    """
    For a list of entities, conditionally submit a job: if the entity isn't being analyzed already.

    One can also specify, for entities that fail to be analyse with the requested workflow repeatedly,
    whether to go ahead or not, as one may want to manually checkout what's wrong there.
    By not providing the two arguments, you are signaling this isn't necessary.
    Check get_repeatedly_failed_entities(...)

    When there are multiple entities given in enames, one can specify an expression for batch submission.
    For example, say etype is 'sample', and enames are samples to be analysed with workflow BLAH.
    BLAH is configured in a way such that its root entity is 'sample', i.e. same as etype.
    In this case, "expression" can simply be "this.samples".
    This is intuitive if one has configured workflows on Terra whose root entity is "sample_set", but some inputs
    takes attributes of individual 'sample's.
    :param ns:
    :param ws:
    :param workflow_name:
    :param etype:
    :param enames:
    :param use_callcache:
    :param batch_type_name: type name of the resulting set, when batch submission mode is turned on
    :param expression: if not None, will submit all entities given in enames in one batch
                       Note that this will create a dummy set, for the purpose of batch submission.
    :param days_back: how many day back to check for repeated failures
    :param count: repeated failure threshold, >= which it won't be re-submitted.
    :return:
    """
    if 1 == len(enames) or expression is None:
        failures = dict()
        for e in _analyzable_entities(ns, ws, workflow_name, etype, enames, days_back, count):
            response = fapi.create_submission(wnamespace=ns, workspace=ws, cnamespace=ns,
                                              config=workflow_name,
                                              entity=e,
                                              etype=etype,
                                              use_callcache=use_callcache)
            if response.ok:
                logger.info(f"Submitted {etype} {e} submitted for analysis with {workflow_name}.")
            else:
                failures[e] = response.json()
                logger.warning(f"Failed to submit {etype} {e} for analysis with {workflow_name} due to"
                               f" \n {response.json()}")
        if failures:
            import pprint
            logger.error(f"Failed to submit jobs for the following entities:\n"
                         f"{pprint.pformat(failures)}")
            raise RuntimeError("Check above!!!")
    else:
        if batch_type_name is None:
            raise ValueError("When submitting in batching mode, batch_type_name must be specified")

        analyzable_entities = _analyzable_entities(ns, ws, workflow_name, etype, enames, days_back, count)
        if 0 == len(analyzable_entities):
            logger.warning(f"No analyzable entities in\n  {enames}")
            return

        now_str = datetime.datetime.now(tz=local_tz).strftime("%Y-%m-%dT%H-%M-%S")
        dummy_set_name_following_terra_convention = f'{workflow_name}_{now_str}_lrmaCU'
        add_one_set(ns, ws,
                    etype=batch_type_name,
                    ename=dummy_set_name_following_terra_convention,
                    member_type=etype,
                    members=analyzable_entities,
                    attributes=None)
        response = fapi.create_submission(ns, ws, cnamespace=ns, config=workflow_name,
                                          entity=dummy_set_name_following_terra_convention, etype=batch_type_name,
                                          expression=expression,
                                          use_callcache=use_callcache)
        if not response.ok:
            logger.error(f"Failed to submit batch job using batch {dummy_set_name_following_terra_convention}"
                         f" to workspace {ns}/{ws} with workflow {workflow_name}.")
            raise FireCloudServerError(response.status_code, response.text)
        logger.info(f"Submitted {etype}s {enames} for analysis with {workflow_name} in a batch.")


# GET-like #############################################################################################################
class EntityStatuses:

    """
    Modeling the number of times an entity has been successfully/unsuccessfully processed,
    the latest status, and the time of that.
    Note that this workflow-specific.
    """

    FAIL_STATUS = 'Failed'
    SUCC_STATUS = 'Succeeded'
    RUNN_STATUS = 'Running'

    def __init__(self, status: str, timing: datetime.datetime, workflow: str, ename: str, etype: str):
        self.ename = ename
        self.etype = etype
        self.workflow = workflow

        self.succ_cnt = 0
        self.fail_cnt = 0
        if EntityStatuses.SUCC_STATUS == status:
            self.bump_succ()
        elif EntityStatuses.FAIL_STATUS == status:
            self.bump_fail()

        self.latest_status = status
        self.latest_timing = timing

    def __str__(self):
        return f"{self.etype} {self.ename} has been analyzed with {self.workflow}: successfully {self.succ_cnt} times" \
               f", unsuccessfully {self.fail_cnt} times." \
               f" The latest status is {self.latest_status} at {self.latest_timing}."

    def bump_succ(self):
        self.succ_cnt += 1

    def bump_fail(self):
        self.fail_cnt += 1

    def update_latest_status_and_timing(self, status: str, timing: datetime.datetime):
        # status count need to be bumped regardless
        if EntityStatuses.SUCC_STATUS == status:
            self.bump_succ()
        elif EntityStatuses.FAIL_STATUS == status:
            self.bump_fail()
        if timing > self.latest_timing:  # latest status optional
            self.latest_status = status
            self.latest_timing = timing


# todo: this check is quite slow, anyway to speed it up?
def get_repeatedly_failed_entities(ns: str, ws: str, workflow: str, etype: str, days_back: int, count: int) \
        -> dict[str, int]:
    """
    Get entities that **repeatedly** failed to be processed by a particular workflow, up to a certain datetime back.

    :param ns:
    :param ws:
    :param workflow:
    :param etype:
    :param days_back:
    :param count: entities that failed to be processed, >= this number of times, will be reported
    :return: a dict {entity_name: failure_count}, within the days_back limit
    """

    entity_statuses = get_entities_analyzed_by_workflow(ns, ws, workflow, days_back, etype)

    res = dict[str, int]()
    for e, info in entity_statuses.items():
        if EntityStatuses.FAIL_STATUS == info.latest_status and info.fail_cnt >= count:
            res[e] = info.fail_cnt

    return res


def get_submissions_for_workflow(ns: str, ws: str, workflow: str, days_back: int) -> List[dict]:
    """
    Get submissions information for a particular workflow, up to a certain datetime back.

    Note that the level of details are not comparable to that retrieved with fapi.get_submission(...)
    :param ns:
    :param ws:
    :param workflow:
    :param days_back:
    :return:
    """
    response = fapi.list_submissions(ns, ws)
    if not response.ok:
        logger.error(f"Failed to list submissions in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    all_submissions = sorted(response.json(), key=lambda sub: parser.parse(sub['submissionDate']))
    cut_off_date = datetime.datetime.utcnow().astimezone(datetime.timezone.utc) - datetime.timedelta(days=days_back)
    drill_down = [sub for sub in all_submissions if
                  parser.parse(sub['submissionDate']) > cut_off_date
                  and sub['methodConfigurationName'] == workflow]
    return drill_down


def _collect_entities_and_statuses(ns: str, ws: str, workflow: str, etype: str, relevant_submissions: List[dict]) \
        -> Dict[str, EntityStatuses]:
    """
    For given submissions for a workflow acting on a specific type of entities, collect the entities' analysis statuses.

    :param ns:
    :param ws:
    :param workflow:
    :param etype:
    :param relevant_submissions:
    :return:
    """
    entity_status_and_timing = dict()
    for sub in relevant_submissions:
        succ = list()  # [(entity name, timing), ...]
        fail = list()
        running = list()
        if sub['submissionEntity']['entityType'].endswith('_batch'):
            s, f, r = get_entities_in_a_batch(ns, ws, sub['submissionId'])
            succ.extend(s)
            fail.extend(f)
            running.extend(r)
        else:
            e = sub['submissionEntity']['entityName']
            response = fapi.get_submission(ns, ws, sub['submissionId'])
            if not response.ok:
                logger.error(f"Failed to get submission {sub['submissionId']} in workspace {ns}/{ws}.")
                raise FireCloudServerError(response.status_code, response.text)
            detailed = response.json()
            timing = parser.parse(detailed['workflows'][0]['statusLastChangedDate'])
            if 'Succeeded' == detailed['workflows'][0]['status']:
                succ.append((e, timing))
            elif 'Failed' == detailed['workflows'][0]['status']:
                fail.append((e, timing))
            else:
                running.append((e, timing))

        for e, t in succ:
            if e in entity_status_and_timing:
                blah = entity_status_and_timing.get(e)
                blah.update_latest_status_and_timing(EntityStatuses.SUCC_STATUS, t)
                entity_status_and_timing[e] = blah
            else:
                entity_status_and_timing[e] = \
                    EntityStatuses(EntityStatuses.SUCC_STATUS, t, workflow, e, etype)
        for e, t in fail:
            if e in entity_status_and_timing:
                blah = entity_status_and_timing[e]
                blah.update_latest_status_and_timing(EntityStatuses.FAIL_STATUS, t)
                entity_status_and_timing[e] = blah
            else:
                entity_status_and_timing[e] = \
                    EntityStatuses(EntityStatuses.FAIL_STATUS, t, workflow, e, etype)
        for e, t in running:
            if e in entity_status_and_timing:
                blah = entity_status_and_timing[e]
                blah.update_latest_status_and_timing(EntityStatuses.RUNN_STATUS, t)
                entity_status_and_timing[e] = blah
            else:
                entity_status_and_timing[e] = \
                    EntityStatuses(EntityStatuses.RUNN_STATUS, t, workflow, e, etype)
    return entity_status_and_timing


def get_entities_in_a_batch(ns: str, ws: str, submission_id: str) -> \
        (List[Tuple[str, datetime.datetime]], List[Tuple[str, datetime.datetime]], List[Tuple[str, datetime.datetime]]):
    """
    Get (success, failed, running) entities in a batch submission, together with time when it's last updated.

    :param ns:
    :param ws:
    :param submission_id: id
    :return: Terra uuid for the (success, failed, running) entities in that batch submission,
             and time when it's last updated
    """
    response = fapi.get_submission(ns, ws, submission_id)
    if not response.ok:
        logger.error(f"Failed to get submission {submission_id} in workspace {ns}/{ws}.")
        raise FireCloudServerError(response.status_code, response.text)

    batch_submission_json = response.json()
    success = list()
    failure = list()
    running = list()
    for w in batch_submission_json['workflows']:
        e = w['workflowEntity']['entityName']
        t = parser.parse(w['statusLastChangedDate'])
        if 'Failed' == w['status']:
            failure.append((e, t))
        elif 'Succeeded' == w['status']:
            success.append((e, t))
        else:
            running.append((e, t))
    return success, failure, running


def get_entities_analyzed_by_workflow(ns: str, ws: str, workflow: str, days_back: int, etype: str) \
        -> Dict[str, EntityStatuses]:
    """
    Get entities of the requested type, that have been analyzed by a workflow.

    :param ns:
    :param ws:
    :param workflow:
    :param days_back:
    :param etype:
    :return:
    """

    relevant_submissions = get_submissions_for_workflow(ns, ws, workflow, days_back)
    return _collect_entities_and_statuses(ns, ws, workflow, etype, relevant_submissions)


def _analyzable_entities(ns: str, ws: str, workflow_name: str, etype: str, enames: List[str],
                         days_back: int or None, count: int or None) -> List[str]:
    """
    Given a homogeneous (in terms of etype) list of entities, return a sub-list of them who are analyzable now.

    One can also specify, for entities that fail to be analyse with the requested workflow repeatedly,
    whether to go ahead or not, as one may want to manually checkout what's wrong there.
    By not providing the two arguments, you are signaling this isn't necessary.
    Check get_repeatedly_failed_entities(...)

    Analyzable defined as:
       * isn't being analyzed, and
       * hasn't been analyzed, and
       * has been analyzed, but no success yet, and optionally
       * has been marked as repeated failure
    :param ns: namespace
    :param ws: workspace
    :param workflow_name: workflow name
    :param etype: entity type
    :param enames: list of entity names (assumed to have the same etype)
    :param days_back
    :param count
    :return: list of running jobs (as dict's) optionally filtered
    """
    entity_statuses = get_entities_analyzed_by_workflow(ns, ws, workflow_name, PRACTICAL_DAYS_LOOKBACK, etype)
    running = {e for e, statuses in entity_statuses.items() if statuses.latest_status == EntityStatuses.RUNN_STATUS}

    candidates = set(enames) - running

    fresh = candidates.difference(set(entity_statuses.keys()))

    failed = {e for e, statuses in entity_statuses.items() if statuses.latest_status == EntityStatuses.FAIL_STATUS}
    redo = candidates.intersection(failed)

    if days_back is not None and count is not None:
        waste = get_repeatedly_failed_entities(ns, ws, workflow_name, etype, days_back, count)
        redo = redo.difference(set(waste))

    return list(fresh.union(redo))
