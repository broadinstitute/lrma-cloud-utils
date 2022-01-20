import datetime
import re
import urllib.request
from enum import Enum
from itertools import groupby

from dateutil import parser
from termcolor import colored

from ..utils import *

DEBUG = False


########################################################################################################################
def fetch_timing_html(cromwell_server: str, submission_id: str, local_html: str) -> None:
    """
    For fetching timing chart of a cromwell execution and saving that to a local HTML page

    :param cromwell_server: cromwell server address
    :param submission_id: hex-string uuid of the submission
    :param local_html: where to save locally
    """
    s = cromwell_server.rstrip('/')
    timing_url = f'{s}/api/workflows/v1/{submission_id}/timing'
    urllib.request.urlretrieve(timing_url, local_html)


########################################################################################################################
class CROMWELL_METADATA_MODEL_EXPLANATION:

    @staticmethod
    def print():
        general = f'Here we explain the assumptions and the model used in interpreting Cromwell Metadata Tree. ' \
                  f'' \
                  f'We assume there exists three types of computing nodes:\n' \
                  f'  1) simple tasks, which are computing units defined in WDL with keyword "task",' \
                  f' and maps to concrete computing instances;\n' \
                  f'  2) subworkflows (subWF), which are computing units defined in WDL with keyword "workflow",' \
                  f' and does not map to concrete computing instances, but delegates to other computing nodes;\n' \
                  f'  3) scatters, which are computing units signified in WDL with keyword "scatter",' \
                  f' and does not map to concrete computing instances, but have homogenous shards.' \
                  f' Each shard may compose of several computing nodes.'

        task_str = f"We assume a simple task's json is representable by list of a (relatively) simple dict. " \
                   f"The length of the list is the # of attempts made for preemptible tasks. " \
                   f"The json contains almost all the following keys:\n" \
                   f"{TaskMinimalDiagnosisMetadata.LEAF_TASK_META_KEYS}"

        subWF_str = f"We assume a subWF's json is representable by a length-1 list," \
                    f" where the only element is a dict holding the signatory key 'subWorkflowMetadata'. " \
                    f"Under 'subWorkflowMetadata' is a dict with two important keys: 'workflowName' and 'calls'. " \
                    f"'workflowName' holds the alias of the subWF, whereas the relayed-to computing nodes" \
                    f" are held under 'calls' as dictionaries. " \
                    f"Each dictionary is of length-1, where the key is the name of the computing node."

        scatter_str = f"We assume a scatter's json is represented by a list whose length is always > 1. " \
                      f"Here the situation can be divided into two sub-cases:\n" \
                      f"  A simple scatter's shards are simple tasks (or their multiple attempts).\n" \
                      f"  A complex scatter's shards are, more complicated." \
                      f" In this sub-case we assume each shard is represented by a list. In the list, possible " \
                      f"values are [simple task, subWF, nested scatter] (nested scatter isn't supported yet)." \
                      f" We refer to them as 'callables'. The length of the list for a shard, then, is the # of " \
                      f"these callables called in that shard."

        print(general)
        print()

        print(task_str)
        print()

        print(subWF_str)
        print()

        print(scatter_str)
        print()


class NonLeafNode(Enum):
    ATTEMPTS = 0         # attempts of a preemptible task
    SUBWORKFLOW = 1      # subworkflow
    SIMPLE_SCATTER = 2   # scatter where each shard is a simple task('s attempt)
    COMPLEX_SCATTER = 3  # scatter where each shard is a list of callables


class TaskMinimalDiagnosisMetadata:
    """
    Modeling a simple task('s attempt).
    If one wants to see more message about a task printed in a diagnosis routinely, this is the class to update.
    """

    PAPI_CODE_PATTERN = re.compile('PAPI error code [0-9]+')

    LEAF_TASK_META_KEYS = ['attempt',
                           'backend',
                           'backendLabels',
                           'backendLogs',
                           'backendStatus',
                           'callCaching',
                           'callRoot',
                           'commandLine',
                           'compressedDockerSize',
                           'dockerImageUsed',
                           'end',
                           'executionEvents',
                           'executionStatus',
                           'failures',  # may not always available
                           'inputs',
                           'jes',
                           'jobId',
                           'labels',
                           'outputs',
                           'preemptible',
                           'returnCode',  # may not always available
                           'runtimeAttributes',
                           'shardIndex',
                           'start',
                           'stderr',
                           'stdout'
                           ]

    def __init__(self, task_metadata: dict, task_default_name: str):

        if 'labels' in task_metadata:
            labels = task_metadata['labels']
            better = None
            if 'wdl-task-name' in labels:
                better = labels['wdl-task-name']
            if 'wdl-call-alias' in labels:
                better = labels['wdl-call-alias']
        else:
            better = None
        self.name = task_default_name if better is None else better

        if 'returnCode' in task_metadata:
            self.is_success = 0 == int(task_metadata.get('returnCode'))
        elif 'executionStatus' in task_metadata.keys():
            self.is_success = 'Done' == task_metadata.get('executionStatus')
        else:
            raise ValueError(
                f"{self.name}'s metadata has neither returnCode nor executionStatus:\n {task_metadata}")

        start = parser.isoparse(task_metadata['start'])
        end = parser.isoparse(task_metadata['end'])

        #####
        self.attempt = int(task_metadata['attempt'])
        self.shard_idx = int(task_metadata['shardIndex'])
        self.log = task_metadata['backendLogs']['log']
        self.timing = end - start

        self.failures = task_metadata['failures'] if 'failures' in task_metadata else dict()

    def __str__(self):
        return str(self.to_pprint())

    def to_pprint(self) -> dict:
        """
        For pretty print.
        """
        d = {'status': 'Success' if self.is_success else 'Fail',
             'shardIdx': self.shard_idx,
             'attempt': self.attempt,
             'log': self.log,
             'wallclock': _format_wallclock_timing_to_minutes(self.timing)}
        if 0 < len(self.failures):
            d['failures'] = self.failures

        return d


class WorkflowMinimumDiagnosisMetadata:

    def __init__(self, metadata: dict):

        # get the basics
        self.name = metadata['workflowName']
        self.uuid = metadata['id']

        self.exe_status = metadata['status']

        start = parser.isoparse(metadata['start'])
        end = parser.isoparse(metadata['end'])
        self.timing = end - start

        # parse the json tree
        self.tree = self.__resolve_non_scatter_children(metadata['calls'], 0, self.name)

    def topology(self):
        print(f"Workflow:   {self.name}\n")
        WorkflowMinimumDiagnosisMetadata.__split_leaf_and_branch(self.tree, level=0,
                                                                 parent_level_name=self.name,
                                                                 parent_node_type=None,
                                                                 show_topology=True,
                                                                 show_diagnosis=False,
                                                                 show_successes=False)

    def diagnose(self, show_success_too: bool = False):
        print(f"Workflow:   {self.name}\n"
              f"            (uuid: {self.uuid})\n"
              f"Status:     {self.exe_status}\n"
              f"Wall-Clock: {_format_wallclock_timing_to_minutes(self.timing)}\n\n")

        if self.exe_status.lower() == 'success' and not show_success_too:
            print("Workflow execution Succeeded. Nothing to show. Bye!")
            return

        print(colored('Diagnosis', 'blue', attrs=['bold']))
        print()

        WorkflowMinimumDiagnosisMetadata.__split_leaf_and_branch(self.tree, level=0,
                                                                 parent_level_name=self.name,
                                                                 parent_node_type=None,
                                                                 show_topology=False,
                                                                 show_diagnosis=True,
                                                                 show_successes=show_success_too)

    ####################################################################################################################
    # This is used when parsing the raw metadata JSON file.
    # Some of the logic are different from later when the resulting tree is parsed.
    ####################################################################################################################
    @staticmethod
    def __is_scatter(metadata: list) -> bool:
        return 1 < len(metadata)

    @staticmethod
    def __is_single_leaf(metadata: list) -> bool:
        """
        A single leaf is an attempt for a task.
        But remember a task may be a shard in a scatter, and/or attempted multiple times.
        Regardless, this maps to a single, concrete computing hardware.
        :param metadata:
        :return:
        """
        return (not WorkflowMinimumDiagnosisMetadata.__is_scatter(metadata)) \
               and ('subWorkflowMetadata' not in metadata[0])

    @staticmethod
    def __is_subworkflow(metadata: list) -> bool:
        return (not WorkflowMinimumDiagnosisMetadata.__is_scatter(metadata)) \
               and ('subWorkflowMetadata' in metadata[0])

    @staticmethod
    def __resolve_non_scatter_children(this_level_metadata: dict,
                                       parent_level: int,
                                       parent_name=None) -> list:
        info_for_this_node = list()
        current_level = 1 + parent_level

        pretty_parent_name = parent_name
        if 0 == parent_level and parent_name is None:
            pretty_parent_name = list(this_level_metadata.keys())[0].split(".")[0]

        for unit_name, metadata in this_level_metadata.items():  # metadata is a list
            single_word_name = unit_name.split('.')[-1]
            if WorkflowMinimumDiagnosisMetadata.__is_scatter(metadata):  # metadata length > 1
                pretty_name = pretty_parent_name + "." + single_word_name
                shards = WorkflowMinimumDiagnosisMetadata.__resolve_scatter_children(metadata, current_level,
                                                                                     pretty_name)
                info_for_this_node.append({single_word_name: shards})
            elif WorkflowMinimumDiagnosisMetadata.__is_subworkflow(metadata):
                pretty_name = pretty_parent_name + "." + single_word_name
                child_subworkflow_info = WorkflowMinimumDiagnosisMetadata. \
                    __resolve_non_scatter_children(metadata[0].get('subWorkflowMetadata').get('calls'),
                                                   current_level, pretty_name)
                info_for_this_node.append({single_word_name: child_subworkflow_info})
            elif WorkflowMinimumDiagnosisMetadata.__is_single_leaf(metadata):
                tmm = TaskMinimalDiagnosisMetadata(metadata[0], single_word_name)
                info_for_this_node.append(tmm)
            else:
                raise AssertionError("Model assumption is broken.")

        return info_for_this_node

    @staticmethod
    def __resolve_scatter_children(shards: list,
                                   parent_level: int,
                                   parent_name=None) -> list:

        info_for_this_node = list()
        current_level = 1 + parent_level

        pretty_parent_name = parent_name

        for s in shards:
            if isinstance(s, dict):
                idx = int(s['shardIndex'])
                if 'subWorkflowMetadata' in s:  # this shard is a subworkflow itself
                    n = s.get('subWorkflowMetadata').get('workflowName')
                    p = pretty_parent_name.split('.')[-1]
                    if n == p:
                        pretty_name = pretty_parent_name
                    else:
                        pretty_name = pretty_parent_name + '.' + s.get('subWorkflowMetadata').get('workflowName')
                    sub = s.get('subWorkflowMetadata').get('calls')  # sub is a dict
                    x = WorkflowMinimumDiagnosisMetadata.__resolve_non_scatter_children(sub, current_level, pretty_name)
                    info_for_this_node.append((x, idx))
                else:  # this shard is a simple task's attempts (one attempt is non-preemptible)
                    single_word_name = parent_name.split('.')[-1]
                    tmm = TaskMinimalDiagnosisMetadata(s, single_word_name)
                    info_for_this_node.append(tmm)
            elif isinstance(s, list):  # a double scatter?, haven't seen it yet, but will support once seen an example
                # will be something like this
                raise NotImplementedError(f"I haven't seen scatter into scatter yet, sorry!\n{parent_name}\n  {s}")
            else:
                raise AssertionError('Assumption that element in shards are always dict is broken.')

        return info_for_this_node

    ####################################################################################################################
    # This is used when parsing the tree.
    # The logic, particularly deciding if the type of a node in the tree, is different from above when parsing raw JSON.
    ####################################################################################################################
    @staticmethod
    def __split_leaf_and_branch(parsed_tree: list, level: int,
                                parent_level_name: str,
                                parent_node_type: NonLeafNode,
                                show_topology: bool,
                                show_diagnosis: bool,
                                show_successes: bool = False):
        """
        A recursion to treat
          *) leaf task/attempts (excluding shards),
          *) subworkflows, and
          *) shards
        separately at a given level.
        Note that for shards, we separate into two cases,
          *) simple shards where each shard contains one and only one simple task/attempt
          *) complex shards where each shard contains a list of callables, of which at least one is a subworkflow/scatter
        Depending on the requested usage, prints out different information.
        """

        # The general strategy is to print only
        #   *) leaf tasks/attempts (excluding shards)
        #   *) shards of simple scatters
        # and at the same time collect
        #   *) subworkflows
        #   *) complex scatters
        # for 1-level deeper treatment
        leaves_at_this_level = list()
        simple_scatters_at_this_level = list[dict]()  # list of len-1 dict's {scatter_name: [shards]}

        subworkflows_at_this_level = list[dict]()  # list of len-1 dict's {subWF name: [calls]}

        complex_scatters_at_this_level = list[dict]()  # list of len-1 dict's {scatter_name: [shards]}

        if parent_node_type == NonLeafNode.COMPLEX_SCATTER:
            # parsed_tree is a list of lists, outer length == # shards, inner length == # calls in a shard
            descriptor = f"Level-{level-1} complex scatter"
            leaf_type_node = "shard-attempts"
            leaf_units = parsed_tree
            if show_topology:
                WorkflowMinimumDiagnosisMetadata.__print_topology(level, descriptor, parent_level_name, leaf_type_node,
                                                                  leaf_units, simple_scatters_at_this_level,
                                                                  subworkflows_at_this_level,
                                                                  complex_scatters_at_this_level)
            for shard in parsed_tree:  # shard is a list
                idx = shard[1]
                calls = shard[0]
                WorkflowMinimumDiagnosisMetadata.__split_leaf_and_branch(calls, 1+level,
                                                                         parent_level_name + '.shard-' + str(idx),
                                                                         parent_node_type.SUBWORKFLOW,  # ? is this right?
                                                                         show_topology, show_diagnosis, show_successes)
        elif parent_node_type == NonLeafNode.SIMPLE_SCATTER:
            leaves_at_this_level = [node for node in parsed_tree]

            descriptor = f"Level-{level-1} simple scatter"
            leaf_type_node = "shard-attempts"
            leaf_units = leaves_at_this_level
        else:  # a subworkflow (root level is a subworkflow)

            for node in parsed_tree:
                if isinstance(node, TaskMinimalDiagnosisMetadata):
                    assert node.shard_idx < 0,\
                        f"TaskMinimalDiagnosisMetadata to be classified as leaves should have shard index < 0. {node}"
                    leaves_at_this_level.append(node)
                elif isinstance(node, dict):  # a subWF or scatter (simple or complex)
                    if 1 != len(node):
                        raise AssertionError("For a node represented with dict, I assume it's always length-1")
                    children = next(iter(node.values()))
                    if not (isinstance(children, list) and 1 < len(children)):
                        raise AssertionError(f"I'm assuming a non-leaf node to have multiple entries"
                                             f" (scatter, or subworkflow's children)."
                                             f"Seems not the case for {node}")

                    node_name = next(iter(node.keys()))  # node name could be a subworkflow name, or a scatter name
                    type_of_node = WorkflowMinimumDiagnosisMetadata.__classify_node(node_name, children)
                    if NonLeafNode.SIMPLE_SCATTER == type_of_node:
                        scatter_name = node_name
                        simple_scatters_at_this_level.append({scatter_name: children})
                    elif NonLeafNode.ATTEMPTS == type_of_node:
                        leaves_at_this_level.extend(children)
                    elif NonLeafNode.SUBWORKFLOW == type_of_node:
                        subworkflow_name = node_name
                        subworkflows_at_this_level.append({subworkflow_name: children})
                    else:
                        scatter_name = node_name
                        complex_scatters_at_this_level.append({scatter_name: children})
                else:
                    raise AssertionError(f"A [Leaf, dict] is expected, but I'm seeing \n{node}")

            descriptor = f"Level-{level-1} subworkflow"
            leaf_type_node = "leaves"
            leaf_units = leaves_at_this_level

        if 0 == level:
            descriptor = "Workflow"
            leaf_type_node = "leaves"
            leaf_units = leaves_at_this_level

        if show_topology and parent_node_type != NonLeafNode.COMPLEX_SCATTER:
            WorkflowMinimumDiagnosisMetadata.__print_topology(level, descriptor, parent_level_name, leaf_type_node,
                                                              leaf_units, simple_scatters_at_this_level,
                                                              subworkflows_at_this_level,
                                                              complex_scatters_at_this_level)

        if show_diagnosis:
            WorkflowMinimumDiagnosisMetadata.__diagnose_leafs_at_this_level(leaves_at_this_level,
                                                                            parent_node_type == NonLeafNode.SIMPLE_SCATTER,
                                                                            level, parent_level_name,
                                                                            show_successes)

        if 0 < len(simple_scatters_at_this_level):
            simple_scatters_at_this_level = sorted(simple_scatters_at_this_level, key=lambda e: next(iter(e.keys())))
            for sc in simple_scatters_at_this_level:
                n = next(iter(sc.keys()))
                shards = next(iter(sc.values()))
                WorkflowMinimumDiagnosisMetadata.__split_leaf_and_branch(shards, 1 + level,
                                                                         parent_level_name + '.' + n,
                                                                         NonLeafNode.SIMPLE_SCATTER,
                                                                         show_topology,
                                                                         show_diagnosis,
                                                                         show_successes)

        if 0 < len(subworkflows_at_this_level):
            subworkflows_at_this_level = sorted(subworkflows_at_this_level, key=lambda e: next(iter(e.keys())))
            for swf in subworkflows_at_this_level:
                n = next(iter(swf.keys()))
                children = next(iter(swf.values()))
                WorkflowMinimumDiagnosisMetadata.__split_leaf_and_branch(children, 1 + level,
                                                                         parent_level_name + '.' + n,
                                                                         NonLeafNode.SUBWORKFLOW,
                                                                         show_topology,
                                                                         show_diagnosis,
                                                                         show_successes)

        if 0 < len(complex_scatters_at_this_level):
            complex_scatters_sorted_by_name = sorted(complex_scatters_at_this_level, key=lambda d: next(iter(d.keys())))
            for complex_scatter in complex_scatters_sorted_by_name:
                scatter_name = next(iter(complex_scatter.keys()))
                shards = sorted(next(iter(complex_scatter.values())), key=lambda t: t[1])
                WorkflowMinimumDiagnosisMetadata.__split_leaf_and_branch(shards, 1+level, scatter_name,
                                                                         NonLeafNode.COMPLEX_SCATTER,
                                                                         show_topology, show_diagnosis, show_successes)

    @staticmethod
    def __classify_node(node_name: str, elements: list) -> NonLeafNode:
        # If elements are of different types, then a subWF for sure.
        if 1 < len(set(type(e) for e in elements)):  # children of different type
            return NonLeafNode.SUBWORKFLOW

        # below is when children are homogeneously typed
        if all(isinstance(e, TaskMinimalDiagnosisMetadata) for e in elements):  # all children are leafs
            shard_indexes = sorted(set(e.shard_idx for e in elements))
            attempts = sorted(set(e.attempt for e in elements))
            if 1 == len(shard_indexes) and 1 == len(attempts):
                return NonLeafNode.SUBWORKFLOW
            elif 1 < len(shard_indexes) and is_contiguous(shard_indexes):
                return NonLeafNode.SIMPLE_SCATTER
            elif 1 < len(attempts) and is_contiguous(attempts):
                return NonLeafNode.ATTEMPTS
            else:
                raise AssertionError(f"A list sharing the same shardIdx and the same attempts, haven't seen before:"
                                     f"\n{node_name}\n{elements}")

        if all(isinstance(e, tuple) for e in elements):
            assert is_contiguous(sorted(set(t[1] for t in elements))), \
                f"Assumption that when all elements are tuples, then it represents shards of a complex scatter" \
                f" is broken\n{node_name}\n{elements}"
            return NonLeafNode.COMPLEX_SCATTER

        if all(isinstance(e, dict) for e in elements):
            assert all(1 == len(e) for e in elements), \
                f"Assumption that when all children are dict's, all their length are the same--1," \
                f" is broken\n{node_name}\n{elements}"
            children_names = [next(iter(e.keys())) for e in elements]
            if 1 == len(set(children_names)):
                return NonLeafNode.COMPLEX_SCATTER
            else:
                return NonLeafNode.SUBWORKFLOW

    @staticmethod
    def __handle_a_complex_scatter(shards_of_the_complex_scatter: list, level: int, parent_scatter_name: str,
                                   show_topology: bool,
                                   show_diagnosis: bool,
                                   show_successes: bool = False):
        # shards_of_the_complex_scatter is a list of lists, outer length == # shards, inner length == # calls in a shard
        descriptor = f"Level-{level-1} complex scatter"
        leaf_type_node = "shard-attempts"
        leaf_units = shards_of_the_complex_scatter
        if show_topology:
            WorkflowMinimumDiagnosisMetadata.__print_topology(level, descriptor, parent_scatter_name, leaf_type_node,
                                                              leaf_units,  list(), list())
        for shard in shards_of_the_complex_scatter:  # shard is a list
            idx = shard[1]
            calls = shard[0]
            WorkflowMinimumDiagnosisMetadata.__split_leaf_and_branch(calls, 1+level,
                                                                     parent_scatter_name + '.shard-' + str(idx),
                                                                     NonLeafNode.SUBWORKFLOW,  # ? is this right?
                                                                     show_topology, show_diagnosis, show_successes)

    ############################################################
    @staticmethod
    def __print_topology(level, descriptor, parent_level_name, leaf_type_node, leaf_units,
                         simple_scatters_at_this_level, subworkflows_at_this_level, complex_scatters_at_this_level):
        print(f"{WorkflowMinimumDiagnosisMetadata.__compute_leading_whitespaces(level)}"
              f"{descriptor}: {parent_level_name}"
              f", {len(leaf_units)} {leaf_type_node}"
              f", {len(simple_scatters_at_this_level)} simple scatters"
              f", {len(subworkflows_at_this_level)} subworkflows"
              f", {len(complex_scatters_at_this_level)} complex scatters\n")

    ############################################################
    @staticmethod
    def __diagnose_leafs_at_this_level(leaves_at_this_level: list, leaves_are_shards: bool, level: int,
                                       name_prefix: str,
                                       show_successes: bool = False):

        leading_spaces_offset = WorkflowMinimumDiagnosisMetadata.__compute_leading_whitespaces(level) + ' ' * 2

        if leaves_are_shards:
            scatter_name = name_prefix + '.' + leaves_at_this_level[0].name
            shards = [shard.to_pprint() for shard in leaves_at_this_level]
            shards = sorted(shards, key=lambda d: d['shardIdx'])  # sort-then-group by shardIdx,
            failed_shards = set()
            accumulated_messages = list()
            for i, attempts in groupby(shards, lambda s: s['shardIdx']):  # each shard may be attempted multiple times
                attempts = sorted(attempts, key=lambda d: d['attempt'])
                is_success, message = WorkflowMinimumDiagnosisMetadata.__handle_attempts(list(attempts))
                accumulated_messages.append(f'shard {i} {message}')
                if not is_success:
                    failed_shards.add(i)
            separator = f'\n{leading_spaces_offset}  '
            error_messages = [accumulated_messages[i] for i in failed_shards]
            if 0 == len(failed_shards) and show_successes:
                formatted_all_message = separator.join(accumulated_messages)
                print(f"{leading_spaces_offset}{scatter_name} has {len(list(shards))} shards,\n"
                      f"{leading_spaces_offset}  {formatted_all_message}")
            elif 0 < len(failed_shards):
                scatter_name = colored(scatter_name, 'red')
                formatted_error_message = separator.join(error_messages)
                print(f"{leading_spaces_offset}{scatter_name} has {len(list(shards))} shards,\n"
                      f"{leading_spaces_offset}  {formatted_error_message}")
        else:
            transformed = [{leaf.name: leaf.to_pprint()} for leaf in leaves_at_this_level]
            transformed = sorted(transformed, key=lambda leaf: next(iter(leaf.keys())))  # sort-then-group by task name
            for task, task_specific_attempts in groupby(transformed, lambda leaf: next(iter(leaf.keys()))):
                name = name_prefix + '.' + task
                attempts = sorted([list(t.values())[0] for t in task_specific_attempts], key=lambda d: d['attempt'])
                is_success, message = WorkflowMinimumDiagnosisMetadata.__handle_attempts(attempts)
                if not is_success or show_successes:
                    name = colored(name, 'red')
                    print(f'{leading_spaces_offset}{name} is not sharded, '
                          f'{message}')

    @staticmethod
    def __handle_attempts(attempts: list) -> (bool, str):
        """
        :param attempts: list of TaskMinimalDiagnosisMetadata associated with each attempt for a task/shard
        :return: tuple of (success or not, message)
        """

        num_attempts = len(attempts)
        succeeded = 'Success' in [a['status'] for a in attempts]

        message = f'was attempted {num_attempts} times, ultimately {"succeeded" if succeeded else "failed"}.'

        if not succeeded:
            sorted_attempts = sorted(attempts, key=lambda k: k['attempt'])

            failure_msgs = [a['failure']['message'] for a in sorted_attempts if 'failure' in a]
            possible_failure_msgs = [TaskMinimalDiagnosisMetadata.PAPI_CODE_PATTERN.findall('PAPI error code [0-9]+', m)
                                     for m in failure_msgs]
            available_failure_msgs = [papi for papi in possible_failure_msgs if papi]
            message += f' PAPI codes for all attempts in order: {available_failure_msgs}.'

            last_attempt = sorted_attempts[-1]
            last_attempt_log = colored(last_attempt['log'], attrs=['underline'])
            message += f' Last attempt log file: {last_attempt_log}'

        return succeeded, message

    ############################################################
    @staticmethod
    def __compute_leading_whitespaces(level: int) -> str:
        return ' ' * 2 * level  # increase indent by 2 spaces, each level


def _format_wallclock_timing_to_minutes(timing: datetime.timedelta) -> str:
    """
    Simple utility to format the time spent on a computing unit, into hours and minutes.
    """
    return ':'.join(str(timing).split('.')[0].split(':')[0:-1])
