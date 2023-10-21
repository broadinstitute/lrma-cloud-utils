# This script does not contain any serious code, it's supposed to document concepts, examples
# that are beyond particular functions or modules
# These may be used in future manual writing.

WORKFLOW_CONFIG_EXAMPLE_JSON = """
Example config of a method
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


WORKFLOW_CONCEPT_EXPLAINER = """
A little explanation on terminology regarding several concepts:
[method, method_config, submission, workflow].

During casual discussions with contexts, for many, the word 'workflow' and 
a concept describing a set of WDLs organized around a main WDL callable 'workflow' unit 
(also commonly referred to as--though a bit ambiguously--a pipeline) 
are used interchangeably.

However, here, and on Terra, we want to be more precise and follow a naming convention.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%
A method is a particular WDL pipeline with a main `workflow` callable.
A WDL pipeline, when brought into a Terra workspace, can be duplicated to different names.
These duplicates all point to the same WDL pipeline (though possibly different versions of it)
can be configured to run on different root entity types
(if the WDL pipeline is designed with such flexibility), or 
configured only slightly differently for experimentation.
A method corresponds to one record on the 'WORKFLOWS' page on Terra.

%%%%%
A method_config is a method with a particular configuration.
The configuration of a method may be modified multiple times during its lifetime in a workspace.
Note that, when a method's config is changed, its name in submissions prior to the change 
is automatically updated by Terra by appending a random string.

%%%%%
A submission is a single record on the 'JOB HISTORY' page. 
A submission always has a unique submission ID in a Terra workspace.

%%%%%
A submission contains one or more workflows/executions for a particular method_config.
Hence here, workflow and execution are used interchangeably.
A workflow always has a unique workflow ID in a Terra workspace.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
THEREFORE, IN lrmaCUX, WE DO NOT USE workflow, method, AND method_config INTERCHANGEABLY. 
"""
