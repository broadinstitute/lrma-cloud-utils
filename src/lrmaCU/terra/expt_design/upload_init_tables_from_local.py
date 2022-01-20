import itertools
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.terra.expt_design import definitions
from src.terra.expt_design.definitions import TablesReadyForUpload, SequencingTechnology, Flowcell, Sample

SELECTION_CRITERION = ['tech', 'tissue', 'extraction']
SELECTION_CRITERIA = list(itertools.combinations(SELECTION_CRITERION, 2))


def make_tables_ready_for_upload(table_to_process: str, tech: str, column_name_conforming_to_naming_schema: str) -> \
        TablesReadyForUpload:
    """
    Given a Terra data table with a column that conforms to a naming schema,
    returns a data frame with annotated flowcells.
    :param table_to_process: path to Terra data table file
    :param tech: sequencing technology that generated data described in the table
    :param column_name_conforming_to_naming_schema: name of the column holding the description of the flowcell
           following the pre-defined-pattern
    :return: a tuple-3 of tables:
              1) the original table, with extra annotations added from annotate_nice_flowcells,
              2) mega_set table offering different set level views on the input Terra table
              3) a membership table describing the membership of the mega_set table
    """
    if tech not in SequencingTechnology.list():
        raise ValueError(f"Provided tech value isn't in the accepted list: {SequencingTechnology.list()}")

    ready_root_table, fc_uuid_2_fc_obj = \
        annotate_and_make_ready_root_table(table_to_process, tech, column_name_conforming_to_naming_schema)

    # then classify and mega-merge
    classified_tables, flowcell_mega_set_table = make_set_tables(ready_root_table, tech, fc_uuid_2_fc_obj)

    definitions.understand_vocabulary()
    participants = list()
    for p in flowcell_mega_set_table['participant'].unique():
        participants.append((p,
                             flowcell_mega_set_table[flowcell_mega_set_table['participant'] == p].iloc[:, 0].tolist())
                            )
    participants_table = pd.DataFrame.from_records(participants,
                                                   columns=['entity:participant_id', f'{tech.lower()}-samples'])

    return TablesReadyForUpload(ready_root_table, classified_tables, flowcell_mega_set_table, participants_table)


########################################################################################################################
def annotate_and_make_ready_root_table(table_to_process: str, tech: str, column_name_conforming_to_naming_schema: str) \
        -> (pd.DataFrame, Dict[str, Flowcell]):
    """
    Annotate root table and make it ready for uploading to Terra.
    :return: a tuple 2 of (formatted table, a dictionary of {flowcell_uuid: Flowcell_object})
    """

    if not Path(table_to_process).is_file():
        raise ValueError(f"Terra table TSV {table_to_process} doesn't seem to exist.")

    # load root level table to be annotated
    original_table = pd.read_csv(table_to_process, sep='\t', parse_dates=True)
    original_table['tech'] = pd.Series([tech] * len(original_table))

    # annotate root level table
    annotations, fc_uuid_2_fc_obj = annotate_nice_flowcells(original_table, column_name_conforming_to_naming_schema)

    # then merge
    original_table.drop(['tech'], axis=1, inplace=True)  # avoid tech_x, tech_y
    annotated_root_table = original_table.merge(annotations, how='outer', indicator=True,
                                                left_on=original_table.columns[0],
                                                right_on=annotations.columns[0])
    if any(annotated_root_table['_merge'] != 'both'):
        problematic_entities = list(annotated_root_table[annotated_root_table['_merge'] != 'both'].iloc[:, 0])
        raise ValueError(f"The annotation on nice flowcells don't fully agree with input flowcells: "
                         f"  {problematic_entities}")
    annotated_root_table.drop(['uuid', '_merge'], axis=1, inplace=True)

    # change the root entity level name to be tech specific
    annotated_root_table.rename({'entity:flowcell_id': f'entity:{tech.lower()}-flowcell_id'}, axis=1, inplace=True)

    return annotated_root_table, fc_uuid_2_fc_obj


def annotate_nice_flowcells(raw_data: pd.DataFrame, column_name_conforming_to_naming_schema: str) -> \
        (pd.DataFrame, Dict[str, Flowcell]):
    """
    Given a Terra data table with a column that conforms to a naming schema,
    returns a data frame with annotated flowcells.
    :param raw_data: loaded Terra data table
    :param column_name_conforming_to_naming_schema: name of the column holding the description of the flowcell
           following the pre-defined-pattern
    :return: a tuple-2 of :
              1) 5-col dataframe ['uuid', 'tech', 'tissue', 'extraction', 'participant'], where each row is one FC
              2) a dictionary from flowcell uuid to flowcell object
    """

    minimum_column_names = ['entity:flowcell_id', 'tech', column_name_conforming_to_naming_schema]
    just_enough_columns = raw_data[minimum_column_names]
    flowcells = just_enough_columns.apply(lambda row: Flowcell(row.iloc[0], row.iloc[1], row.iloc[2]),
                                          axis=1)

    annotated_flowcells = flowcells.apply(lambda fc:
                                          pd.Series([fc.uuid, fc.tech, fc.tissue, fc.extraction, fc.parti_uuid]))
    annotated_flowcells.columns = ['uuid', 'tech', 'tissue', 'extraction', 'participant']

    fc_uuid_2_fc_obj = dict(zip(just_enough_columns.iloc[:, 0], flowcells))
    return annotated_flowcells, fc_uuid_2_fc_obj


########################################################################################################################
def make_set_tables(annotated_root_table: pd.DataFrame, tech: str, fc_uuid_2_fc_obj: Dict[str, Flowcell]) -> \
        (Dict[str, pd.DataFrame], pd.DataFrame):
    """
    Given an annotated root level table,
    :return: a dict of tables where the key is the criteria (actually the value under this criteria), and
                                    the value is a 2-col table with (set_uuid, members, classifier name, participant id),
             a mega table where all tables in the dict are concatenated together
    """

    # name of the mega set table uuid column, and members column
    mega_set_entity_type_col_name = f'entity:{tech}-sample_id'
    member_type_col_name = re.sub('entity:', '', re.sub('_id$', '', annotated_root_table.columns[0])) + 's'

    # key: the value under classification criteria, value: the set level table
    classified_tables = dict()
    # all those tables in the dict above concatenated by row, with one extra column signalling generated by which criteria
    flowcell_mega_set_table = pd.DataFrame()

    # this only holds root entity ids as list
    entity_mega_sets = classify_root_entities(annotated_root_table, fc_uuid_2_fc_obj)
    for level in entity_mega_sets.keys():
        for classifier in entity_mega_sets[level].keys():
            for value in entity_mega_sets[level][classifier].keys():

                samples_df = __turn_samples_to_formatted_table(entity_mega_sets, level, classifier, value,
                                                               member_type_col_name)

                classified_tables[value] = samples_df

                old = samples_df.columns[0]
                new = f"entity:{tech}-mega-sample_id"
                new_rows = samples_df.rename({old: new}, axis=1)
                flowcell_mega_set_table = flowcell_mega_set_table.append(new_rows, ignore_index=True)

    return classified_tables, flowcell_mega_set_table


def classify_root_entities(annotated_root_table: pd.DataFrame, fc_uuid_2_fc_obj: Dict[str, Flowcell]) -> \
        Dict[str,
             Dict[str,
                  Dict[str, List[Sample]]]]:
    """
    Given a data frame that has annotation as provided by `annotate_nice_flowcells`,
    make a 3-level dict, where the keys for the 1st level are one of ('level-one', 'level-two').

    The inner-dict under 'level-one' uses different factors as their keys (e.g. tech, tissue, extraction),
    and underneath that, is another dict whose keys are the unique values of this factor,
    and values are corresponding samples.

    The 'level-two' dict is similarly defined, except flowcells are now classified using two criteria.
    :return:
    """

    set_dictionary = {'level_one': dict(),
                      'level_two': dict()}

    level_one_dict = dict.fromkeys(SELECTION_CRITERION)
    for c in SELECTION_CRITERION:
        level_one_dict[c] = dict()
        for u in annotated_root_table[c].unique():  # unique values under this selection criterion
            # by definition, 1 sample per participant for a specified criterion
            member_fc_uuids = annotated_root_table[annotated_root_table[c] == u].iloc[:, 0].tolist()
            classifier_names_and_values = [(c, u)]
            fc_objects = [fc_uuid_2_fc_obj.get(uuid) for uuid in member_fc_uuids]
            participant_2_fcs = dict()  # dict.fromkeys(set([fc.parti_uuid for fc in fc_objects]))  # group by participant
            for fc in fc_objects:
                participant_2_fcs.setdefault(fc.parti_uuid, []).append(fc)
            samples = [Sample(fcs, classifier_names_and_values) for p, fcs in participant_2_fcs.items()]
            level_one_dict[c][u] = samples
    set_dictionary['level_one'] = level_one_dict

    level_two_dict = dict()
    for c in SELECTION_CRITERIA:
        k1 = '_'.join(c)
        level_two_dict[k1] = dict()
        p1 = annotated_root_table[c[0]].unique()  # unique values under selection criterion 1
        p2 = annotated_root_table[c[1]].unique()  # unique values under selection criterion 2
        for comb in itertools.product(p1, p2):  # essentially a nested loop
            k2 = '_'.join(comb)
            a = annotated_root_table[c[0]] == comb[0]
            b = annotated_root_table[c[1]] == comb[1]
            member_fc_uuids = annotated_root_table[a & b].iloc[:, 0].tolist()
            classifier_names_and_values = [(c[0], comb[0]), (c[1], comb[1])]
            fc_objects = [fc_uuid_2_fc_obj.get(uuid) for uuid in member_fc_uuids]
            participant_2_fcs = dict()  # dict.fromkeys(set([fc.parti_uuid for fc in fc_objects]))  # group by participant
            for fc in fc_objects:
                participant_2_fcs.setdefault(fc.parti_uuid, []).append(fc)
            samples = [Sample(fcs, classifier_names_and_values) for p, fcs in participant_2_fcs.items()]
            level_two_dict[k1][k2] = samples
    set_dictionary['level_two'] = level_two_dict

    return set_dictionary


def __turn_samples_to_formatted_table(entity_mega_sets: Dict[str, Dict[str, Dict[str, List[Sample]]]],
                                      level: str, classifier: str, value: str,
                                      member_type_col_name: str) -> pd.DataFrame:

    """
    Given a particular triplet of (level, classifier, and value), gather the samples in entity_mega_sets
    :param entity_mega_sets:
    :param level:
    :param classifier:
    :param value:
    :param member_type_col_name:
    :return:
    """
    samples_in_this_table = entity_mega_sets[level][classifier][value]
    data = [(sm.uuid, sm.flowcells_uuid, classifier, sm.parti_uuid) for sm in samples_in_this_table]
    samples_table = pd.DataFrame.from_records(data,
                                              columns=[f"entity:{value}-sample_id", member_type_col_name,
                                                       'classifier', 'participant'])

    # participant_2_sm_list = dict()  # group by participant
    # for sample in entity_mega_sets[level][classifier][value]:
    #     participant = sample.parti_uuid
    #     participant_2_sm_list.setdefault(participant, []).append(sample.uuid)
    # df = pd.DataFrame.from_dict(participant_2_sm_list, orient='index')
    # matrix = df.values.tolist()
    # for i in range(len(matrix)):
    #     l = matrix[i]
    #     matrix[i] = [e for e in l if e is not None]
    # df['flowcells'] = matrix
    #
    # table_uuid_col_name = f"entity:{value}-sample_id"
    # formatted = {table_uuid_col_name: df.index.tolist(),
    #              member_type_col_name: df.loc[:, 'flowcells'].tolist()}
    #
    # return pd.DataFrame(formatted)
    return samples_table


########################################################################################################################
def construct_bare_family_table_from_individual_table(individual_fc_table: pd.DataFrame,
                                                      flowcell_id_col: str,
                                                      group_by: str,
                                                      family_id_col: str,
                                                      relation_col: str) -> pd.DataFrame:
    """
    Supporting a common usage scenario:
      Each member of families may have multiple flowcells, we want to group together the multiple FC's by individual,
      and still keeping the family and relation-within-family metadata.
    :param individual_fc_table: root level table, one row represents one flowcell
    :param flowcell_id_col: the column name that identifies the flowcell, i.e. the uuid column
    :param group_by: the column name that identifies which individual this flowcell belongs to
    :param family_id_col: the column name that identifies which family this individual belongs to
    :param relation_col: the column name that identifies the relation within the family of the individual
    :return:
    """
    multi_fc_individual_table = individual_fc_table\
        .groupby(group_by)\
        .agg({group_by: lambda x: x.tolist()[0],
              flowcell_id_col: lambda x: x.tolist(),
              family_id_col: lambda x: x.tolist()[0],
              relation_col: lambda x: x.tolist()[0]})\
        .reset_index(drop=True)

    return multi_fc_individual_table
