import pprint
import re
from enum import Enum
from typing import List, Dict, Tuple

import pandas as pd


# THIS SERVES AS TEMPLATE ON HOW TO DESCRIBE EXPERIMENT DESIGNS FOR MANY PROJECTS

class ExtractionProtocol(Enum):
    Autogen = 1
    Chemagen = 2

    @classmethod
    def list(cls):
        return [re.sub('ExtractionProtocol.', '', str(e)) for e in cls]


class TissueType(Enum):
    WBC = 1
    WholeBlood = 2
    Saliva = 3

    @classmethod
    def list(cls):
        return [re.sub('TissueType.', '', str(e)) for e in cls]


class SequencingTechnology(Enum):
    CCS = 1
    ONT = 2
    CLR = 3

    @classmethod
    def list(cls):
        return [re.sub('SequencingTechnology.', '', str(e)) for e in cls]


class Classifier(Enum):
    tech = 1
    tissue = 2
    extraction = 3

    @classmethod
    def list(cls):
        return [re.sub('Classifier.', '', str(e)) for e in cls]


########################################################################################################################
def understand_vocabulary() -> None:
    """
    Print to screen the vocabulary critical to understanding this class, and the set up.
    """
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint({"flowcell":    Flowcell.self_describe(),
               "participant": Participant.self_describe(),
               "sample":      Sample.self_describe()})


class Flowcell:

    """
    See: understand_vocabulary()
    """

    extract = f"{'|'.join(ExtractionProtocol.list())}"
    tissue = f"{'|'.join(TissueType.list())}"
    hud_alpha_participant_description_pattern = re.compile(f"(T[0-9]+)_([0-9]+_)*({extract})_({tissue})")
    fc_appendix_pattern = re.compile("(_)?[a-zA-Z]+$")

    @staticmethod
    def self_describe() -> str:
        return "the root level data"

    def __init__(self, flowcell_uuid: str, tech: str, flowcell_description: str,
                 extra_description: dict = None):
        self.parti_uuid, blah, self.extraction, self.tissue = \
            Flowcell.parse_flowcell_description(flowcell_description)
        self.uuid = flowcell_uuid
        self.flowcell_description = flowcell_description
        self.tech = tech

        # example: 6061-SL-0052_c and 6061-SL-0050b, they are top-offs, indicating
        # multiple FCs exist for the same participant's same tissue extracted with the same protocol
        self.sample_uuid = re.sub(Flowcell.fc_appendix_pattern, '', self.uuid)

        if extra_description:
            self.attributes = extra_description

    @classmethod
    def parse_flowcell_description(cls, participant_info: str) -> (str, str, str, str):
        """
        Note, this is purely based on experience from seeing the data. If that changes, then we'll be forced to adapt.
        Given participant description
        :return: tuple 4 (participant name, mystic str that looks like datetime, extraction method, tissue type)
        """
        parsed_result = re.search(cls.hud_alpha_participant_description_pattern, participant_info).groups()
        if parsed_result[2] not in ExtractionProtocol.__members__:
            raise ValueError(f"Provided participant {participant_info} doesn't seem to conform the expected format."
                             f" Specifically, its extraction method is not one of {ExtractionProtocol.list()}.")
        if parsed_result[3] not in TissueType.__members__:
            raise ValueError(f"Provided participant {participant_info} doesn't seem to conform the expected format."
                             f" Specifically, its tissue type is not one of {TissueType.list()}.")
        return parsed_result


class Sample:

    @staticmethod
    def self_describe() -> str:
        return (
            f"a set of flowcells from the same participant, where the DNA is extracted with\n"
            f"    * a specified protocol (one of {ExtractionProtocol.list()}), and/or from\n"
            f"    * a specified tissue type (one of {TissueType.list()}), and/or then sequenced with\n"
            f"    * a specified sequencing technology (one of {SequencingTechnology.list()});\n"
            f"not all three criteria need to be specified, but for specified criterion, "
            f"the flowcells in this sample must share the same value under that criterion"
        )

    def __init__(self, flowcells: List[Flowcell],
                 classifier_names_and_values: List[Tuple[str, str]]):
        for n, v in classifier_names_and_values:
            if n not in Classifier.list():
                raise KeyError(f"unsupported classifier name {n}. Accepted values are {Classifier.list()}.")

        Sample.__validate_flowcells(flowcells, classifier_names_and_values)

        self.flowcells_uuid = [fc.uuid for fc in flowcells]

        fc = flowcells[0]
        self.parti_uuid = fc.parti_uuid

        postfix = '_'.join([t[1] for t in classifier_names_and_values])
        self.uuid = f"{self.parti_uuid}_{postfix}"

    @classmethod
    def __validate_flowcells(cls, flowcells: List[Flowcell],
                             classifier_names_and_values: List[Tuple[str, str]]) -> None:
        tissues = set([fc.tissue for fc in flowcells])
        extractions = set([fc.extraction for fc in flowcells])
        techs = set([fc.tech for fc in flowcells])
        for t in classifier_names_and_values:
            c = t[0]
            if c == Classifier.tech.name.lstrip("Classifier."):
                if len(techs) != 1:
                    raise ValueError(f"Provided flowcells to construct sample don't share the same tech.")
            elif c == Classifier.tissue.name.lstrip("Classifier."):
                if len(tissues) != 1:
                    raise ValueError(f"Provided flowcells to construct sample don't share the same tissue.")
            elif c == Classifier.extraction.name.lstrip("Classifier."):
                if len(extractions) != 1:
                    raise ValueError(f"Provided flowcells to construct sample don't share the same extraction.")


class Participant:

    @staticmethod
    def self_describe() -> str:
        return "the actual individual donating the DNA"

    def __init__(self, samples: List[Sample]):
        Participant.__validate_samples(samples)

        self.samples_uuid = [sm.uuid for sm in samples]
        self.uuid = samples[0].parti_uuid

    @classmethod
    def __validate_samples(cls, samples: List[Sample]):
        saved = samples[0]
        for sm in samples:
            if not sm.parti_uuid == saved.parti_uuid:
                raise ValueError("Attempting to construct a participant with samples from different participant.")


########################################################################################################################
class TablesReadyForUpload:

    def __init__(self,
                 root_level_table: pd.DataFrame,
                 individual_set_tables: Dict[str, pd.DataFrame],
                 mega_set_table: pd.DataFrame,
                 participant_table: pd.DataFrame):
        # mega_set_table is essentially concatenated individual_set_tables
        self.root_level_table = root_level_table
        self.mega_set_table = mega_set_table
        self.individual_set_tables = individual_set_tables
        self.participant_table = participant_table
