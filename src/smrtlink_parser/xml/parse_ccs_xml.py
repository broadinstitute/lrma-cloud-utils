from typing import List
import logging
from .xml_utils import *


class CCS_XML:

    """
    A class to represent the content of a *.consensusreadset.xml.
    """

    external_resources_keychain = ['pbds:ConsensusReadSet', 'pbbase:ExternalResources', 'pbbase:ExternalResource']
    collections_metadata_keychain = ['pbds:ConsensusReadSet', 'pbds:DataSetMetadata',
                                     'Collections', 'CollectionMetadata']
    collections_metadata_keychain_alt_1 = ['pbds:ConsensusReadSet', 'pbds:DataSetMetadata',
                                     'pbmeta:Collections', 'pbmeta:CollectionMetadata']

    def __init__(self, xml_path: str, storate_client: google.cloud.storage.client.Client = None):
        """

        :param xml_path: local path or GS path to the XML file to be parsed
        :param storate_client: must be provided when the XML path is a GS path
        """

        self.contents = parse_xml(xml_path, storate_client)
        try:
            self.external_resources = self._get_external_resources()

            self.collection_metadata = self._get_collection_metadata()

            self.primary_information = self.get_primary_information()

            self.run_group_info = self.get_run_group_info()

            self.sample_meta = self.get_sample_meta()
        except KeyError as err:
            logging.exception(f"{xml_path} breaks assumption on keys.\n{str(err)}")
            raise

    ####################################################################################################################
    # get location of directory, BAM and companion files
    def _get_external_resources(self) -> dict:
        """
        For engineering and debugging purpose, few reasons to call directly.
        :return:
        """
        return get_value(self.contents, self.external_resources_keychain)

    def get_bam_file(self) -> str:
        return self.external_resources['@ResourceId']

    def get_pbi_file(self) -> str:
        return get_value(self.external_resources,
                         ['pbbase:FileIndices', 'pbbase:FileIndex', '@ResourceId'])

    def get_various_companion_files(self) -> List[str]:
        other_files_metadata = get_value(self.external_resources,
                                         ['pbbase:ExternalResources', 'pbbase:ExternalResource'])
        return [companion['@ResourceId'] for companion in other_files_metadata]

    ############################################################
    def get_primary_information(self) -> dict:
        """
        For engineering and debugging purpose, few reasons to call directly.
        :return:
        """
        return self.collection_metadata['pbmeta:Primary']

    def get_hosting_directory(self) -> str:
        return get_value(self.primary_information,
                         ['pbmeta:OutputOptions', 'pbmeta:CollectionPathUri'])

    def get_config_file_name(self) -> str:
        """
        Supposedly, this is the config used by SMRTLink to make a run.
        If so, super important to get it's contents correct.
        :return:
        """
        return self.primary_information['pbmeta:ConfigFileName']

    def get_ccs_options(self) -> dict:
        return self.primary_information['pbmeta:CCSOptions']

    ####################################################################################################################
    # get information particular this cell itself
    def _get_collection_metadata(self) -> dict:
        """
        For engineering and debugging purpose, few reasons to call directly.
        :return:
        """
        try:
            res = get_value(self.contents, self.collections_metadata_keychain)
        except KeyError:
            res = get_value(self.contents, self.collections_metadata_keychain_alt_1)
        return res

    def get_instrument_id(self) -> str:
        return self.collection_metadata['@InstrumentId']

    def get_instrument_name(self) -> str:
        return self.collection_metadata['@InstrumentName']

    def get_run_group_info(self) -> dict:
        """
        For engineering and debugging purpose, few reasons to call directly.
        :return:
        """
        return self.collection_metadata['pbmeta:RunDetails']

    def get_run_id(self) -> str:
        return self.run_group_info['pbmeta:TimeStampedName']

    def get_fct(self) -> str:
        """
        fct stands for flowcell-construction-ticket; maybe GC specific
        :return:
        """
        return self.run_group_info['pbmeta:Name']

    def get_creator(self) -> str:
        """
        Unclear how creator and operator are distinguished with different GCs.
        :return:
        """
        return self.run_group_info['pbmeta:CreatedBy']

    def get_operator(self) -> str:
        """
        Unclear how creator and operator are distinguished with different GCs.
        :return:
        """
        return self.run_group_info['pbmeta:StartedBy']

    def get_creation_time(self) -> str:
        """
        Unclear what creation time means EXACTLY.
        :return:
        """
        return self.run_group_info['pbmeta:WhenCreated']

    def get_starting_time(self) -> str:
        """
        Unclear what starting time means EXACTLY.
        :return:
        """
        return self.run_group_info['pbmeta:WhenStarted']

    def get_movie_name(self) -> str:
        return self.collection_metadata['@Context']

    def get_smrtcell_id(self) -> str:
        return get_value(self.collection_metadata, ['pbmeta:CellPac', '@Barcode'])

    def get_application(self) -> str:
        return self.sample_meta['pbmeta:Application']

    def get_well_name(self) -> str:
        return self.sample_meta['pbmeta:WellName']

    def get_insert_size(self) -> int:
        return int(self.sample_meta['pbmeta:InsertSize'])

    def get_concentration(self) -> float:
        """
        Unclear about the difference between concentration and on plate loading concentration.
        :return:
        """
        return float(self.sample_meta['pbmeta:Concentration'])

    def get_on_plate_loading_concentration(self) -> float:
        """
        Unclear about the difference between concentration and on plate loading concentration.
        :return:
        """
        return float(self.sample_meta['pbmeta:OnPlateLoadingConcentration'])

    def is_isoseq(self) -> bool:
        return self.sample_meta['pbmeta:IsoSeq'].lower() == 'true'

    def get_num_records(self) -> int:
        return int(get_value(self.contents,
                             ['pbds:ConsensusReadSet', 'pbds:DataSetMetadata', 'pbds:NumRecords']))

    def get_total_bases(self) -> float:
        """
        Note return type is float, but really is a long.
        I'm confused by what python 3's int max value could be (could be 32 bit on some system), so am playing safe.
        :return:
        """
        return float(get_value(self.contents,
                               ['pbds:ConsensusReadSet', 'pbds:DataSetMetadata', 'pbds:TotalLength']))

    ####################################################################################################################
    # sample identity (multiple)
    def get_sample_meta(self) -> dict:
        """
        For engineering and debugging purpose, few reasons to call directly.
        :return:
        """
        return self.collection_metadata['pbmeta:WellSample']

    def get_collaborator_participant_id(self) -> str:
        return get_value(self.sample_meta, ['pbsample:BioSamples', 'pbsample:BioSample', '@Name'])

    def get_collaborator_sample_id(self) -> str:
        return self.sample_meta['@Description']

    def get_internal_sample_id(self) -> str:
        return self.sample_meta['@Name']

    ####################################################################################################################
    # metrics related
    def get_instrument_ctrl_version(self) -> str:
        return self.collection_metadata['pbmeta:InstCtrlVer']

    def get_signal_processor_version(self) -> str:
        return self.collection_metadata['pbmeta:SigProcVer']

    def get_smrtlink_components_versions(self) -> dict:
        """
        This returns a dictionary, as the importance of the versions of the various components is unclear.

        If we ever decide there are components worth picking out, it should be easy.
        :return:
        """
        return self.collection_metadata['pbmeta:ComponentVersions']

    def get_user_defined_fields(self) -> dict:
        """
        Unsure how each GC makes use of this user defined fields,
        and it's unclear from SMRTLink manual either how to specify them in the GUI.
        :return:
        """
        return self.collection_metadata['pbmeta:UserDefinedFields']

    def get_automation_name(self) -> str:
        """
        This might be just diffusion loading automation.
        But there could be more automation types.
        :return:
        """
        return get_value(self.collection_metadata,
                         ['pbmeta:Automation', '@Name'])

    def get_automation_parameters(self) -> dict:
        return get_value(self.collection_metadata,
                         ['pbmeta:Automation', 'pbbase:AutomationParameters', 'pbbase:AutomationParameter'])

    def get_binding_kit(self) -> dict:
        """
        Individual attributes to be cherry-picked.
        :return:
        """
        return self.collection_metadata['pbmeta:BindingKit']

    def get_control_kit(self) -> dict:
        """
        Individual attributes to be cherry-picked.
        :return:
        """
        return self.collection_metadata['pbmeta:ControlKit']

    def get_template_prep_kit(self) -> dict:
        """
        Individual attributes to be cherry-picked.
        :return:
        """
        return self.collection_metadata['pbmeta:TemplatePrepKit']

    def get_sequencing_kit_plate(self) -> dict:
        """
        Individual attributes to be cherry-picked.
        :return:
        """
        return self.collection_metadata['pbmeta:SequencingKitPlate']
