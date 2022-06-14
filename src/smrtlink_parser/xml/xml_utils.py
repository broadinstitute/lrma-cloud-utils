import google.cloud.storage.client
from google.cloud import storage
import xmltodict
import re
import tempfile


def parse_xml(xml_path: str, storate_client: google.cloud.storage.client.Client = None) -> dict:
    """
    Parse and return a dictionary from the provided XML file.
    :param xml_path: local path to the XML file
    :param storate_client: must be provided when the xml_path is a GCS path
    :return: a dictionary modeling the contents of the XML file
    """
    if xml_path.startswith('gs://'):
        assert storate_client is not None, \
            "When xml_path is a GCS path, must provide a valid storage client"

        arr = re.sub("^gs://", '', xml_path).split('/')
        bucket = arr[0]
        prefix = '/'.join(arr[1:-1])
        file = arr[-1]
        absolute_file_path = f'{prefix}/{file}' if prefix else file
        blob = storage.Blob(bucket=storate_client.bucket(bucket), name=absolute_file_path)

        with tempfile.NamedTemporaryFile() as fp:
            blob.download_to_filename(filename=fp.name)
            contents = xmltodict.parse(fp.read())
    else:
        with open(xml_path, 'r') as ff:
            contents = xmltodict.parse(ff.read())

    return contents


def get_value(raw_xml_dict: dict, keychain: list) -> str or dict:
    """
    Return the object obtained by querying the key from the nested dict.

    :param raw_xml_dict: the nested dict representation of the XML file
    :param keychain: chain of keys to request
    :return: either a value (string) or a dictionary (that can be further queried)
    """
    if keychain is None or 0 == len(keychain):
        raise ValueError("You gave me an empty key!")

    being_parsed = raw_xml_dict[keychain[0]]
    for key in keychain[1:]:
        being_parsed = being_parsed[key]

    return being_parsed
