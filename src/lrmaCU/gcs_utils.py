import logging
import re
import tempfile
from pathlib import Path
from typing import List

import pandas
from google.cloud import storage

logger = logging.getLogger(__name__)


########################################################################################################################
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket.

    Copied from
    https://cloud.google.com/storage/docs/uploading-objects#uploading-an-object
    :param bucket_name: your-bucket-name
    :param source_file_name: "local/path/to/file"
    :param destination_blob_name: "storage-object-name"
    :return:
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


class GcsPath:
    """
    Modeling after GCS storage object.

    Offering simplistic way of
        * checking if the paths exists, and if exists,
        * represents a file or
        * emulates a 'directory'.
        * if a file, getting it's size and/or downloading the file as blob
    """

    def __init__(self, gs_path: str):

        if not gs_path.startswith("gs://"):
            raise ValueError(f"Provided gs path isn't valid: {gs_path}")

        arr = re.sub("^gs://", '', gs_path).split('/')
        self.bucket = arr[0]
        self.prefix = '/'.join(arr[1:-1])
        self.file = arr[-1]

        # absolute path to file, sans bucket name, also guard against empty prefix
        self.absolute_file_path = f'{self.prefix}/{self.file}' if self.prefix else self.file

    def get_blob(self, client: storage.client.Client) -> storage.Blob:
        return storage.Blob(bucket=client.bucket(self.bucket), name=self.absolute_file_path)

    def download_and_parse_flat_file(self, client: storage.client.Client) -> List[str]:
        """
        Download the file as text and do minimal parsing as flat text file

        :param client:
        :return: a list, one entry per line
        """
        return [line for line in self.get_blob(client).download_as_text().split('\n') if line]

    def download_and_parse_csv(self, client: storage.client.Client, header: bool, sep: str) -> pandas.DataFrame:
        """
        Download the file and parse as a CSV file.

        :param client:
        :param header: if true, assumes the first line is the header
        :param sep: tab or comma
        :return:
        """
        all_lines = [line for line in self.get_blob(client).download_as_text().split('\n') if line]
        temp = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        with temp as temp_csv:
            for l in all_lines:
                temp_csv.write(f"{l}\n")
        df = pandas.read_csv(temp.name, sep=sep, header=0 if header else None)
        Path(temp.name).unlink()  # delete the temporary file
        return df

    def exists(self, client: storage.client.Client) -> bool:
        return self.is_file(client=client) or self.is_emulate_dir(client=client)

    def is_file(self, client: storage.client.Client) -> bool:
        return storage.Blob(bucket=client.bucket(self.bucket), name=self.absolute_file_path).exists(client)

    def is_emulate_dir(self, client: storage.client.Client) -> bool:
        if self.is_file(client=client):
            return False
        return any(True for _ in client.list_blobs(client.bucket(self.bucket), prefix=self.absolute_file_path))

    def size(self, client: storage.client.Client) -> int:
        blob = storage.Blob(bucket=client.bucket(self.bucket), name=self.absolute_file_path)
        if blob.exists(client=client):
            blob.reload()
            return blob.size
        else:
            return 0
