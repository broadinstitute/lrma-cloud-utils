import base64
import logging
import os
from typing import List

import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Attachment, FileContent, FileName, FileType, Disposition, ContentId)
from sendgrid.helpers.mail import From as SGFrom, To as SGTo, Subject as SGSubject
from sendgrid.helpers.mail import Mail, PlainTextContent, HtmlContent

########################################################################################################################
logger = logging.getLogger(__name__)


########################################################################################################################
def get_dict_depth(d: dict, level: int) -> int:
    """
    Simple utility (using recursion) to get the max depth of a dictionary

    :param d: dict to explore
    :param level: current level
    :return: max depth of the given dict
    """
    cls = [level]
    for _, v in d.items():
        if isinstance(v, dict):
            cls.append(get_dict_depth(v, 1 + level))

    return max(cls)


def is_contiguous(arr: List[int]) -> bool:
    """
    Check if a list of integers is contiguous
    :param arr: an array of integers
    :return:
    """
    arr_shift = arr[1:]
    arr_shift.append(arr[-1]+1)
    return all(1 == (a_i - b_i) for a_i, b_i in zip(arr_shift, arr))


def absolute_file_paths(directory):
    """https://stackoverflow.com/a/9816863"""
    for dir_path, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dir_path, f))


########################################################################################################################
def retry_fiss_api_call(func_name: str, max_attempts: int, *args, **kwargs) -> requests.Response:
    """
    Re-try calling FISS API function for ConnectionResetError.
    :param func_name: name of FISS API function
    :param max_attempts: max retries + 1
    :param kwargs: args to forward to the FISS API call
    :return: request response object
    """
    from socket import error as SocketError
    import errno
    from firecloud import api as fapi
    cnt = 0
    response = requests.Response()
    connection_reset = True
    while max_attempts != cnt and connection_reset:
        try:
            # call FISS API function by its name
            fiss_call = getattr(fapi, func_name)
            response = fiss_call(*args, **kwargs)
            if 200 == response.status_code:
                connection_reset = False
        except SocketError as e:  # but only retries for connect reset errors
            if e.errno != errno.ECONNRESET:
                logger.warning(f"Seeing error other than ConnectionRest during {cnt}-th attempt.")
                break
            else:
                connection_reset = True
                logger.warning(f"Seeing connection reset error for the {cnt}-th time.")
        except Exception as ee:  # exit for all other types of errors
            logger.error(f"Seeing error other than ConnectionRest during {cnt}-th attempt.")
            break

        cnt += 1

    return response


########################################################################################################################
def send_notification(notification_sender_name: str,
                      notification_receiver_names: List[str], notification_receiver_emails: List[str],
                      email_subject: str, email_body: str,
                      html_body: str = None) -> None:
    """
    Sending notification email to (potentially) multiple recipients.
    Note that this assumes two environment variables are set appropriately: "SENDGRID_API_KEY" & "SENDER_EMAIL".

    Provide html_body at your own risk.

    Shameless copy from
    https://github.com/sendgrid/sendgrid-python/blob/main/examples/helpers/mail_example.py#L9
    :return:
    """
    if len(notification_receiver_emails) != len(notification_receiver_names):
        raise ValueError("Different number of recipients and recipients' emails")

    sg = SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

    email_core = _construct_sendgrid_mail_core(notification_sender_name, email_subject, email_body, html_body)

    # send
    failed_responses = list()
    from copy import deepcopy
    for i in range(len(notification_receiver_emails)):
        message = deepcopy(email_core)
        message.add_to(SGTo(notification_receiver_emails[i], notification_receiver_names[i]))
        response = sg.client.mail.send.post(request_body=message.get())
        if 202 != response.status_code:
            failed_responses.append(i)
    if 0 < len(failed_responses):
        failures = '  \n'.join([notification_receiver_names[i]+':'+notification_receiver_emails[i]
                                for i in failed_responses])
        logger.warning(f"Failed to send message to some receivers: \n  {failures}")


def send_notification_with_attachments(notification_sender_name: str,
                                       notification_receiver_names: List[str], notification_receiver_emails: List[str],
                                       email_subject: str, email_body: str,
                                       html_body: str = None,
                                       txt_names_and_contents: list = None,
                                       tsv_names_and_dataframe: list = None,
                                       pdf_names_and_paths: list = None
                                       ) -> None:
    """
    Sending notification email to (potentially) multiple recipients.
    Note that this assumes two environment variables are set appropriately: "SENDGRID_API_KEY" & "SENDER_EMAIL".

    Provide html_body at your own risk.

    Shameless copy from
    https://github.com/sendgrid/sendgrid-python/blob/main/examples/helpers/mail_example.py#L9
    :return:
    """
    if len(notification_receiver_emails) != len(notification_receiver_names):
        raise ValueError("Different number of recipients and recipients' emails")

    sg = SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

    email_core = _construct_sendgrid_mail_core(notification_sender_name, email_subject, email_body, html_body)

    # attach
    attachments = _attach_files_to_mail(txt_names_and_contents, tsv_names_and_dataframe, pdf_names_and_paths)
    for a in attachments:
        email_core.add_attachment(a)

    # send
    failed_responses = list()
    from copy import deepcopy
    for i in range(len(notification_receiver_emails)):
        message = deepcopy(email_core)
        message.add_to(SGTo(notification_receiver_emails[i], notification_receiver_names[i]))
        response = sg.client.mail.send.post(request_body=message.get())
        if 202 != response.status_code:
            failed_responses.append(i)
    if 0 < len(failed_responses):
        failures = '  \n'.join([notification_receiver_names[i]+':'+notification_receiver_emails[i]
                                for i in failed_responses])
        logger.warning(f"Failed to send message to some receivers: \n  {failures}")


def _construct_sendgrid_mail_core(notification_sender_name: str,
                                  email_subject: str,
                                  email_body: str,
                                  html_body: str = None) -> Mail:

    """
    Construct core content of notification email to (potentially) multiple recipients.
    Note that this assumes two environment variables are set appropriately: "SENDGRID_API_KEY" & "SENDER_EMAIL".
    The returned Mail object DOES NOT specify recipients, caller should customize that.

    Provide html_body at your own risk.

    Shameless copy from
    https://github.com/sendgrid/sendgrid-python/blob/main/examples/helpers/mail_example.py#L9
    :return:
    """

    # arg validation
    assert "SENDGRID_API_KEY" in os.environ, \
        'environment variable SENDGRID_API_KEY is needed.'
    assert "SENDER_EMAIL" in os.environ, \
        'environment variable SENDER_EMAIL is needed.'

    # construct core
    notification_sender_email = os.environ.get('SENDER_EMAIL')
    email_core = Mail(from_email=SGFrom(notification_sender_email, notification_sender_name),
                      subject=SGSubject(email_subject),
                      plain_text_content=PlainTextContent(email_body),
                      html_content=HtmlContent(html_body) if html_body else None,
                      is_multiple=True)  # recipients won't see each other
    return email_core


def _attach_files_to_mail(txt_names_and_contents: list = None,
                          tsv_names_and_dataframe: list = None,
                          pdf_names_and_paths: list = None) -> list:
    """
    Return a list of SendGrid Attachments for attaching to the email core.

    :param txt_names_and_contents: list of tuple2 (file name, file content)
    :param tsv_names_and_dataframe: list of tuple2 (file name, file content)
    :param pdf_names_and_paths: list of tuple2 (file name, file content)
    :return:
    """

    has_something_to_attach = False
    for a in [txt_names_and_contents, tsv_names_and_dataframe, pdf_names_and_paths]:
        if a is not None and 0 != len(a):
            has_something_to_attach = True
            break

    assert has_something_to_attach, "No valid inputs for building attachments"

    attachments = list()

    # txt
    if txt_names_and_contents is not None:
        for attachment_txt_name, contents in txt_names_and_contents:
            base64_txt = \
                base64.b64encode(('\n'.join(contents)).encode('utf-8')).decode('utf-8')
            txt_attachment = Attachment(
                FileContent(base64_txt),
                FileName(attachment_txt_name),
                FileType('text/plain'),
                Disposition('attachment'),
                ContentId(attachment_txt_name)
            )
            attachments.append(txt_attachment)
    # tsv
    if tsv_names_and_dataframe is not None:
        for attachment_tsv_name, attachment_dataframe in tsv_names_and_dataframe:
            base64_csv = \
                base64.b64encode(attachment_dataframe.to_csv(header=True, index=False, sep='\t').encode()).decode()
            tsv_attachment = Attachment(
                FileContent(base64_csv),
                FileName(attachment_tsv_name),
                FileType('text/csv'),
                Disposition('attachment'),
                ContentId('dataframe')
            )
            attachments.append(tsv_attachment)
    # pdf
    if pdf_names_and_paths is not None:
        for attachment_pdf_name, attachment_pdf_path in pdf_names_and_paths:
            with open(attachment_pdf_path, 'rb') as f:
                data = f.read()
            pdf_content = base64.b64encode(data).decode()

            pdf_attachment = Attachment(
                FileContent(pdf_content),
                FileName(attachment_pdf_name),
                FileType('application/pdf'),
                Disposition('attachment')
            )
            attachments.append(pdf_attachment)

    return attachments
