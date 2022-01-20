import logging
import os
from typing import List

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, From, To, Subject, PlainTextContent, HtmlContent

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


def is_contiguous(arr: list) -> bool:
    arr_shift = arr[1:]
    arr_shift.append(arr[-1]+1)
    return all(1 == (a_i - b_i) for a_i, b_i in zip(arr_shift, arr))


def absolute_file_paths(directory):
    """https://stackoverflow.com/a/9816863"""
    for dir_path, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dir_path, f))


def send_notification(notification_sender_name: str,
                      notification_receiver_names: List[str], notification_receiver_emails: List[str],
                      email_subject: str, email_body: str,
                      html_body: str = None) -> None:
    """
    Sending notification email to (potentially) multiple recipients.

    Provide html_body at your own risk.

    Shameless copy from
    https://github.com/sendgrid/sendgrid-python/blob/main/examples/helpers/mail_example.py#L9
    :return:
    """

    assert "SENDGRID_API_KEY" in os.environ, \
        'environment variable SENDGRID_API_KEY is needed.'
    assert "SENDER_EMAIL" in os.environ, \
        'environment variable SENDER_EMAIL is needed.'

    if len(notification_receiver_emails) != len(notification_receiver_names):
        raise ValueError("Different number of recipients and recipients' emails")

    sg = SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    notification_sender_email = os.environ.get('SENDER_EMAIL')
    failed_responses = list()
    for i in range(len(notification_receiver_emails)):
        message = Mail(from_email=From(notification_sender_email, notification_sender_name),
                       to_emails=To(notification_receiver_emails[i], notification_receiver_names[i]),
                       subject=Subject(email_subject),
                       plain_text_content=PlainTextContent(email_body),
                       html_content=HtmlContent(html_body) if html_body else None)
        response = sg.client.mail.send.post(request_body=message.get())
        if 202 != response.status_code:
            failed_responses.append(i)
    if 0 < len(failed_responses):
        failures = '\n'.join([notification_receiver_names[i]+':'+notification_receiver_emails[i]
                              for i in failed_responses])
        logger.warning(f"Failed to send message to some receivers: {failures}")
