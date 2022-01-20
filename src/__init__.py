import logging
import logging.handlers
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_configured_logger(log_level: int = logging.DEBUG,
                          flush_level: int = logging.ERROR,
                          buffer_capacity: int = 0,
                          formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')):

    handler = __config_stream_or_file_handler(log_level, formatter)
    memory_handler = __config_memory_handler(buffer_capacity, flush_level, handler)

    custom_logger = logging.getLogger()
    custom_logger.setLevel(log_level)
    custom_logger.addHandler(memory_handler)
    return custom_logger


def __config_stream_or_file_handler(log_level: int, formatter: logging.Formatter, log_file: str = None):
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    return handler


def __config_memory_handler(capacity: int, flush_level: int, handler: logging.handlers):
    return logging.handlers.MemoryHandler(
        capacity=capacity,
        flushLevel=flush_level,
        target=handler
    )


########################################################################################################################
logger = get_configured_logger(log_level=logging.INFO)
