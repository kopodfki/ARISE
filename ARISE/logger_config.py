import logging


class LevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler_debug = logging.StreamHandler()
console_handler_debug.setLevel(logging.DEBUG)
console_handler_debug.addFilter(LevelFilter(logging.DEBUG))

console_handler_info = logging.StreamHandler()
console_handler_info.setLevel(logging.INFO)
console_handler_info.addFilter(LevelFilter(logging.INFO))

console_handler_warn = logging.StreamHandler()
console_handler_warn.setLevel(logging.WARNING)
console_handler_warn.addFilter(LevelFilter(logging.WARNING))

console_handler_error = logging.StreamHandler()
console_handler_error.setLevel(logging.ERROR)
console_handler_error.addFilter(LevelFilter(logging.ERROR))

console_handler_fatal = logging.StreamHandler()
console_handler_fatal.setLevel(logging.FATAL)
console_handler_fatal.addFilter(LevelFilter(logging.FATAL))

formatter_debug = logging.Formatter(
    '\u001b[33;1m%(levelname)s:  %(message)s\033[0m')
formatter_info = logging.Formatter(
    '\u001b[36m%(levelname)s:  \033[0m%(message)s')
formatter_warn = logging.Formatter(
    '\u001b[33m%(levelname)s:  \033[0m%(message)s')
formatter_error = logging.Formatter(
    '\u001b[35m%(levelname)s:  \033[0m%(message)s')
formatter_fatal = logging.Formatter(
    '\u001b[31m%(levelname)s:  %(message)s\033[0m')

console_handler_debug.setFormatter(formatter_debug)
console_handler_info.setFormatter(formatter_info)
console_handler_warn.setFormatter(formatter_warn)
console_handler_error.setFormatter(formatter_error)
console_handler_fatal.setFormatter(formatter_fatal)

logger.addHandler(console_handler_debug)
logger.addHandler(console_handler_info)
logger.addHandler(console_handler_warn)
logger.addHandler(console_handler_error)
logger.addHandler(console_handler_fatal)


def close_logger():
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def set_debug_logging():
    """Set the logging level to DEBUG."""
    logger.setLevel(logging.DEBUG)
