import logging
_LOGGER_NAME = "pyglauber"
_logger = logging.getLogger(_LOGGER_NAME)
if not _logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    _logger.addHandler(h)
    _logger.setLevel(logging.WARNING)
def set_log_level(level: str = "INFO"):
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))
def get_logger():
    return _logger
