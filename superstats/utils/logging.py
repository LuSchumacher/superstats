"""Small logging helpers for superstats."""

import logging as _logging

logger = _logging.getLogger("superstats")
_WARNED_MESSAGES: set[str] = set()


def _format_message(msg, *args, **kwargs) -> str:
    """Format messages with BayesFlow-style ``str.format`` arguments."""
    if args or kwargs:
        return str(msg).format(*args, **kwargs)
    return str(msg)


def info(msg, *args, **kwargs) -> None:
    logger.info(_format_message(msg, *args, **kwargs))


def warning(msg, *args, **kwargs) -> None:
    logger.warning(_format_message(msg, *args, **kwargs))


def error(msg, *args, **kwargs) -> None:
    logger.error(_format_message(msg, *args, **kwargs))


def warn_once(msg, *args, **kwargs) -> None:
    message = _format_message(msg, *args, **kwargs)
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    logger.warning(message)


__all__ = ["logger", "info", "warning", "error", "warn_once"]
