import logging
import uuid

from superstats.utils.logging import error, info, warn_once, warning


def test_logging_helpers_format_and_emit(caplog):
    with caplog.at_level(logging.INFO, logger="superstats"):
        info("hello {}", "there")
        warning("careful {}", "now")
        error("bad {}", "thing")

    messages = [record.getMessage() for record in caplog.records if record.name == "superstats"]

    assert messages == ["hello there", "careful now", "bad thing"]


def test_warn_once_logs_only_first_occurrence(caplog):
    message = f"warn_once_{uuid.uuid4()}"

    with caplog.at_level(logging.WARNING, logger="superstats"):
        warn_once("{}", message)
        warn_once("{}", message)

    messages = [record.getMessage() for record in caplog.records if record.name == "superstats"]

    assert messages == [message]
