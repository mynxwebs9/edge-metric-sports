import logging

from nfl_predict.logging_conf import ContextFormatter, get_logger


def test_get_logger_returns_named_logger():
    logger = get_logger("nfl_predict.some.module")
    assert logger.name == "nfl_predict.some.module"


def test_context_formatter_renders_extra_fields():
    formatter = ContextFormatter(fmt="%(levelname)s %(name)s: %(message)s")
    record = logging.LogRecord(
        name="nfl_predict.data",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="ingested games",
        args=(),
        exc_info=None,
    )
    record.season = 2023
    record.count = 272

    rendered = formatter.format(record)

    assert "ingested games" in rendered
    assert "season=2023" in rendered
    assert "count=272" in rendered


def test_context_formatter_no_extra_fields_is_clean():
    formatter = ContextFormatter(fmt="%(levelname)s %(name)s: %(message)s")
    record = logging.LogRecord(
        name="nfl_predict.data",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="no extra context here",
        args=(),
        exc_info=None,
    )

    rendered = formatter.format(record)

    assert rendered == "WARNING nfl_predict.data: no extra context here"
