import logging

from geocalib.extractor import GeoCalib  # noqa

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

__module_name__ = __name__
