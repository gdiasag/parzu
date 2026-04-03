import os
from typing import final

from . import error


@final
class Config:
    def __init__(self) -> None:
        self.smor_model: str = require("PARZU_SMOR_MODEL")
        self.tagger_dir: str = require("PARZU_TAGGER_DIR")
        self.tmp_dir: str = os.path.abspath("/tmp")


def require(name: str) -> str:
    match os.getenv(name):
        case None:
            raise error.ConfigError(
                f"Missing required environment variable: {name}"
            )
        case s:
            return s
