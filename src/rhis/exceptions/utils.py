from __future__ import annotations

from loguru import logger

from rhis.exceptions import RhisEvolNotCalledError


def raise_if_no_rhis_run(*, is_rhis_complete: bool) -> None:
    if not is_rhis_complete:
        msg = "Rhis.evol() should be run before adding rhis compliant data to the dataframe."
        logger.debug(msg)
        raise RhisEvolNotCalledError(msg)
