from __future__ import annotations

from loguru import logger

from rhis.exceptions import RhisEvolNotCalledError


def raise_if_no_rhis_df_exists(*, is_rhis_complete: bool) -> None:
    if not is_rhis_complete:
        msg = "Rhis.build_rhis_evol_df() should be run before plot."
        logger.error(msg)
        raise RhisEvolNotCalledError(msg)
