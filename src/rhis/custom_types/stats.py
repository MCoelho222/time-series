"""Typing classes for hypothesis tests."""
from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias

RhisCode: TypeAlias = Literal["r", "h", "i", "s"]
RhisStat: TypeAlias = Literal["R", "H", "I", "S", "min"]


class MannWhitneyResults(NamedTuple):
    statistic: float
    p_value: float
    reject: bool | None
    alternative: str


class WaldWolfowitzResults(NamedTuple):
    statistic: float
    p_value: float
    reject: bool | None


class RunsTestResults(NamedTuple):
    statistic: float
    p_value: float
    reject: bool | None
    alternative: str


class WallisMooreResults(NamedTuple):
    statistic: float
    p_value: float
    reject: bool | None
    alternative: str


class MannKendallResults(NamedTuple):
    statistic: float
    p_value: float
    reject: bool | None
    alternative: str


class TestDecisionNormal(NamedTuple):
    p_value: float
    alpha: float
    reject: bool | None
    alternative: str
