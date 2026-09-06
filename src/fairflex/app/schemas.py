"""Pydantic contracts for the local FairFlex Demonstrator API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Reject fields the browser should never be able to influence."""

    model_config = ConfigDict(extra="forbid")


class RunCreateRequest(StrictModel):
    scenario_id: str = Field(min_length=1, max_length=80)
    include_v3_shadow: bool = False


class RunAccepted(BaseModel):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]


class ComparisonRequest(StrictModel):
    mode: Literal["v3_shadow"] = "v3_shadow"


class ApiError(BaseModel):
    code: str
    message: str
