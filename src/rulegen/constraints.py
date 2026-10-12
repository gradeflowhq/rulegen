from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
)

MAX_VALUES = 50
MAX_TEXT_LENGTH = 2000
MAX_SAFE_JSON_INTEGER = 2**53 - 1
MAX_ABS_FLOAT = 1e100

FiniteInteger = Annotated[StrictInt, Field(ge=-MAX_SAFE_JSON_INTEGER, le=MAX_SAFE_JSON_INTEGER)]


def _reject_integer_float_fallback(value: object) -> object:
    if isinstance(value, int) and not isinstance(value, bool):
        raise ValueError("integer values must use the JSON-safe integer branch")
    return value


FiniteFloat = Annotated[
    StrictFloat,
    Field(allow_inf_nan=False, ge=-MAX_ABS_FLOAT, le=MAX_ABS_FLOAT),
    BeforeValidator(_reject_integer_float_fallback),
]
FiniteNumber: TypeAlias = FiniteInteger | FiniteFloat
NonNegativeFiniteInteger = Annotated[StrictInt, Field(ge=0, le=MAX_SAFE_JSON_INTEGER)]
NonNegativeFiniteFloat = Annotated[
    StrictFloat,
    Field(allow_inf_nan=False, ge=0, le=MAX_ABS_FLOAT),
    BeforeValidator(_reject_integer_float_fallback),
]
NonNegativeFiniteNumber: TypeAlias = NonNegativeFiniteInteger | NonNegativeFiniteFloat
AnswerText = Annotated[StrictStr, Field(max_length=MAX_TEXT_LENGTH)]
BoundedString = Annotated[StrictStr, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]
LargeBoundedString = Annotated[StrictStr, Field(min_length=1, max_length=MAX_TEXT_LENGTH * 10)]
BoundedAnswerList = Annotated[list[AnswerText | FiniteNumber | None], Field(max_length=MAX_VALUES)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
