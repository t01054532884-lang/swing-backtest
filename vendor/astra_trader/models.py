"""원본 models.py에서 swing_strategy.py가 쓰는 D, StrictModel, Candle만 옮김."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def D(value) -> Decimal:
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Non-finite number")
    return number


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Candle(StrictModel):
    open_time: int = Field(ge=0)
    close_time: int = Field(ge=0)
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def valid_ohlc(self):
        if self.close_time <= self.open_time:
            raise ValueError("Invalid candle time")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("Invalid OHLC bounds")
        return self
