from attrs import define

from baikal.adapters.binance.enums import (
    BinanceDataInterval,
    BinanceDataType,
    BinanceInstrumentType,
)


@define
class BinanceDataConfig:
    data_type: BinanceDataType
    instrument_type: BinanceInstrumentType
    interval: BinanceDataInterval
    symbol: str
