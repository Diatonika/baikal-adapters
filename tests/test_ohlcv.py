import datetime

from pathlib import Path

from polars import len as length

from baikal.adapters.binance import (
    BinanceAdapter,
    BinanceDataConfig,
    BinanceDataInterval,
    BinanceDataType,
    BinanceInstrumentType,
)


def test_monthly_ohlcv(datadir: Path) -> None:
    adapter = BinanceAdapter(datadir)

    ohlcv = adapter.load_ohlcv(
        BinanceDataConfig(
            BinanceDataType.OHLCV,
            BinanceInstrumentType.SPOT,
            BinanceDataInterval.ONE_MINUTE,
            "BTCUSDT",
        ),
        datetime.datetime(2020, 1, 30, tzinfo=datetime.UTC),
        datetime.datetime(2020, 3, 2, tzinfo=datetime.UTC),
    )

    assert ohlcv.null_count().sum_horizontal().item() == 9_270
    assert ohlcv.select(length()).item() == 46_080
    assert ohlcv.count().drop("date_time").n_unique() == 1
