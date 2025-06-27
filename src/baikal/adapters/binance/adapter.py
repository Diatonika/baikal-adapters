import datetime
import logging

from pathlib import Path
from zipfile import ZipFile

from pandera.typing.polars import DataFrame
from polars import (
    DataFrame as PolarDataFrame,
    Expr,
    Int16,
    Int64,
    LazyFrame as PolarLazyFrame,
    coalesce,
    col,
    concat,
    datetime_range,
    from_epoch,
    len as length,
    lit,
    read_csv,
)
from rich.progress import Progress

from baikal.adapters.binance._data_granularity import BinanceDataGranularity
from baikal.adapters.binance._ohlcv import BinanceOHLCV
from baikal.adapters.binance.config import BinanceDataConfig
from baikal.adapters.binance.enums import BinanceInstrumentType
from baikal.common.rich import RichConsoleStack, with_handler
from baikal.common.trade.models import OHLCV


class BinanceAdapter:
    LOGGER = logging.getLogger(__name__)

    def __init__(self, root: Path) -> None:
        if not root.exists():
            error = f"Binance data directory {root} not found."
            raise OSError(error)

        if not root.is_dir():
            error = f"Invalid Binance data directory {root} (not a directory)."
            raise OSError(error)

        self._root = root

    @with_handler(LOGGER)
    def load_ohlcv(
        self,
        config: BinanceDataConfig,
        start: datetime.datetime,
        end: datetime.datetime,
    ) -> DataFrame[OHLCV]:
        """Loads local **binance.vision** OHLCV data.

        Loads **binance.vision** OHLCV bulk data on left-closed
        interval [`start`, `end`) from local directory.

        Parameters
        ----------
        config: BinanceDataConfig
        start : datetime.datetime
        end : datetime.datetime

        Returns
        -------
        DataFrame[OHLCV]
            OHLCV time series data.

        Notes
        -----

        Returned OHLCV data time series is continuous on the whole requested half-interval.
        Trade data is not post-processed and may contain `null` sequences.

        Aggregates daily and monthly data if both present.

        Warnings
        --------

        In case of inconsistencies between daily and monthly data, warning is logged.
        """
        daily_data = self._load_ohlcv(config, start, end, BinanceDataGranularity.DAILY)
        monthly_data = self._load_ohlcv(
            config, start, end, BinanceDataGranularity.MONTHLY
        )

        filled_data = (
            PolarLazyFrame()
            .with_columns(
                date_time=datetime_range(start, end, config.interval, closed="left"),
                open=lit(None),
                high=lit(None),
                low=lit(None),
                close=lit(None),
                volume=lit(None),
            )
            .join(
                daily_data.lazy().select(OHLCV.column_names()),
                how="left",
                on="date_time",
                coalesce=False,
                maintain_order="left",
                suffix="_daily",
            )
            .join(
                monthly_data.lazy().select(OHLCV.column_names()),
                how="left",
                on="date_time",
                coalesce=False,
                maintain_order="left",
                suffix="_monthly",
            )
            .with_columns(
                open=coalesce("open_daily", "open_monthly"),
                high=coalesce("high_daily", "high_monthly"),
                low=coalesce("low_daily", "low_monthly"),
                close=coalesce("close_daily", "close_monthly"),
                volume=coalesce("volume_daily", "volume_monthly"),
            )
            .collect()
        )

        ambiguous_entries = filled_data.filter(
            (col("open_daily").ne_missing(col("open_monthly")))
            | (col("high_daily").ne_missing(col("high_monthly")))
            | (col("low_daily").ne_missing(col("low_monthly")))
            | (col("close_daily").ne_missing(col("close_monthly")))
            | (col("volume_daily").ne_missing(col("volume_monthly"))),
            col("date_time_daily").is_not_null(),
            col("date_time_monthly").is_not_null(),
        )

        ambiguous_entries_count: int = ambiguous_entries.select(length()).item()
        if ambiguous_entries_count:
            self.LOGGER.warning(
                f"{config.symbol}-{config.interval}: "
                f"found {ambiguous_entries_count} ambiguous entries\n"
                f"{ambiguous_entries}"
            )

        return DataFrame[OHLCV](
            filled_data.select(OHLCV.column_names()),
            OHLCV.column_names(),
        )

    def _load_ohlcv(
        self,
        config: BinanceDataConfig,
        start: datetime.datetime,
        end: datetime.datetime,
        granularity: BinanceDataGranularity,
    ) -> DataFrame[BinanceOHLCV]:
        chunks: list[PolarDataFrame] = []
        interval_seconds = (end - start).total_seconds()

        with Progress(
            console=RichConsoleStack.active_console(), transient=True
        ) as progress:
            task = progress.add_task(
                f"{granularity} {config.symbol}-{config.interval} OHLCV"
            )

            chunk_date_time = start
            while chunk_date_time < end:
                chunk = self._load_ohlcv_file(
                    config, chunk_date_time.date(), granularity
                )

                if chunk is not None:
                    chunks.append(chunk)

                chunk_date_time = granularity.next_chunk(chunk_date_time)

                loaded_seconds = (chunk_date_time - start).total_seconds()
                completed_percentage = (loaded_seconds / interval_seconds) * 100
                progress.update(task, completed=completed_percentage)

        if not len(chunks):
            return DataFrame[BinanceOHLCV]({}, BinanceOHLCV.column_names())

        data = concat(chunks, how="vertical", rechunk=False)
        return DataFrame[BinanceOHLCV](data, BinanceOHLCV.column_names())

    def _load_ohlcv_file(
        self,
        config: BinanceDataConfig,
        date: datetime.date,
        granularity: BinanceDataGranularity,
    ) -> DataFrame[BinanceOHLCV] | None:
        path = self._find_file_path(config, date, granularity)
        if path is None:
            return None

        raw_schema = BinanceOHLCV.polar_schema() | {
            "date_time": Int64,
            "close_date_time": Int64,
            "ignore": Int16,
        }

        raw_data = read_csv(
            ZipFile(path).read(path.with_suffix(".csv").name),
            has_header=False,
            new_columns=BinanceOHLCV.column_names(),
            schema=raw_schema,
        ).with_columns(
            date_time=self._from_unix(config, "date_time", date),
            close_date_time=self._from_unix(config, "close_date_time", date),
        )

        return DataFrame[BinanceOHLCV](
            raw_data.select(BinanceOHLCV.column_names()),
            BinanceOHLCV.column_names(),
        )

    def _find_file_path(
        self,
        config: BinanceDataConfig,
        date: datetime.date,
        granularity: BinanceDataGranularity,
    ) -> Path | None:
        file_name = (
            f"{config.symbol}-{config.interval}-{granularity.file_date(date)}.zip"
        )

        folder = (
            Path(config.instrument_type)
            / granularity
            / config.data_type
            / config.symbol
            / config.interval
        )

        path = self._root / folder / file_name
        return path if path.exists() and path.is_file() else None

    @staticmethod
    def _from_unix(
        config: BinanceDataConfig, unix_column: str, date: datetime.date
    ) -> Expr:
        if (
            config.instrument_type == BinanceInstrumentType.SPOT
            and date >= datetime.date(2025, 1, 1)
        ):
            return from_epoch(unix_column, "us")

        return from_epoch(unix_column, "ms")
