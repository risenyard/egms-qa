from egms_encoder.data.tile_batch import pad_tile_batch
from egms_encoder.data.tile_store import TileStore, iter_tile_batches

__all__ = [
    "TileStore",
    "iter_tile_batches",
    "load_metadata",
    "pad_tile_batch",
    "parquet_columns",
    "summarize_dataset",
    "time_columns",
]


def load_metadata(*args, **kwargs):
    from egms_encoder.data.schema import load_metadata as _load_metadata

    return _load_metadata(*args, **kwargs)


def parquet_columns(*args, **kwargs):
    from egms_encoder.data.schema import parquet_columns as _parquet_columns

    return _parquet_columns(*args, **kwargs)


def summarize_dataset(*args, **kwargs):
    from egms_encoder.data.schema import summarize_dataset as _summarize_dataset

    return _summarize_dataset(*args, **kwargs)


def time_columns(*args, **kwargs):
    from egms_encoder.data.schema import time_columns as _time_columns

    return _time_columns(*args, **kwargs)
