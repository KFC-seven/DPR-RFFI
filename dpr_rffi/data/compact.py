from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable


def load_compact_dataset(path: str | Path) -> dict:
    dataset_path = Path(path)
    if dataset_path.is_dir():
        dataset_path = dataset_path / f"{dataset_path.stem}.pkl"
    with dataset_path.open("rb") as handle:
        dataset = pickle.load(handle)
    _validate_compact_dataset(dataset, dataset_path)
    return dataset


def _validate_compact_dataset(dataset: dict, path: Path) -> None:
    required = {
        "tx_list",
        "rx_list",
        "capture_date_list",
        "equalized_list",
        "data",
    }
    missing = sorted(required.difference(dataset))
    if missing:
        raise ValueError(f"{path} is missing compact dataset keys: {missing}")


def get_leaf(dataset: dict, tx: str, rx: str, date: str, equalized: int):
    tx_index = dataset["tx_list"].index(tx)
    rx_index = dataset["rx_list"].index(rx)
    date_index = dataset["capture_date_list"].index(date)
    equalized_index = dataset["equalized_list"].index(equalized)
    return dataset["data"][tx_index][rx_index][date_index][equalized_index]


def paired_count(
    dataset: dict,
    tx: str,
    rx: str,
    date: str,
    required_equalized: Iterable[int],
) -> int:
    counts = [
        int(getattr(get_leaf(dataset, tx, rx, date, value), "shape", (0,))[0])
        for value in required_equalized
    ]
    return min(counts) if counts else 0
