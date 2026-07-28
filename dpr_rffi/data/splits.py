from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .compact import load_compact_dataset, paired_count


def build_split_records(
    known_txs: list[str],
    unknown_txs: list[str],
    source_rxs: list[str],
    drift_rxs: list[str],
    source_date: str,
    day_shift_date: str,
) -> list[dict]:
    records: list[dict] = []
    specifications = [
        ("source_train", known_txs, source_rxs, source_date, "source", True, False),
        (
            "shifted_known_rx",
            known_txs,
            drift_rxs,
            source_date,
            "receiver_shift",
            True,
            True,
        ),
        (
            "shifted_known_day",
            known_txs,
            source_rxs,
            day_shift_date,
            "day_shift",
            True,
            True,
        ),
        (
            "unknown_source_rx",
            unknown_txs,
            source_rxs,
            source_date,
            "source_unknown",
            False,
            False,
        ),
        (
            "unknown_drift_rx",
            unknown_txs,
            drift_rxs,
            source_date,
            "receiver_shift_unknown",
            False,
            False,
        ),
        (
            "unknown_source_day",
            unknown_txs,
            source_rxs,
            day_shift_date,
            "day_shift_unknown",
            False,
            False,
        ),
    ]
    for (
        split_name,
        transmitters,
        receivers,
        date,
        domain_type,
        is_known,
        is_shifted_known,
    ) in specifications:
        records.extend(
            _records_for(
                split_name=split_name,
                txs=transmitters,
                rxs=receivers,
                date=date,
                known_txs=known_txs,
                domain_type=domain_type,
                is_known=is_known,
                is_shifted_known=is_shifted_known,
            )
        )
    return records


def build_manifest(config: dict) -> dict:
    dataset = load_compact_dataset(config["dataset"]["path"])
    source_date = config["dates"]["source"]
    day_shift_date = config["dates"]["day_shift"]
    required_equalized = list(config.get("required_equalized", dataset["equalized_list"]))
    tx_pool, rx_pool, matrix_summary = select_filtered_matrix(
        dataset=dataset,
        source_date=source_date,
        day_shift_date=day_shift_date,
        required_equalized=required_equalized,
        min_samples_per_triple=int(config["filter"]["min_samples_per_triple"]),
        tx_pool_size=int(config["filter"]["tx_pool_size"]),
        rx_pool_size=int(config["filter"]["rx_pool_size"]),
    )
    protocols = [
        _build_protocol_manifest(
            dataset=dataset,
            protocol_cfg=protocol_config,
            protocol_index=protocol_index,
            tx_pool=tx_pool,
            rx_pool=rx_pool,
            source_date=source_date,
            day_shift_date=day_shift_date,
            required_equalized=required_equalized,
            tx_split_repeats=int(config.get("tx_split_repeats", 1)),
        )
        for protocol_index, protocol_config in enumerate(config["protocols"])
    ]
    return {
        "dataset": {
            "name": config["dataset"]["name"],
            "path": str(Path(config["dataset"]["path"])),
            "tx_total": len(dataset["tx_list"]),
            "rx_total": len(dataset["rx_list"]),
            "dates": list(dataset["capture_date_list"]),
            "equalized": list(dataset["equalized_list"]),
        },
        "dates": {"source": source_date, "day_shift": day_shift_date},
        "required_equalized": required_equalized,
        "filter": dict(config["filter"]),
        "matrix_summary": matrix_summary,
        "tx_pool": tx_pool,
        "rx_pool": rx_pool,
        "protocols": protocols,
        "capabilities": _capabilities(protocols),
    }


def select_filtered_matrix(
    dataset: dict,
    source_date: str,
    day_shift_date: str,
    required_equalized: Iterable[int],
    min_samples_per_triple: int,
    tx_pool_size: int,
    rx_pool_size: int,
) -> tuple[list[str], list[str], dict]:
    transmitters = list(dataset["tx_list"])
    receivers = list(dataset["rx_list"])
    usable = np.zeros((len(transmitters), len(receivers)), dtype=bool)
    for transmitter_index, transmitter in enumerate(transmitters):
        for receiver_index, receiver in enumerate(receivers):
            usable[transmitter_index, receiver_index] = (
                paired_count(
                    dataset,
                    transmitter,
                    receiver,
                    source_date,
                    required_equalized,
                )
                >= min_samples_per_triple
                and paired_count(
                    dataset,
                    transmitter,
                    receiver,
                    day_shift_date,
                    required_equalized,
                )
                >= min_samples_per_triple
            )
    receiver_indices = _choose_rx_pool(usable, rx_pool_size)
    complete_transmitters = [
        index
        for index in range(len(transmitters))
        if bool(np.all(usable[index, receiver_indices]))
    ]
    if len(complete_transmitters) < tx_pool_size:
        raise ValueError(
            "Filtered matrix has only "
            f"{len(complete_transmitters)} complete transmitters for "
            f"{rx_pool_size} receivers; requested {tx_pool_size}."
        )
    selected_transmitters = complete_transmitters[:tx_pool_size]
    summary = {
        "usable_pairs": int(usable.sum()),
        "total_pairs": int(usable.size),
        "complete_tx_for_selected_rx": int(len(complete_transmitters)),
        "selected_tx_count": int(len(selected_transmitters)),
        "selected_rx_count": int(len(receiver_indices)),
    }
    return (
        [transmitters[index] for index in selected_transmitters],
        [receivers[index] for index in receiver_indices],
        summary,
    )


def _build_protocol_manifest(
    *,
    dataset: dict,
    protocol_cfg: dict,
    protocol_index: int,
    tx_pool: list[str],
    rx_pool: list[str],
    source_date: str,
    day_shift_date: str,
    required_equalized: list[int],
    tx_split_repeats: int,
) -> dict:
    source_count = int(protocol_cfg["source_rx_count"])
    drift_count = int(protocol_cfg["drift_rx_count"])
    if source_count + drift_count > len(rx_pool):
        raise ValueError(
            f"{protocol_cfg['name']} requests {source_count + drift_count} receivers "
            f"from a pool of {len(rx_pool)}."
        )
    receiver_seed = int(
        protocol_cfg.get(
            "receiver_seed",
            int(protocol_cfg.get("seed", 0)) + 7919 * protocol_index,
        )
    )
    rng = random.Random(receiver_seed)
    source_rxs = sorted(rng.sample(rx_pool, source_count), key=rx_pool.index)
    remaining = [receiver for receiver in rx_pool if receiver not in source_rxs]
    drift_rxs = (
        remaining
        if len(remaining) == drift_count
        else sorted(rng.sample(remaining, drift_count), key=rx_pool.index)
    )
    tx_splits = []
    for split_id in range(1, tx_split_repeats + 1):
        known_txs, unknown_txs = _draw_tx_split(
            tx_pool=tx_pool,
            known_count=int(protocol_cfg["known_tx_count"]),
            unknown_count=int(protocol_cfg["unknown_tx_count"]),
            seed=int(protocol_cfg.get("seed", 0)) + 104729 * split_id,
        )
        records = build_split_records(
            known_txs,
            unknown_txs,
            source_rxs,
            drift_rxs,
            source_date,
            day_shift_date,
        )
        tx_splits.append(
            {
                "split_id": split_id,
                "known_txs": known_txs,
                "unknown_txs": unknown_txs,
                "sample_counts": _sample_counts(
                    dataset,
                    records,
                    required_equalized,
                ),
            }
        )
    return {
        "name": protocol_cfg["name"],
        "source_rxs": source_rxs,
        "drift_rxs": drift_rxs,
        "source_rx_count": source_count,
        "drift_rx_count": drift_count,
        "known_tx_count": int(protocol_cfg["known_tx_count"]),
        "unknown_tx_count": int(protocol_cfg["unknown_tx_count"]),
        "tx_splits": tx_splits,
    }


def _records_for(
    *,
    split_name: str,
    txs: list[str],
    rxs: list[str],
    date: str,
    known_txs: list[str],
    domain_type: str,
    is_known: bool,
    is_shifted_known: bool,
) -> list[dict]:
    return [
        {
            "split_name": split_name,
            "true_tx": transmitter,
            "known_label": (
                known_txs.index(transmitter) if transmitter in known_txs else -1
            ),
            "rx_id": receiver,
            "day_id": date,
            "domain_type": domain_type,
            "is_known": is_known,
            "is_shifted_known": is_shifted_known,
        }
        for transmitter in txs
        for receiver in rxs
    ]


def _choose_rx_pool(usable: np.ndarray, pool_size: int) -> list[int]:
    if pool_size > usable.shape[1]:
        raise ValueError(
            f"rx_pool_size={pool_size} exceeds available receivers={usable.shape[1]}."
        )
    if math.comb(usable.shape[1], pool_size) <= 250_000:
        best_combination: tuple[int, ...] | None = None
        best_score: tuple | None = None
        for combination in itertools.combinations(range(usable.shape[1]), pool_size):
            score = (
                int(np.all(usable[:, combination], axis=1).sum()),
                int(usable[:, combination].sum()),
                tuple(-index for index in combination),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_combination = combination
        return list(best_combination or [])
    selected: list[int] = []
    remaining = set(range(usable.shape[1]))
    while len(selected) < pool_size:
        best_receiver = max(
            remaining,
            key=lambda receiver: (
                int(np.all(usable[:, selected + [receiver]], axis=1).sum()),
                int(usable[:, selected + [receiver]].sum()),
                -receiver,
            ),
        )
        selected.append(int(best_receiver))
        remaining.remove(best_receiver)
    return selected


def _draw_tx_split(
    *,
    tx_pool: list[str],
    known_count: int,
    unknown_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    if known_count + unknown_count > len(tx_pool):
        raise ValueError("Known and unknown transmitter counts exceed the pool size.")
    shuffled = list(tx_pool)
    random.Random(seed).shuffle(shuffled)
    order = {transmitter: index for index, transmitter in enumerate(tx_pool)}
    known = sorted(shuffled[:known_count], key=order.get)
    unknown = sorted(
        shuffled[known_count : known_count + unknown_count],
        key=order.get,
    )
    return known, unknown


def _sample_counts(
    dataset: dict,
    records: list[dict],
    required_equalized: list[int],
) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[row["split_name"]] += paired_count(
            dataset,
            row["true_tx"],
            row["rx_id"],
            row["day_id"],
            required_equalized,
        )
    return dict(sorted(counts.items()))


def _capabilities(protocols: list[dict]) -> dict:
    supports_shifted_osr = False
    supports_unknown = False
    for protocol in protocols:
        for split in protocol["tx_splits"]:
            counts = split["sample_counts"]
            shifted = counts.get("shifted_known_rx", 0) + counts.get(
                "shifted_known_day", 0
            )
            unknown = (
                counts.get("unknown_source_rx", 0)
                + counts.get("unknown_drift_rx", 0)
                + counts.get("unknown_source_day", 0)
            )
            supports_shifted_osr = supports_shifted_osr or (
                shifted > 0 and unknown > 0
            )
            supports_unknown = supports_unknown or unknown > 0
    return {
        "supports_h_a": bool(supports_shifted_osr),
        "supports_h_b_clean": bool(supports_unknown),
        "supports_h_b_contam": bool(supports_shifted_osr and supports_unknown),
    }
