import json
import runpy
from pathlib import Path


EXPECTED_MANYSIG_PROTOCOLS = [
    "RX9-3_TX2-4",
    "RX6-6_TX3-3",
    "RX6-6_TX2-4",
    "RX9-3_TX3-3",
]
EXPECTED_MANYSIG_RECEIVER_SEEDS = [12121, 20041, 35881, 43801]
EXPECTED_MANYTX_PROTOCOLS = [
    "MTX_RX9-3_TX20-20",
    "MTX_RX6-6_TX20-20",
    "MTX_RX3-9_TX20-80",
    "MTX_RX9-3_TX40-40",
    "MTX_RX6-6_TX40-40",
    "MTX_RX9-3_TX20-40",
    "MTX_RX6-6_TX20-40",
]


def test_data_package_exposes_paper_pipeline_entry_points():
    from dpr_rffi.data import (
        build_manifest,
        build_split_records,
        load_compact_dataset,
        load_config,
        materialize_records,
    )

    assert all(
        callable(value)
        for value in (
            build_manifest,
            build_split_records,
            load_compact_dataset,
            load_config,
            materialize_records,
        )
    )


def test_manysig_release_contract_matches_paper():
    config = json.loads(Path("configs/manysig.yaml").read_text(encoding="utf-8"))
    assert [row["name"] for row in config["protocols"]] == EXPECTED_MANYSIG_PROTOCOLS
    assert [row["receiver_seed"] for row in config["protocols"]] == (
        EXPECTED_MANYSIG_RECEIVER_SEEDS
    )

    matrix_script = runpy.run_path("scripts/run_paper_matrix.py")

    assert matrix_script["MANY_SIG"] == EXPECTED_MANYSIG_PROTOCOLS
    assert matrix_script["MANYSIG_RECORD_LIMIT"] == 500
    assert matrix_script["MANYSIG_SAMPLE_MODE"] == "random"
    assert matrix_script["MANY_TX"] == EXPECTED_MANYTX_PROTOCOLS
    assert matrix_script["MANYTX_RECORD_LIMIT"] == 30
    assert matrix_script["MANYTX_SAMPLE_MODE"] == "head"
