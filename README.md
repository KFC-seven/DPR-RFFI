# DPR-RFFI

Official implementation of **DPR-RFFI: Dual Perturbation References for Open-Set RF Fingerprint Identification under Domain Shift**.

DPR-RFFI identifies enrolled radio transmitters while rejecting signals from transmitters that are absent during source training. Training, perturbation screening, reference construction, score standardization, and threshold calibration use source data only.

## Method

The release uses the same names and definitions as the paper.

- **Dual Perturbation References (DPR)** evaluates 52 RF perturbation settings with the perturbation retention score

  \[
  \eta_p=\frac{A_p-1/K}{\max(A_0-1/K,\epsilon_A)}.
  \]

  Settings with \(\eta_p\geq0.90\) form the low-impact pool, settings with \(\eta_p<0.50\) form the high-impact pool, and the remaining settings are neutral. RMS normalization is applied before perturbation. The low-impact reference set uses equal clean and eligible augmented budgets for every enrolled transmitter, with augmented samples balanced across perturbation families. High-impact perturbations provide a classification-degrading contrast and are not treated as pseudo-unknown devices. Their reference set is balanced across perturbation families and matched to the realized size of the low-impact reference set.

  DPR computes the distance from a received feature to each global reference set and divides the low-impact distance by the high-impact distance. The global low-impact set retains a nonempty subset for every enrolled transmitter, while the score itself does not require a candidate-class prediction.

- **Concordance-Controlled Assessment (CCA)** computes the Spearman rank correlation between DPR and five-neighbor rejection scores on source validation data. The coefficient is the constrained projection of centered validation-rank vectors onto \([0,1]\), derived in Proposition 2 of the paper. CCA therefore balances DPR and nearest-neighbor evidence without target-domain data or manual weight tuning.

- Accepted samples are identified by five-neighbor cosine voting. The rejection threshold is calibrated to a source-validation false-rejection budget of 0.03.

This repository provides the paper-facing implementation.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[test,plot]"
pytest
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Data

The experiments use equalized IEEE 802.11 preamble I/Q samples from the public WiSig ManySig and ManyTx collections. Download WiSig from the dataset provider, convert it to the compact structure described in [docs/DATA.md](docs/DATA.md), and place the files at:

```text
data/ManySig.pkl
data/ManyTx.pkl
```

Local paths can also be supplied with `--data`; no repository file needs to be edited.

## Run one protocol

ManySig:

```bash
python scripts/run_protocol.py \
  --config configs/manysig.yaml \
  --data /path/to/ManySig.pkl \
  --protocol RX9-3_TX2-4 \
  --split 1 \
  --architecture tiny \
  --embedding-dim 64 \
  --epochs 100 \
  --max-samples-per-record 500 \
  --sample-mode random \
  --output outputs/manysig_rx9-3_tx2-4_split1.json
```

ManyTx:

```bash
python scripts/run_protocol.py \
  --config configs/manytx.yaml \
  --data /path/to/ManyTx.pkl \
  --protocol MTX_RX9-3_TX20-20 \
  --split 1 \
  --architecture resnet1d \
  --embedding-dim 128 \
  --epochs 100 \
  --max-samples-per-record 30 \
  --sample-mode head \
  --output outputs/manytx_rx9-3_tx20-20_split1.json
```

The output records the source-only calibration flag, source validation result, reference-set sizes, and all reported open-set metrics.

## Run the paper matrix

The paper contains four ManySig protocols and seven ManyTx protocols with three splits each, for 33 DPR-RFFI runs:

```bash
python scripts/run_paper_matrix.py \
  --manysig-data /path/to/ManySig.pkl \
  --manytx-data /path/to/ManyTx.pkl \
  --output-dir outputs/paper
```

## Baselines

The repository includes the implementations used for Energy, kNN, NNDR, OpenMax, OpenSVDD, HyperRSI, MeDAE, and OSSEI. The shared-score methods are in `dpr_rffi/baselines/posthoc.py`; the trainable RFFI baselines have separate modules under `dpr_rffi/baselines/`. [docs/BASELINES.md](docs/BASELINES.md) lists their public entry points and the source-validation calibration rule.

The four methods that share the source classifier can be run together with `scripts/run_shared_baselines.py`. OpenSVDD, HyperRSI, MeDAE, and OSSEI use their method-specific training entry points listed in the baseline guide.

All methods must use the same protocol split and calibrate their own rejection threshold from source validation scores. Target labels are evaluation-only.

## Repository layout

```text
DPR-RFFI/
├── dpr_rffi/
│   ├── model.py              # DPR reference construction, CCA, and inference
│   ├── screening.py          # Perturbation retention score and role assignment
│   ├── perturbations.py      # 52 paper perturbations
│   ├── training.py           # ManySig and ManyTx source encoders
│   ├── metrics.py            # H-score, OSCR, AUROC, rejection, ACC, and FRR
│   ├── data/                 # WiSig loading and deterministic protocol splits
│   └── baselines/            # Paper baseline implementations
├── configs/                  # ManySig and ManyTx protocols
├── scripts/                  # Single-run and 33-run orchestration
├── tests/                    # Unit and end-to-end synthetic tests
├── pyproject.toml
└── LICENSE
```

## Reproducibility notes

- Every reference sampling operation is seeded.
- The low-impact reference set has equal clean and augmented budgets per enrolled transmitter. The clean budget is capped at 1,000 features per class, and the augmented budget is balanced across eligible perturbation families.
- The high-impact reference set is global, family-balanced, and no larger than the realized low-impact reference set.
- CCA is a single source-validation weight for each run, not a sample-specific target-domain estimate.
- If no setting satisfies the high-impact criterion, DPR reduces to the low-impact reference distance, as specified in the paper.
- Numerical floors are \(10^{-6}\).
- The code never reads target labels during fitting or threshold calibration.

## Citation

```bibtex
@article{zhong2026dprrffi,
  title   = {DPR-RFFI: Dual Perturbation References for Open-Set RF Fingerprint Identification under Domain Shift},
  author  = {Zhong, Yuan and Li, Dongming},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT. See [LICENSE](LICENSE).
