import numpy as np
import torch

from dpr_rffi.model import (
    DPRConfig,
    DPRRFFI,
    correlation_calibrated_weight,
)
from dpr_rffi.perturbations import PerturbationSpec
from dpr_rffi.training import encoder_function


def _dataset(samples_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    class_zero = np.zeros((samples_per_class, 16, 2), dtype=np.float32)
    class_one = np.zeros((samples_per_class, 16, 2), dtype=np.float32)
    class_zero[:, :, 0] = 1.0
    class_one[:, :, 1] = 1.0
    class_zero += rng.normal(0.0, 0.01, size=class_zero.shape)
    class_one += rng.normal(0.0, 0.01, size=class_one.shape)
    return (
        np.concatenate([class_zero, class_one], axis=0),
        np.asarray([0] * samples_per_class + [1] * samples_per_class, dtype=np.int64),
    )


def _encode(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=1).astype(np.float32)


def _classify(x: np.ndarray) -> np.ndarray:
    return np.argmax(_encode(x), axis=1).astype(np.int64)


def test_correlation_calibrated_weight_clips_and_falls_back():
    rho, alpha = correlation_calibrated_weight(
        np.asarray([0.0, 1.0, 1.0, 2.0]),
        np.asarray([0.0, 1.0, 1.0, 2.0]),
    )
    assert rho == 1.0
    assert alpha == 1.0

    rho, alpha = correlation_calibrated_weight(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([3.0, 2.0, 1.0, 0.0]),
    )
    assert rho == -1.0
    assert alpha == 0.0

    rho, alpha = correlation_calibrated_weight(
        np.ones(4),
        np.asarray([0.0, 1.0, 2.0, 3.0]),
    )
    assert rho is None
    assert alpha == 0.5


def test_fit_normalizes_before_perturbation():
    train_x, train_y = _dataset(8)
    val_x, val_y = _dataset(4)
    train_x *= 3.0
    val_x *= 3.0
    classified_rms: list[np.ndarray] = []
    encoded_rms: list[np.ndarray] = []

    def encode(values: np.ndarray) -> np.ndarray:
        encoded_rms.append(_rms(values))
        return _encode(values)

    def classify(values: np.ndarray) -> np.ndarray:
        classified_rms.append(_rms(values))
        return _classify(values)

    specs = [
        PerturbationSpec("identity", "amplitude", 1, {"alpha": 1.0}),
        PerturbationSpec("double_gain", "amplitude", 2, {"alpha": 2.0}),
    ]
    DPRRFFI(
        DPRConfig(
            screening_samples_per_class=4,
            low_augmentations_per_sample=1,
            low_reference_limit_per_class=4,
            knn_k=3,
        )
    ).fit(
        source_train_x=train_x,
        source_train_y=train_y,
        source_val_x=val_x,
        source_val_y=val_y,
        encode=encode,
        predict_labels=classify,
        perturbation_specs=specs,
    )

    np.testing.assert_allclose(encoded_rms[0], 1.0, atol=1e-5)
    np.testing.assert_allclose(classified_rms[0], 1.0, atol=1e-5)
    np.testing.assert_allclose(classified_rms[2], 2.0, atol=1e-5)


def test_encoder_wrapper_can_preserve_perturbation_gain():
    class MeanEncoder(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            embedding = torch.mean(x, dim=1)
            return embedding, embedding

    values = np.zeros((2, 8, 2), dtype=np.float32)
    values[:, :, 0] = 2.0
    encode = encoder_function(
        MeanEncoder(),
        batch_size=2,
        device="cpu",
        normalize=False,
    )
    embedding = encode(values)
    np.testing.assert_allclose(embedding[:, 0], 2.0)


def test_end_to_end_fit_uses_balanced_references_and_global_cca():
    train_x, train_y = _dataset(12)
    val_x, val_y = _dataset(6)
    specs = [
        PerturbationSpec("identity", "phase", 1, {"theta": 0.0}),
        PerturbationSpec("half_turn", "phase", 2, {"theta": np.pi}),
    ]
    model = DPRRFFI(
        DPRConfig(
            screening_samples_per_class=6,
            low_augmentations_per_sample=1,
            low_reference_limit_per_class=4,
            knn_k=3,
        )
    ).fit(
        source_train_x=train_x,
        source_train_y=train_y,
        source_val_x=val_x,
        source_val_y=val_y,
        encode=_encode,
        predict_labels=_classify,
        perturbation_specs=specs,
    )
    prediction = model.predict(val_x)
    assert model.reference_summary["low_impact_settings"] == 1
    assert model.reference_summary["high_impact_settings"] == 1
    assert (
        model.reference_summary["low_clean_reference_features"]
        == model.reference_summary["low_impact_augmented_features"]
    )
    assert (
        model.reference_summary["high_reference_features"]
        == model.reference_summary["low_reference_features"]
    )
    assert 0.0 <= model.cca_alpha <= 1.0
    assert prediction.score.shape == (val_x.shape[0],)
    assert np.all(np.isfinite(prediction.score))
    assert prediction.cca_alpha == model.cca_alpha


def _rms(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.sqrt(np.mean(np.sum(array * array, axis=-1), axis=1))
