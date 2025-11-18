"""
Tests for the VaR calculation helpers defined in `examples.agent_2_tools_quant`.

These tests focus on deterministic inputs to make sure the statistical
calculations (parametric, historical and Monte Carlo VaR) return the expected
magnitudes. The functions under test return human-readable strings, so we parse
out the numeric parts we care about and compare them against pre-computed
reference values.
"""

import re

import pytest

from examples.agent_2_tools_quant import (
    calculer_var_historique,
    calculer_var_monte_carlo,
    calculer_var_parametrique,
)


def _extract_euro_value(text: str, label: str) -> float:
    """Pulls a `label: <value>€` float out of a formatted response string."""
    match = re.search(rf"{label}: ([\d,\.]+)€", text)
    if not match:
        raise AssertionError(f"Impossible de lire '{label}' dans:\n{text}")
    raw = match.group(1).replace(",", "")
    return float(raw)


def test_calculer_var_parametrique_matches_reference():
    positions = [1_000_000, 500_000, 300_000]
    volatilites = [0.18, 0.05, 0.25]
    correlations = [
        [1.0, 0.2, 0.4],
        [0.2, 1.0, -0.1],
        [0.4, -0.1, 1.0],
    ]

    result = calculer_var_parametrique(
        positions=positions,
        volatilites=volatilites,
        correlations=correlations,
        niveau_confiance=0.95,
        horizon_jours=1,
    )

    var_value = _extract_euro_value(result, "VaR absolue")
    assert var_value == pytest.approx(23_371.10, rel=1e-3)


def test_calculer_var_historique_matches_reference():
    rendements_historiques = [
        [0.01, -0.02, 0.015, 0.005],
        [0.005, -0.01, 0.008, -0.002],
        [-0.015, 0.012, -0.02, 0.01],
    ]
    positions = [1_000_000, 500_000, 300_000]

    result = calculer_var_historique(
        rendements_historiques=rendements_historiques,
        positions=positions,
        niveau_confiance=0.95,
        horizon_jours=1,
    )

    var_value = _extract_euro_value(result, "VaR absolue")
    assert var_value == pytest.approx(17_140.0, rel=1e-3)


def test_calculer_var_monte_carlo_matches_reference():
    positions = [1_000_000, 500_000, 300_000]
    volatilites = [0.18, 0.05, 0.25]
    correlations = [
        [1.0, 0.2, 0.4],
        [0.2, 1.0, -0.1],
        [0.4, -0.1, 1.0],
    ]
    rendements_moyens = [0.08, 0.03, 0.12]

    result = calculer_var_monte_carlo(
        positions=positions,
        volatilites=volatilites,
        correlations=correlations,
        rendements_moyens=rendements_moyens,
        niveau_confiance=0.95,
        horizon_jours=1,
        simulations=10_000,
    )

    var_value = _extract_euro_value(result, "VaR absolue")
    # Monte Carlo reste pseudo-aléatoire mais la fonction fixe np.random.seed(42)
    assert var_value == pytest.approx(22_880.06, rel=2e-3)

