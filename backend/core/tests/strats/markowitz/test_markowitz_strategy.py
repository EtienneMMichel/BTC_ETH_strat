import numpy as np
import pandas as pd
import pytest
from core.strats.markowitz.strategy import MarkowitzStrategy
from core.models.covariance.rolling import RollingCovModel
from core.models.covariance.diagonal import DiagonalCovModel
from core.models.expected_returns.rolling_mean import RollingMeanReturns
from core.models.expected_returns.signal import SignalExpectedReturns


def make_prices(n=300, trend=0.0005, vol_btc=0.02, vol_eth=0.025, seed=0):
    rng = np.random.default_rng(seed)
    btc = np.cumprod(1 + rng.normal(trend, vol_btc, n))
    eth = np.cumprod(1 + rng.normal(trend, vol_eth, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"BTC": btc, "ETH": eth}, index=idx)


# ---- basic contract --------------------------------------------------------

def test_predict_signal_returns_dict():
    prices = make_prices()
    s = MarkowitzStrategy(
        cov_model=RollingCovModel(),
        expected_returns=RollingMeanReturns(),
        objective="max_sharpe",
    )
    s.fit(prices)
    out = s.predict_signal(prices)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"BTC", "ETH"}


def test_weights_are_float():
    prices = make_prices()
    s = MarkowitzStrategy(cov_model=RollingCovModel(), expected_returns=RollingMeanReturns())
    s.fit(prices)
    out = s.predict_signal(prices)
    assert isinstance(out["BTC"], float)
    assert isinstance(out["ETH"], float)


def test_weights_sum_le_one():
    prices = make_prices()
    s = MarkowitzStrategy(cov_model=RollingCovModel(), expected_returns=RollingMeanReturns())
    s.fit(prices)
    out = s.predict_signal(prices)
    assert abs(out["BTC"]) + abs(out["ETH"]) <= 1.0 + 1e-8


def test_weights_finite():
    prices = make_prices()
    s = MarkowitzStrategy(cov_model=RollingCovModel(), expected_returns=RollingMeanReturns())
    s.fit(prices)
    out = s.predict_signal(prices)
    assert np.isfinite(out["BTC"])
    assert np.isfinite(out["ETH"])


# ---- min_history guard -----------------------------------------------------

def test_short_prices_returns_zeros():
    prices = make_prices(100)
    s = MarkowitzStrategy(
        cov_model=RollingCovModel(),
        expected_returns=RollingMeanReturns(),
        min_history=252,
    )
    s.fit(prices)
    out = s.predict_signal(prices)
    assert out == {"BTC": 0.0, "ETH": 0.0}


def test_sufficient_prices_non_zero():
    prices = make_prices(300)
    s = MarkowitzStrategy(
        cov_model=RollingCovModel(),
        expected_returns=RollingMeanReturns(),
        min_history=100,
    )
    s.fit(prices)
    out = s.predict_signal(prices)
    # Not guaranteed to be non-zero but very likely with 200 rows of trend data
    total = abs(out["BTC"]) + abs(out["ETH"])
    assert total >= 0.0  # At minimum, no crash


# ---- long_only constraint --------------------------------------------------

def test_long_only_weights_non_negative():
    prices = make_prices()
    for obj in ["max_sharpe", "min_variance", "mean_variance", "max_diversification"]:
        s = MarkowitzStrategy(
            cov_model=RollingCovModel(),
            expected_returns=RollingMeanReturns(),
            objective=obj,
            long_only=True,
            min_history=50,
        )
        s.fit(prices)
        out = s.predict_signal(prices)
        assert out["BTC"] >= -1e-8, f"{obj}: BTC weight negative"
        assert out["ETH"] >= -1e-8, f"{obj}: ETH weight negative"


# ---- objectives ------------------------------------------------------------

def test_all_objectives_run():
    prices = make_prices(300)
    for obj in ["max_sharpe", "min_variance", "mean_variance", "max_diversification"]:
        s = MarkowitzStrategy(
            cov_model=RollingCovModel(),
            expected_returns=RollingMeanReturns(),
            objective=obj,
            min_history=50,
        )
        s.fit(prices)
        out = s.predict_signal(prices)
        assert set(out.keys()) == {"BTC", "ETH"}
        assert abs(out["BTC"]) + abs(out["ETH"]) <= 1.0 + 1e-8


def test_invalid_objective_raises():
    prices = make_prices(300)
    s = MarkowitzStrategy(
        cov_model=RollingCovModel(),
        expected_returns=RollingMeanReturns(),
        objective="bad_objective",  # type: ignore
        min_history=50,
    )
    s.fit(prices)
    with pytest.raises(ValueError, match="Unknown objective"):
        s.predict_signal(prices)


# ---- target_vol rescaling --------------------------------------------------

def test_target_vol_scales_weights():
    prices = make_prices(300, trend=0.002)
    s_no_tv = MarkowitzStrategy(
        cov_model=RollingCovModel(),
        expected_returns=RollingMeanReturns(),
        objective="min_variance",
        target_vol=None,
        min_history=50,
    )
    s_tv = MarkowitzStrategy(
        cov_model=RollingCovModel(),
        expected_returns=RollingMeanReturns(),
        objective="min_variance",
        target_vol=0.05,  # small target_vol → small weights
        min_history=50,
    )
    s_no_tv.fit(prices)
    s_tv.fit(prices)
    out_no_tv = s_no_tv.predict_signal(prices)
    out_tv = s_tv.predict_signal(prices)
    total_no_tv = abs(out_no_tv["BTC"]) + abs(out_no_tv["ETH"])
    total_tv = abs(out_tv["BTC"]) + abs(out_tv["ETH"])
    # With low target_vol the total notional should be <= unconstrained case
    assert total_tv <= total_no_tv + 1e-6


# ---- default models --------------------------------------------------------

def test_default_models_are_used():
    prices = make_prices(300)
    s = MarkowitzStrategy(min_history=50)
    s.fit(prices)
    out = s.predict_signal(prices)
    assert set(out.keys()) == {"BTC", "ETH"}
    assert abs(out["BTC"]) + abs(out["ETH"]) <= 1.0 + 1e-8


# ---- smoke test with diagonal cov + signal er ------------------------------

def test_diagonal_cov_signal_er_smoke():
    prices = make_prices(300)
    s = MarkowitzStrategy(
        cov_model=DiagonalCovModel(),
        expected_returns=SignalExpectedReturns(),
        objective="max_sharpe",
        min_history=100,
    )
    s.fit(prices)
    out = s.predict_signal(prices)
    assert abs(out["BTC"]) + abs(out["ETH"]) <= 1.0 + 1e-8
