"""
Comprehensive tests for the ML forecasting and anomaly detection modules.

Tests cover:
- LightGBM forecaster training, prediction, and quantile intervals
- Isolation Forest anomaly detection and merge/deduplication
- Feature engineering (lag features, calendar features, make_feature_matrix)
- Seasonal naive baseline
- Edge cases (insufficient data, missing columns, untrained models)
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest


# ── Feature Engineering Tests ────────────────────────────────────────────

class TestFeatureEngineering(unittest.TestCase):
    """Tests for ml.features module."""

    def setUp(self):
        """Create a sample time-series DataFrame."""
        dates = pd.date_range("2024-01-01", periods=100, freq="5min")
        self.df = pd.DataFrame({
            "bucket": dates,
            "avg_val": np.random.uniform(5, 50, 100),
        })

    def test_build_calendar_features(self):
        from ml.features import build_calendar_features

        result = build_calendar_features(self.df, ts_col="bucket")

        # Should add calendar columns
        for col in ["hour", "dayofweek", "month", "is_weekend",
                     "hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            self.assertIn(col, result.columns, f"Missing column: {col}")

        # hour_sin and hour_cos should be in [-1, 1]
        self.assertTrue((result["hour_sin"].between(-1, 1)).all())
        self.assertTrue((result["hour_cos"].between(-1, 1)).all())

        # is_weekend should be 0 or 1
        self.assertTrue(result["is_weekend"].isin([0, 1]).all())

    def test_build_lag_features(self):
        from ml.features import build_lag_features

        result = build_lag_features(self.df, "avg_val", lags=[1, 2, 3])

        # Should create lag columns
        self.assertIn("lag_1", result.columns)
        self.assertIn("lag_2", result.columns)
        self.assertIn("lag_3", result.columns)

        # lag_1 should equal previous row's value
        self.assertEqual(result["lag_1"].iloc[1], self.df["avg_val"].iloc[0])

        # First row lag should be NaN
        self.assertTrue(np.isnan(result["lag_1"].iloc[0]))

    def test_build_lag_features_with_rolling(self):
        from ml.features import build_lag_features

        result = build_lag_features(
            self.df, "avg_val",
            lags=[1, 2],
            rolling_windows=[3, 6],
        )

        self.assertIn("roll_mean_3", result.columns)
        self.assertIn("roll_std_3", result.columns)
        self.assertIn("roll_mean_6", result.columns)
        self.assertIn("roll_std_6", result.columns)

    def test_make_feature_matrix(self):
        from ml.features import make_feature_matrix

        X, y = make_feature_matrix(self.df, "avg_val", ts_col="bucket")

        # X and y should have the same length
        self.assertEqual(len(X), len(y))

        # X should not contain the target or timestamp columns
        self.assertNotIn("avg_val", X.columns)
        self.assertNotIn("bucket", X.columns)

        # No NaNs after make_feature_matrix (dropna applied)
        self.assertFalse(X.isna().any().any())
        self.assertFalse(y.isna().any())

        # Should have rows (fewer than original due to lag NaN dropout)
        self.assertGreater(len(X), 0)
        self.assertLess(len(X), len(self.df))

    def test_make_feature_matrix_too_small(self):
        """Tiny dataset should still work (may produce 0 rows after dropna)."""
        from ml.features import make_feature_matrix

        small_df = pd.DataFrame({
            "bucket": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "avg_val": [1.0, 2.0, 3.0],
        })
        X, y = make_feature_matrix(small_df, "avg_val", ts_col="bucket")
        # With lags up to 48, all rows will be NaN — result should be empty
        self.assertEqual(len(X), 0)


# ── LightGBM Forecaster Tests ────────────────────────────────────────────

class TestLightGBMForecaster(unittest.TestCase):
    """Tests for ml.forecast.LightGBMForecaster."""

    def setUp(self):
        """Create a larger synthetic dataset for training."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        # Synthetic seasonal signal + noise
        t = np.arange(n)
        signal = 20 + 10 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 2, n)
        self.df = pd.DataFrame({
            "bucket": dates,
            "avg_val": signal,
        })

    def test_forecaster_import(self):
        from ml.forecast import LightGBMForecaster, HAS_LIGHTGBM
        self.assertTrue(HAS_LIGHTGBM, "LightGBM should be installed")

    def test_forecaster_train_and_predict(self):
        from ml.forecast import LightGBMForecaster
        from ml.features import make_feature_matrix

        X, y = make_feature_matrix(self.df, "avg_val", ts_col="bucket")
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        forecaster = LightGBMForecaster()
        forecaster.fit(X_train, y_train, X_val, y_val)

        # Should have trained 3 models (10th, 50th, 90th quantiles)
        self.assertEqual(len(forecaster.models), 3)
        self.assertIn(0.1, forecaster.models)
        self.assertIn(0.5, forecaster.models)
        self.assertIn(0.9, forecaster.models)

        # Predict
        result = forecaster.predict(X_val)
        self.assertEqual(len(result.predictions), len(X_val))
        self.assertEqual(len(result.lower_bound), len(X_val))
        self.assertEqual(len(result.upper_bound), len(X_val))
        self.assertEqual(result.model_name, "lightgbm_quantile")

    def test_forecast_result_ordering(self):
        """Lower bound should generally be <= median <= upper bound."""
        from ml.forecast import LightGBMForecaster
        from ml.features import make_feature_matrix

        X, y = make_feature_matrix(self.df, "avg_val", ts_col="bucket")
        split = int(len(X) * 0.8)

        forecaster = LightGBMForecaster()
        forecaster.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        result = forecaster.predict(X.iloc[split:])

        # Most predictions should have lower <= predicted <= upper
        violations = 0
        for lo, mid, hi in zip(result.lower_bound, result.predictions, result.upper_bound):
            if lo > mid or mid > hi:
                violations += 1
        # Allow some noise but mostly should be ordered
        violation_rate = violations / len(result.predictions)
        self.assertLess(violation_rate, 0.3, "Too many quantile ordering violations")

    def test_predict_without_training_raises(self):
        from ml.forecast import LightGBMForecaster

        forecaster = LightGBMForecaster()
        with self.assertRaises(RuntimeError):
            forecaster.predict(pd.DataFrame({"x": [1, 2, 3]}))


class TestSeasonalNaive(unittest.TestCase):
    """Tests for the seasonal naive baseline forecast."""

    def test_seasonal_naive_forecast(self):
        from ml.forecast import seasonal_naive_forecast

        series = pd.Series(np.arange(100, dtype=float))
        result = seasonal_naive_forecast(series, horizon=5, season=10)

        self.assertEqual(result.horizon_steps, 5)
        self.assertEqual(len(result.predictions), 5)
        self.assertEqual(len(result.lower_bound), 5)
        self.assertEqual(len(result.upper_bound), 5)
        self.assertEqual(result.model_name, "seasonal_naive")

    def test_seasonal_naive_lower_upper_bounds(self):
        from ml.forecast import seasonal_naive_forecast

        series = pd.Series([10.0] * 50)
        result = seasonal_naive_forecast(series, horizon=3, season=10)

        for lo, pred, hi in zip(result.lower_bound, result.predictions, result.upper_bound):
            self.assertLessEqual(lo, pred)
            self.assertLessEqual(pred, hi)


class TestSmape(unittest.TestCase):
    """Tests for the sMAPE metric."""

    def test_smape_perfect(self):
        from ml.forecast import smape

        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(smape(actual, predicted), 0.0)

    def test_smape_symmetric(self):
        from ml.forecast import smape

        a = np.array([10.0, 20.0])
        b = np.array([15.0, 25.0])
        self.assertAlmostEqual(smape(a, b), smape(b, a))

    def test_smape_range(self):
        from ml.forecast import smape

        a = np.array([10.0, 20.0, 30.0])
        b = np.array([1.0, 2.0, 3.0])
        result = smape(a, b)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 200)


# ── Isolation Forest / Anomaly Detection Tests ───────────────────────────

class TestIsolationForestDetection(unittest.TestCase):
    """Tests for ml.anomaly module."""

    def setUp(self):
        """Create synthetic metric data with injected anomalies."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")

        # Normal data
        data = {
            "bucket": dates,
            "db.qps": np.random.normal(30, 3, n),
            "db.connections": np.random.normal(10, 2, n),
            "db.cache_hit_ratio": np.random.normal(0.95, 0.02, n),
            "db.deadlocks": np.zeros(n),
            "db.temp_files": np.random.normal(0, 0.5, n).clip(0),
            "db.rollback_ratio": np.random.normal(0.01, 0.005, n).clip(0),
        }
        # Inject anomalies at specific points
        for idx in [50, 100, 150]:
            data["db.qps"][idx] = 200      # massive QPS spike
            data["db.connections"][idx] = 90  # connection storm
            data["db.deadlocks"][idx] = 10   # deadlock burst

        self.df = pd.DataFrame(data)
        self.timestamps = self.df["bucket"]
        self.metric_df = self.df.drop(columns=["bucket"])

    def test_detect_isolation_forest_basic(self):
        from ml.anomaly import detect_isolation_forest

        results = detect_isolation_forest(
            self.metric_df, self.timestamps, contamination=0.05,
        )

        # Should detect some anomalies
        self.assertGreater(len(results), 0)

        # All results should be anomalies
        for a in results:
            self.assertTrue(a.is_anomaly)
            self.assertEqual(a.method, "isolation_forest")
            self.assertIsInstance(a.score, float)

    def test_detect_isolation_forest_finds_injected_anomalies(self):
        """The injected extreme anomalies should be detected."""
        from ml.anomaly import detect_isolation_forest

        results = detect_isolation_forest(
            self.metric_df, self.timestamps, contamination=0.10,
        )

        detected_indices = set()
        for a in results:
            # Find the index of this timestamp
            idx = self.timestamps[self.timestamps == a.ts].index
            if len(idx) > 0:
                detected_indices.add(idx[0])

        # At least 2 of the 3 injected anomalies should be detected
        injected = {50, 100, 150}
        found = detected_indices & injected
        self.assertGreaterEqual(len(found), 2,
            f"Expected ≥2 of {injected} detected, got {found}")

    def test_anomaly_score_properties(self):
        from ml.anomaly import AnomalyScore

        a = AnomalyScore(
            ts=datetime(2024, 1, 1),
            metric="test",
            value=42.0,
            score=0.8,
            method="test_method",
            is_anomaly=True,
        )
        self.assertEqual(a.metric, "test")
        self.assertTrue(a.is_anomaly)
        self.assertEqual(a.context, {})  # default

    def test_merge_and_deduplicate(self):
        from ml.anomaly import merge_and_deduplicate, AnomalyScore

        base_time = datetime(2024, 1, 1, 12, 0, 0)
        list1 = [
            AnomalyScore(ts=base_time, metric="qps", value=100, score=0.9,
                         method="isolation_forest", is_anomaly=True),
            AnomalyScore(ts=base_time + timedelta(seconds=60), metric="qps", value=105,
                         score=0.85, method="isolation_forest", is_anomaly=True),
        ]
        list2 = [
            AnomalyScore(ts=base_time + timedelta(seconds=30), metric="qps", value=102,
                         score=0.8, method="residual", is_anomaly=True),
        ]

        merged = merge_and_deduplicate([list1, list2], cooldown_seconds=300)

        # With 300s cooldown, second anomaly from list1 with same metric+method
        # should be suppressed (only 60s apart)
        self.assertLessEqual(len(merged), 3)
        self.assertGreaterEqual(len(merged), 2)

    def test_merge_empty_lists(self):
        from ml.anomaly import merge_and_deduplicate

        result = merge_and_deduplicate([[], []])
        self.assertEqual(result, [])


class TestResidualAnomalies(unittest.TestCase):
    """Tests for residual-based anomaly detection."""

    def test_detect_residual_anomalies_basic(self):
        from ml.anomaly import detect_residual_anomalies

        n = 100
        actual = pd.Series(np.random.normal(20, 2, n))
        lower = pd.Series(np.full(n, 15.0))
        upper = pd.Series(np.full(n, 25.0))
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=n, freq="5min"))

        # Inject outlier
        actual.iloc[50] = 50.0  # way above upper

        results = detect_residual_anomalies(
            actual, lower, upper, timestamps,
            metric="test_metric",
            persistence_n=1,
            persistence_m=1,
        )

        # Should detect the outlier
        self.assertGreater(len(results), 0)
        anomaly_ts = [a.ts for a in results]
        self.assertIn(timestamps.iloc[50], anomaly_ts)

    def test_residual_no_anomalies_when_within_bounds(self):
        from ml.anomaly import detect_residual_anomalies

        n = 50
        actual = pd.Series(np.full(n, 20.0))
        lower = pd.Series(np.full(n, 15.0))
        upper = pd.Series(np.full(n, 25.0))
        timestamps = pd.Series(pd.date_range("2024-01-01", periods=n, freq="5min"))

        results = detect_residual_anomalies(
            actual, lower, upper, timestamps,
            metric="test_metric",
        )
        self.assertEqual(len(results), 0)


# ── Walk-Forward Evaluation Tests ─────────────────────────────────────────

class TestWalkForwardEvaluate(unittest.TestCase):
    """Tests for walk-forward evaluation in ml.forecast."""

    def test_walk_forward_runs(self):
        from ml.forecast import walk_forward_evaluate

        np.random.seed(42)
        n = 800
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        t = np.arange(n)
        signal = 20 + 10 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 2, n)

        df = pd.DataFrame({"bucket": dates, "avg_val": signal})

        results = walk_forward_evaluate(
            df, target_col="avg_val", ts_col="bucket",
            n_splits=2, horizon=50,
        )

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("split", r)
            self.assertIn("baseline_smape", r)
            self.assertGreater(r["n_train"], 0)


# ── Integration-style Tests ──────────────────────────────────────────────

class TestEndToEndForecastPipeline(unittest.TestCase):
    """
    End-to-end test: raw DataFrame → feature engineering → train → predict.
    Simulates what the API router does without requiring a database.
    """

    def test_full_forecast_pipeline(self):
        from ml.forecast import LightGBMForecaster, smape
        from ml.features import make_feature_matrix

        np.random.seed(123)
        n = 300
        dates = pd.date_range("2024-06-01", periods=n, freq="5min")
        t = np.arange(n)
        qps = 25 + 8 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 1.5, n)

        df = pd.DataFrame({"bucket": dates, "avg_val": qps})

        # Feature engineering
        X, y = make_feature_matrix(df, "avg_val", ts_col="bucket")
        self.assertGreater(len(X), 100)

        # Split
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_val, y_val = X.iloc[split:], y.iloc[split:]

        # Train
        forecaster = LightGBMForecaster()
        forecaster.fit(X_train, y_train, X_val, y_val)

        # Predict
        result = forecaster.predict(X_val)

        # Evaluate
        val_smape = smape(y_val.values, np.array(result.predictions))
        self.assertLess(val_smape, 100, "sMAPE should be reasonable")

        # Model should produce valid predictions
        for pred in result.predictions:
            self.assertIsInstance(pred, float)
            self.assertFalse(np.isnan(pred))


class TestEndToEndAnomalyPipeline(unittest.TestCase):
    """
    End-to-end test: synthetic metrics → Isolation Forest → detect anomalies.
    Simulates what the API anomaly detection endpoint does.
    """

    def test_full_anomaly_pipeline(self):
        from ml.anomaly import detect_isolation_forest, merge_and_deduplicate

        np.random.seed(99)
        n = 200
        dates = pd.date_range("2024-06-01", periods=n, freq="5min")

        normal_data = pd.DataFrame({
            "db.qps": np.random.normal(30, 3, n),
            "db.connections": np.random.normal(10, 2, n),
            "db.cache_hit_ratio": np.random.normal(0.95, 0.02, n),
        })

        # Inject clear anomalies
        normal_data.loc[75, "db.qps"] = 200
        normal_data.loc[75, "db.connections"] = 80
        normal_data.loc[140, "db.qps"] = -10
        normal_data.loc[140, "db.cache_hit_ratio"] = 0.3

        timestamps = pd.Series(dates)

        # Detect
        results = detect_isolation_forest(normal_data, timestamps, contamination=0.05)

        self.assertGreater(len(results), 0)

        # Merge (even with single detector, should work)
        merged = merge_and_deduplicate([results], cooldown_seconds=300)
        self.assertGreater(len(merged), 0)

        # Verify structure
        for a in merged:
            self.assertIsNotNone(a.ts)
            self.assertIsInstance(a.score, float)
            self.assertTrue(a.is_anomaly)


class TestLightGBMPredictFuture(unittest.TestCase):
    """Tests for autoregressive multi-step forward forecasting."""

    def setUp(self):
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        t = np.arange(n)
        signal = 25 + 8 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 1.5, n)
        self.df = pd.DataFrame({"bucket": dates, "avg_val": signal})

    def test_predict_future_steps(self):
        from ml.forecast import LightGBMForecaster
        from ml.features import make_feature_matrix

        X, y = make_feature_matrix(self.df, "avg_val", ts_col="bucket")
        split = int(len(X) * 0.8)

        forecaster = LightGBMForecaster()
        forecaster.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        horizon = 12
        result = forecaster.predict_future(
            self.df,
            target_col="avg_val",
            ts_col="bucket",
            horizon=horizon,
            freq_minutes=5,
        )

        self.assertEqual(result.horizon_steps, horizon)
        self.assertEqual(len(result.predictions), horizon)
        self.assertEqual(len(result.lower_bound), horizon)
        self.assertEqual(len(result.upper_bound), horizon)

        # All values should be non-negative and finite
        for p, lo, hi in zip(result.predictions, result.lower_bound, result.upper_bound):
            self.assertGreaterEqual(p, 0.0)
            self.assertFalse(np.isnan(p))
            self.assertFalse(np.isnan(lo))
            self.assertFalse(np.isnan(hi))


class TestIsolationForestDetectorClass(unittest.TestCase):
    """Tests for IsolationForestDetector class interface."""

    def test_detector_fit_and_detect(self):
        from ml.anomaly import IsolationForestDetector

        np.random.seed(42)
        n = 150
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        df = pd.DataFrame({
            "db.qps": np.random.normal(30, 3, n),
            "db.connections": np.random.normal(10, 2, n),
            "db.cache_hit_ratio": np.random.normal(0.95, 0.02, n),
        })
        # Inject an anomaly
        df.loc[50, "db.qps"] = 500

        detector = IsolationForestDetector(contamination=0.05)
        fit_res = detector.fit(df)

        self.assertEqual(fit_res["samples_used"], n)
        self.assertEqual(len(fit_res["features"]), 3)

        anomalies = detector.detect(df, pd.Series(dates))
        self.assertGreater(len(anomalies), 0)

        # Check anomaly structure
        for a in anomalies:
            self.assertTrue(a.is_anomaly)
            self.assertEqual(a.method, "isolation_forest")
            self.assertIn("metrics", a.context)
            self.assertIn("db.qps", a.context["metrics"])

    def test_detector_detect_missing_columns(self):
        """Test that detector gracefully handles missing feature columns during detection."""
        from ml.anomaly import IsolationForestDetector

        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        df_train = pd.DataFrame({
            "db.qps": np.random.normal(30, 3, n),
            "db.connections": np.random.normal(10, 2, n),
            "db.cache_hit_ratio": np.random.normal(0.95, 0.02, n),
            "db.deadlocks": np.zeros(n),
        })
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(df_train)

        # Detection dataframe is missing db.deadlocks and db.cache_hit_ratio
        df_detect = pd.DataFrame({
            "db.qps": np.random.normal(30, 3, 20),
            "db.connections": np.random.normal(10, 2, 20),
        })
        detect_dates = pd.date_range("2024-01-02", periods=20, freq="5min")

        # Should not raise KeyError or shape mismatch
        anomalies = detector.detect(df_detect, pd.Series(detect_dates))
        self.assertIsInstance(anomalies, list)

    def test_detector_with_all_nans_in_column(self):
        """Test that detector handles columns that are all NaN without error."""
        from ml.anomaly import IsolationForestDetector

        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        df = pd.DataFrame({
            "db.qps": np.random.normal(30, 3, n),
            "db.connections": [np.nan] * n,
        })
        detector = IsolationForestDetector(contamination=0.05)
        fit_res = detector.fit(df)
        self.assertEqual(fit_res["samples_used"], n)

        anomalies = detector.detect(df, pd.Series(dates))
        self.assertIsInstance(anomalies, list)

    def test_detector_save_and_load(self):
        """Test IsolationForestDetector serialization and loading."""
        import tempfile
        import os
        from ml.anomaly import IsolationForestDetector

        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        df = pd.DataFrame({
            "db.qps": np.random.normal(30, 3, n),
            "db.connections": np.random.normal(10, 2, n),
        })
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(df)

        orig_anomalies = detector.detect(df, pd.Series(dates))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "detector.joblib")
            detector.save(model_file)
            self.assertTrue(os.path.exists(model_file))

            loaded = IsolationForestDetector.load(model_file)
            loaded_anomalies = loaded.detect(df, pd.Series(dates))

            self.assertEqual(len(orig_anomalies), len(loaded_anomalies))
            for o, l in zip(orig_anomalies, loaded_anomalies):
                self.assertEqual(o.ts, l.ts)
                self.assertEqual(o.score, l.score)


class TestLightGBMSaveAndLoad(unittest.TestCase):
    """Tests for LightGBMForecaster model persistence."""

    def test_forecaster_save_and_load(self):
        import tempfile
        import os
        from ml.forecast import LightGBMForecaster
        from ml.features import make_feature_matrix

        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="5min")
        t = np.arange(n)
        signal = 25 + 8 * np.sin(2 * np.pi * t / 288) + np.random.normal(0, 1.5, n)
        df = pd.DataFrame({"bucket": dates, "avg_val": signal})

        X, y = make_feature_matrix(df, "avg_val", ts_col="bucket")
        split = int(len(X) * 0.8)

        forecaster = LightGBMForecaster()
        forecaster.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        orig_result = forecaster.predict(X.iloc[split:])

        with tempfile.TemporaryDirectory() as tmpdir:
            forecaster.save(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "meta.json")))

            loaded = LightGBMForecaster.load(tmpdir)
            loaded_result = loaded.predict(X.iloc[split:])

            self.assertEqual(len(orig_result.predictions), len(loaded_result.predictions))
            np.testing.assert_allclose(orig_result.predictions, loaded_result.predictions, rtol=1e-5)


class TestApiIntegration(unittest.TestCase):
    """Integration tests for FastAPI ML endpoints using TestClient and mock DB."""

    def setUp(self):
        from unittest.mock import AsyncMock, MagicMock
        from starlette.testclient import TestClient
        from api.main import app
        from api.database import get_db

        # Build synthetic metrics rows
        np.random.seed(42)
        n = 120
        start = datetime.now() - timedelta(hours=24)
        dates = [start + timedelta(minutes=5 * i) for i in range(n)]

        self.mock_qps_rows = [
            {"bucket": d, "avg_val": float(20 + 5 * np.sin(i / 10))}
            for i, d in enumerate(dates)
        ]

        self.mock_pivot_rows = []
        for metric in ["db.qps", "db.connections", "db.cache_hit_ratio"]:
            for i, d in enumerate(dates):
                val = 25.0 if metric == "db.qps" else (10.0 if metric == "db.connections" else 0.98)
                if i == 50 and metric == "db.qps":
                    val = 500.0  # anomaly
                self.mock_pivot_rows.append({"bucket": d, "metric": metric, "avg_val": val})

        # Mock asyncpg connection and pool
        self.mock_conn = MagicMock()
        self.mock_pool = MagicMock()

        class MockAcquireContext:
            def __init__(self, conn):
                self.conn = conn
            async def __aenter__(self):
                return self.conn
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        self.mock_pool.acquire.return_value = MockAcquireContext(self.mock_conn)

        async def fake_get_db():
            return self.mock_pool

        app.dependency_overrides[get_db] = fake_get_db
        self.client = TestClient(app)

    def tearDown(self):
        from api.main import app
        app.dependency_overrides.clear()

    def test_forecast_flow(self):
        """Test forecast model-status -> train -> predict."""
        # 1. Model status before training
        resp = self.client.get("/forecast/model-status")
        self.assertEqual(resp.status_code, 200)

        # 2. Train model
        self.mock_conn.fetch = AsyncMock(return_value=[
            {"bucket": r["bucket"], "avg_val": r["avg_val"]} for r in self.mock_qps_rows
        ])

        train_resp = self.client.post("/forecast/train", json={"metric": "db.qps", "hours": 24})
        self.assertEqual(train_resp.status_code, 200)
        self.assertEqual(train_resp.json()["status"], "trained")

        # 3. Model status after training
        status_resp = self.client.get("/forecast/model-status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertTrue(status_resp.json()["trained"])

        # 4. Predict
        forecast_resp = self.client.get("/forecast/?metric=db.qps&hours=24&horizon=6")
        self.assertEqual(forecast_resp.status_code, 200)
        data = forecast_resp.json()
        self.assertIn("baseline_forecast", data)
        self.assertIn("lightgbm_forecast", data)
        self.assertEqual(len(data["lightgbm_forecast"]), 6)

    def test_anomaly_flow(self):
        """Test anomaly model-status -> train -> detect."""
        # 1. Model status before training
        resp = self.client.get("/anomalies/model-status")
        self.assertEqual(resp.status_code, 200)

        # 2. Train model
        self.mock_conn.fetch = AsyncMock(return_value=[
            {"bucket": r["bucket"], "metric": r["metric"], "avg_val": r["avg_val"]}
            for r in self.mock_pivot_rows
        ])

        train_resp = self.client.post("/anomalies/train", json={"hours": 24, "contamination": 0.05})
        self.assertEqual(train_resp.status_code, 200)
        self.assertEqual(train_resp.json()["status"], "trained")

        # 3. Model status after training
        status_resp = self.client.get("/anomalies/model-status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertTrue(status_resp.json()["trained"])

        # 4. Detect with record=True
        self.mock_conn.fetchval = AsyncMock(return_value=None)
        self.mock_conn.execute = AsyncMock(return_value="INSERT 0 1")

        detect_resp = self.client.get("/anomalies/detect?hours=6&record=true")
        self.assertEqual(detect_resp.status_code, 200)
        data = detect_resp.json()
        self.assertEqual(data["status"], "detected")
        self.assertEqual(data["model_source"], "pre_trained")
        self.assertIn("anomalies", data)
        self.assertGreater(data["anomaly_count"], 0)


if __name__ == "__main__":
    unittest.main()


