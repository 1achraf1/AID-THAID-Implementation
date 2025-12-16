import time
import gc
import sys
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes, make_friedman1
from aid import AID

# Optional plotting (safe if missing)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Optional OpenML datasets (Ames, CPU Act, etc.)
try:
    import openml
except Exception:
    openml = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_object_size_bytes(obj) -> int:
    # rough python object size, not full RAM (like your notebook remark)
    return int(sys.getsizeof(obj))


def _fit_predict_times(model, X_train, y_train, X_test) -> Tuple[float, float, np.ndarray]:
    gc.collect()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_test)
    pred_s = time.perf_counter() - t0
    return fit_s, pred_s, pred


def _metrics(y_test, pred) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_test, pred),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "R2": float(r2_score(y_test, pred)),
    }


# ------------------------------------------------------------
# Tester
# ------------------------------------------------------------
class AIDTester:
    """Comprehensive testing for AID (regression)."""

    def __init__(self):
        self.results: Dict[str, Any] = {}

    # -----------------------
    # Datasets
    # -----------------------
    def load_datasets(self, random_state: int = 42) -> Dict[str, Dict[str, Any]]:
        datasets: Dict[str, Dict[str, Any]] = {}

        # 1) California Housing (sklearn)
        cal = fetch_california_housing(as_frame=True)
        datasets["california"] = {
            "X": cal.data,
            "y": cal.target,
            "description": f"California Housing (sklearn) ({cal.data.shape[0]} samples, {cal.data.shape[1]} features)"
        }

        # 2) Diabetes (sklearn)
        dia = load_diabetes(as_frame=True)
        datasets["diabetes"] = {
            "X": dia.data,
            "y": dia.target,
            "description": f"Diabetes (sklearn) ({dia.data.shape[0]} samples, {dia.data.shape[1]} features)"
        }

        # 3) Synthetic non-linear (Friedman1)
        Xs, ys = make_friedman1(n_samples=2500, n_features=10, noise=1.0, random_state=random_state)
        datasets["synthetic_friedman1"] = {
            "X": pd.DataFrame(Xs, columns=[f"x{i}" for i in range(Xs.shape[1])]),
            "y": pd.Series(ys),
            "description": f"Synthetic Friedman1 ({Xs.shape[0]} samples, {Xs.shape[1]} features)"
        }

        # 4) OpenML: Ames Housing + CPU Act (if openml available)
        if openml is not None:
            # Ames Housing: often large-ish, regression target varies by version
            # We'll try common IDs; if fail, skip.
            for name, openml_id, target_col in [
                ("ames", 42165, "SalePrice"),  # OpenML "house_prices" style
                ("cpu_act", 561, None),         # "cpu_act" target column auto-detect
            ]:
                try:
                    ds = openml.datasets.get_dataset(openml_id)
                    X, y, _, _ = ds.get_data(target=target_col)
                    # Ensure numeric only (AID expects float)
                    X = X.select_dtypes(include=[np.number]).copy()
                    y = pd.Series(y).astype(float)

                    datasets[name] = {
                        "X": X,
                        "y": y,
                        "description": f"{ds.name} (OpenML id={openml_id}) ({X.shape[0]} samples, {X.shape[1]} numeric features)"
                    }
                except Exception as e:
                    print(f"[OpenML] Could not load {name} (id={openml_id}): {e}")

        
        # ------------------------------------------------------------
        # Export datasets for R (shared_datasets/*.csv)
        # No prints here to preserve output shape.
        # ------------------------------------------------------------
        os.makedirs("shared_datasets", exist_ok=True)
        for _k, _d in datasets.items():
            _X = _d["X"]
            _y = _d["y"]

            if hasattr(_X, "to_csv"):
                _X_df = _X
            else:
                _X_df = pd.DataFrame(_X)

            _X_df.to_csv(f"shared_datasets/{_k}_X.csv", index=False)

            # y as single column, no header (easy to read in R)
            pd.Series(np.asarray(_y, dtype=float)).to_csv(
                f"shared_datasets/{_k}_y.csv", index=False, header=False
            )


        return datasets
    # -----------------------
    # Basic test
    # -----------------------
    def test_basic_functionality(self, datasets: Dict[str, Dict[str, Any]], test_size: float = 0.25, random_state: int = 42):
        print("=" * 90)
        print("BASIC FUNCTIONALITY TEST (train/test split)")
        print("=" * 90)

        rows = []

        for key, data in datasets.items():
            X = data["X"]
            y = data["y"]
            print(f"\n{key.upper()}: {data['description']}")
            print("-" * 70)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Models: AID + DecisionTree + HGB
            model_specs = [
                ("AID", lambda: AID(min_samples_leaf=15, min_samples_split=30, max_depth=6, min_gain=1e-3, store_history=True, presort=True)),
                ("DecisionTreeRegressor", lambda: DecisionTreeRegressor(random_state=random_state)),
                ("HistGradientBoostingRegressor", lambda: HistGradientBoostingRegressor(random_state=random_state)),
            ]

            for model_name, ctor in model_specs:
                try:
                    model = ctor()
                    fit_s, pred_s, pred = _fit_predict_times(model, X_train, y_train, X_test)
                    m = _metrics(y_test, pred)

                    row = {
                        "dataset": data["description"].split(" (")[0],
                        "dataset_key": key,
                        "model": model_name,
                        "fit_s": fit_s,
                        "pred_s": pred_s,
                        **m,
                    }
                    rows.append(row)

                    print(f"{model_name:28s} | fit={fit_s:.4f}s pred={pred_s:.4f}s | RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

                except Exception as e:
                    print(f"{model_name:28s} | ERROR: {e}")

        df = pd.DataFrame(rows).sort_values(["dataset_key", "RMSE"], ascending=[True, True]).reset_index(drop=True)
        self.results["basic"] = df
        return df

    # -----------------------
    # Cross-validation
    # -----------------------
    def test_cross_validation(self, datasets: Dict[str, Dict[str, Any]], cv: int = 5, random_state: int = 42):
        print("\n" + "=" * 90)
        print("CROSS-VALIDATION TEST (KFold)")
        print("=" * 90)

        rows = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        model_specs = [
            ("AID", lambda: AID(min_samples_leaf=15, min_samples_split=30, max_depth=6, min_gain=1e-3, store_history=False, presort=True)),
            ("DecisionTreeRegressor", lambda: DecisionTreeRegressor(random_state=random_state)),
            ("HistGradientBoostingRegressor", lambda: HistGradientBoostingRegressor(random_state=random_state)),
        ]

        for key, data in datasets.items():
            X = data["X"]
            y = data["y"]
            print(f"\n{key.upper()}: {data['description']}")
            print("-" * 70)

            X_np = np.asarray(X)
            y_np = np.asarray(y, dtype=float)

            for model_name, ctor in model_specs:
                fold_rmses = []
                fold_times = []

                try:
                    for fold, (tr, te) in enumerate(kf.split(X_np), 1):
                        model = ctor()

                        t0 = time.perf_counter()
                        model.fit(X_np[tr], y_np[tr])
                        fit_s = time.perf_counter() - t0

                        pred = model.predict(X_np[te])
                        fold_rmses.append(rmse(y_np[te], pred))
                        fold_times.append(fit_s)

                        print(f"  {model_name:28s} fold {fold}: RMSE={fold_rmses[-1]:.4f} fit={fit_s:.4f}s")

                    rows.append({
                        "dataset_key": key,
                        "dataset": data["description"].split(" (")[0],
                        "model": model_name,
                        "cv": cv,
                        "rmse_mean": float(np.mean(fold_rmses)),
                        "rmse_std": float(np.std(fold_rmses)),
                        "fit_s_mean": float(np.mean(fold_times)),
                        "fit_s_std": float(np.std(fold_times)),
                    })

                    print(f"  -> {model_name:28s} mean RMSE={np.mean(fold_rmses):.4f} (+/- {np.std(fold_rmses):.4f})")

                except Exception as e:
                    print(f"  {model_name:28s} | ERROR: {e}")

        df = pd.DataFrame(rows).sort_values(["dataset_key", "rmse_mean"]).reset_index(drop=True)
        self.results["cv"] = df
        return df

    # -----------------------
    # Parameter sensitivity (AID only)
    # -----------------------
    def test_parameter_sensitivity(self, datasets: Dict[str, Dict[str, Any]], random_state: int = 42):
        print("\n" + "=" * 90)
        print("PARAMETER SENSITIVITY TEST (AID only)")
        print("=" * 90)

        param_configs = [
            {"min_samples_leaf": 10, "min_samples_split": 20, "max_depth": 4, "min_gain": 0.0},
            {"min_samples_leaf": 15, "min_samples_split": 30, "max_depth": 6, "min_gain": 1e-3},
            {"min_samples_leaf": 20, "min_samples_split": 40, "max_depth": 6, "min_gain": 1e-3},
            {"min_samples_leaf": 15, "min_samples_split": 30, "max_depth": 8, "min_gain": 1e-3},
            {"min_samples_leaf": 30, "min_samples_split": 60, "max_depth": 5, "min_gain": 1e-3},
        ]

        rows = []

        for key, data in datasets.items():
            X = data["X"]
            y = data["y"]

            print(f"\n{key.upper()}: {data['description']}")
            print("-" * 70)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

            for i, params in enumerate(param_configs, 1):
                try:
                    model = AID(
                        min_samples_leaf=params["min_samples_leaf"],
                        min_samples_split=params["min_samples_split"],
                        max_depth=params["max_depth"],
                        min_gain=params["min_gain"],
                        store_history=False,
                        presort=True
                    )
                    fit_s, pred_s, pred = _fit_predict_times(model, X_train, y_train, X_test)
                    m = _metrics(y_test, pred)

                    rows.append({
                        "dataset_key": key,
                        "dataset": data["description"].split(" (")[0],
                        "config_id": i,
                        **params,
                        "fit_s": fit_s,
                        "pred_s": pred_s,
                        **m,
                    })

                    print(f"  Config {i} {params} -> RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} fit={fit_s:.4f}s")

                except Exception as e:
                    print(f"  Config {i} {params} -> ERROR: {e}")

        df = pd.DataFrame(rows).sort_values(["dataset_key", "RMSE"]).reset_index(drop=True)
        self.results["params"] = df
        return df

    # -----------------------
    # Edge cases
    # -----------------------
    def test_edge_cases(self, datasets: Dict[str, Dict[str, Any]]):
        print("\n" + "=" * 90)
        print("EDGE CASES TEST")
        print("=" * 90)

        # pick first dataset
        first_key = list(datasets.keys())[0]
        X = datasets[first_key]["X"]
        y = datasets[first_key]["y"]

        X_np = np.asarray(X)
        y_np = np.asarray(y, dtype=float)

        test_cases = {
            "single_sample": (X_np[:1], y_np[:1]),
            "two_samples": (X_np[:2], y_np[:2]),
            "single_feature": (X_np[:, :1], y_np),
            "tiny_subset_25": (X_np[:25], y_np[:25]),
        }

        rows = []
        for name, (Xt, yt) in test_cases.items():
            print(f"\n{name.upper()}")
            print("-" * 70)
            try:
                model = AID(min_samples_leaf=2, min_samples_split=2, max_depth=3, min_gain=0.0, store_history=True, presort=True)
                model.fit(Xt, yt)
                pred = model.predict(Xt)
                m = _metrics(yt, pred)
                print(f"  ✓ SUCCESS: RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} | history_splits={len(model.history_) if hasattr(model, 'history_') else 'N/A'}")
                rows.append({"case": name, "success": True, **m})
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                rows.append({"case": name, "success": False, "error": str(e)})

        df = pd.DataFrame(rows)
        self.results["edge"] = df
        return df

    # -----------------------
    # Repeated benchmark (timeit-like) + memory
    # -----------------------
    def repeated_benchmark_aid(self, datasets: Dict[str, Dict[str, Any]], key: str = "california", repeats: int = 3):
        print("\n" + "=" * 90)
        print("REPEATED BENCHMARK (AID) + MEMORY (rough)")
        print("=" * 90)

        data = datasets[key]
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=float)

        times = []
        models = []

        for r in range(repeats):
            gc.collect()
            model = AID(min_samples_leaf=15, min_samples_split=30, max_depth=6, min_gain=1e-3, store_history=True, presort=True)
            t0 = time.perf_counter()
            model.fit(X, y)
            t = time.perf_counter() - t0
            times.append(t)
            models.append(model)

        best_i = int(np.argmin(times))
        best_model = models[best_i]

        print(f"Dataset: {data['description']}")
        print("elapsed_s:", times)
        print("min:", min(times), "mean:", float(np.mean(times)))
        print("sys.getsizeof(model):", _safe_object_size_bytes(best_model), "bytes")
        return best_model

    # -----------------------
    # Visualization + report
    # -----------------------
    def visualize_results(self, out_prefix: str = "aid"):
        if plt is None:
            print("\n[visualize_results] matplotlib not available -> skip plots.")
            return

        if "basic" not in self.results:
            print("\n[visualize_results] no basic results -> skip.")
            return

        df = self.results["basic"].copy()

        # Simple plot: RMSE by model (grouped per dataset)
        fig = plt.figure(figsize=(10, 5))
        for ds in df["dataset_key"].unique():
            sub = df[df["dataset_key"] == ds]
            plt.plot(sub["model"], sub["RMSE"], marker="o", label=ds)
        plt.title("AID vs Trees — RMSE (lower is better)")
        plt.xlabel("Model")
        plt.ylabel("RMSE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = f"{out_prefix}_rmse.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {path}")

    def generate_report(self, out_path: str = "aid_test_report.txt"):
        print("\n" + "=" * 90)
        print("COMPREHENSIVE TEST REPORT")
        print("=" * 90)

        lines: List[str] = []
        lines.append("AID (Regression) — Test Report")
        lines.append("=" * 90)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        if "basic" in self.results:
            lines.append("1) Basic train/test results (sorted by RMSE per dataset)")
            lines.append("-" * 90)
            df = self.results["basic"].copy()
            lines.append(df.to_string(index=False))
            lines.append("")

        if "cv" in self.results:
            lines.append("2) Cross-validation results (mean RMSE)")
            lines.append("-" * 90)
            df = self.results["cv"].copy()
            lines.append(df.to_string(index=False))
            lines.append("")

        if "params" in self.results:
            lines.append("3) AID hyperparameter sensitivity (sorted by RMSE)")
            lines.append("-" * 90)
            df = self.results["params"].copy()
            lines.append(df.head(50).to_string(index=False))
            lines.append("")

        if "edge" in self.results:
            lines.append("4) Edge cases")
            lines.append("-" * 90)
            df = self.results["edge"].copy()
            lines.append(df.to_string(index=False))
            lines.append("")

        report = "\n".join(lines)
        print(report)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✓ Saved: {out_path}")

        return report

    # -----------------------
    # Save tables
    # -----------------------
    def export_tables(self, prefix: str = "aid"):
        if "basic" in self.results:
            path = f"{prefix}_results.csv"
            self.results["basic"].to_csv(path, index=False)
            print(f"✓ Saved: {path}")

        if "cv" in self.results:
            path = f"{prefix}_cv_results.csv"
            self.results["cv"].to_csv(path, index=False)
            print(f"✓ Saved: {path}")

        if "params" in self.results:
            path = f"{prefix}_param_results.csv"
            self.results["params"].to_csv(path, index=False)
            print(f"✓ Saved: {path}")


def run_complete_test():
    print("\n" + "=" * 90)
    print("AID REGRESSOR — COMPLETE TEST SUITE")
    print("=" * 90)

    tester = AIDTester()

    print("\nLoading datasets...")
    datasets = tester.load_datasets()
    print(f"✓ Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    # Run tests
    basic_df = tester.test_basic_functionality(datasets)
    cv_df = tester.test_cross_validation(datasets, cv=5)
    params_df = tester.test_parameter_sensitivity(datasets)
    edge_df = tester.test_edge_cases(datasets)

    # Repeated benchmark (like your notebook cell)
    _ = tester.repeated_benchmark_aid(datasets, key="california", repeats=3)

    # Exports + plots + report
    tester.export_tables(prefix="aid")
    tester.visualize_results(out_prefix="aid")
    tester.generate_report(out_path="aid_test_report.txt")

    print("\n" + "=" * 90)
    print("TEST SUITE COMPLETED")
    print("=" * 90)

    return tester


if __name__ == "__main__":
    # Reduce noise
    warnings.filterwarnings("ignore")
    tester = run_complete_test()
