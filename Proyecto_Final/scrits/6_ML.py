# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:09:46 2024

@author: jrbof
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from datetime import datetime
import json


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
)


from funciones.modulo_funciones  import (makecar,
                                         pick_target,
                                         load_and_process_data,
                                         )

# Suppress pandas FutureWarning triggered inside seaborn regarding 'use_inf_as_na'
warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*")
sns.set_style('darkgrid')
sns.set_style('ticks', {"axes.grid": False})

# working directory
parentDirectory = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))

makecar(parentDirectory, "images")
# directory to save images
dirimg = os.path.join(parentDirectory, "images")
# SensoresR directory
dirsenr = os.path.join(parentDirectory, "data")
# data processing directory
makecar(parentDirectory, "ParamtOleajSensoresR")
dirparolejsenr = os.path.join(parentDirectory, "ParamtOleajSensoresR")

# names of calibrated sensor datasets
config = {
    'lis_caso': ['H10T12', 'H10T25', 'H10T30'],
    'caso': ['Reg']
}

name_lis_caso = {i: name for i, name in enumerate(config['lis_caso'])}
name_caso = {i: sea for i, sea in enumerate(config['caso'])}

# x sensors
x1 = np.array([0, 1.42, 0.93, 0.84, 0.20, 0.19, 0.20, 0.195, 0.20, 0.20, 0.23, 0.20, 0.20, 0.21])

x1[2:] = np.cumsum(x1[1:-1]) + x1[2:]
x2 = x1[1:] - 0.05
x3 = x1[1:] - 0.10

x_sensores = np.hstack((x1[1:], x2, x3))

for tipo in ['Reg']:
    print('----------------------------------------------------------------')
    print(f"Processing data --> {tipo}")

    # Prepare results directory
    makecar(dirparolejsenr, 'ML')
    tipo_dir = os.path.join(dirparolejsenr, 'ML')
    run_time = datetime.now().isoformat(timespec="seconds")

    # Load data for all cases and ensure DataFrame
    df_lab = []
    for i in range(3):
        data_i = load_and_process_data(dirparolejsenr, name_lis_caso[i], tipo)
        if isinstance(data_i, pd.DataFrame):
            df_i = data_i
        else:
            df_i = pd.DataFrame(data_i)
        df_lab.append(df_i)

    # Define columns to analyze
    col_defs = [(4, 'H [m]')]
    # if tipo == 'Reg':
    #     col_defs.append((2, 'eta_bar [m]'))

    # ML: classification of states (low/medium/high) and regression (height/level)

    # Merge data and clean
    df_all = pd.concat(df_lab, axis=0, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan)

    # Select continuous target
    target_candidates = [nm for _, nm in col_defs]  # e.g., ['H [m]', 'eta_bar [m]'] if present

    y_cont_col = pick_target(df_all, target_candidates)
    if y_cont_col is None:
        raise RuntimeError("No numeric column found to use as target.")

    # Numeric features
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    X = df_all[num_cols].copy()
    y_reg = df_all[y_cont_col].astype(float)

    # Drop rows without target
    mask = y_reg.notna()
    X = X.loc[mask]
    y_reg = y_reg.loc[mask]

    # If target column is in X, drop it from features
    if y_cont_col in X.columns:
        X = X.drop(columns=[y_cont_col])

    # Safety: if no features, abort
    if X.shape[1] == 0:
        raise RuntimeError("Not enough numeric columns to train models.")

    # ------------------------------------------------------------
    # Classification: wave states (low/medium/high) from y_reg (quantiles)
    labels = ["low", "medium", "high"]
    try:
        y_cls = pd.qcut(y_reg, q=3, labels=labels, duplicates="drop")
    except ValueError:
        # fallback if not enough variation
        y_cls = pd.cut(y_reg, bins=3, labels=labels)

    if y_cls.nunique() >= 2:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )

        cls_imputer = SimpleImputer(strategy="median")
        # k-NN
        knn_clf = Pipeline(
            steps=[
                ("imputer", cls_imputer),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        )
        knn_clf.fit(Xc_train, yc_train)
        yc_pred_knn = knn_clf.predict(Xc_test)

        # Decision Tree
        tree_clf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        )
        tree_clf.fit(Xc_train, yc_train)
        yc_pred_tree = tree_clf.predict(Xc_test)

        # Metrics
        acc_knn = round(accuracy_score(yc_test, yc_pred_knn), 3)
        f1_knn = round(f1_score(yc_test, yc_pred_knn, average="weighted"), 3)
        acc_tree = round(accuracy_score(yc_test, yc_pred_tree), 3)
        f1_tree = round(f1_score(yc_test, yc_pred_tree, average="weighted"), 3)

        print("Classification k-NN -> acc:", acc_knn, "| f1:", f1_knn)
        print("Classification Tree -> acc:", acc_tree, "| f1:", f1_tree)

        # Save classification metrics
        cls_metrics = pd.DataFrame(
            [
                {"run_time": run_time, "tipo": tipo, "model": "knn", "accuracy": acc_knn, "f1_weighted": f1_knn, "n_test": int(len(yc_test))},
                {"run_time": run_time, "tipo": tipo, "model": "tree", "accuracy": acc_tree, "f1_weighted": f1_tree, "n_test": int(len(yc_test))},
            ]
        )
        try:
            cls_metrics.to_csv(os.path.join(tipo_dir, f"{tipo}_classification_metrics.csv"), index=False)
        except Exception as e:
            print(f"Warn: could not save classification metrics: {e}")

        # Confusion matrix (k-NN)
        cm = confusion_matrix(yc_test, yc_pred_knn, labels=labels)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion matrix k-NN ({tipo})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(dirimg, f"{tipo}_cls_knn_cm.png"), dpi=150)
        finally:
            plt.close()
        # Save confusion matrix data
        try:
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(tipo_dir, f"{tipo}_cls_knn_cm.csv"))
        except Exception as e:
            print(f"Warn: could not save k-NN confusion matrix CSV: {e}")

        # Confusion matrix (Tree)
        cm_t = confusion_matrix(yc_test, yc_pred_tree, labels=labels)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm_t, annot=True, fmt="d", cmap="Greens",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion matrix Tree ({tipo})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(dirimg, f"{tipo}_cls_tree_cm.png"), dpi=150)
        finally:
            plt.close()
        # Save confusion matrix data
        try:
            pd.DataFrame(cm_t, index=labels, columns=labels).to_csv(os.path.join(tipo_dir, f"{tipo}_cls_tree_cm.csv"))
        except Exception as e:
            print(f"Warn: could not save Tree confusion matrix CSV: {e}")

        # Save tree feature importances (classification)
        try:
            importances_cls = tree_clf.named_steps["model"].feature_importances_
            feat_imp_cls = pd.DataFrame({"feature": Xc_train.columns, "importance": importances_cls}).sort_values("importance", ascending=False)
            feat_imp_cls.to_csv(os.path.join(tipo_dir, f"{tipo}_cls_tree_feature_importances.csv"), index=False)
        except Exception as e:
            print(f"Warn: could not save classification feature importances: {e}")
    else:
        print("Classification skipped: insufficient distinct classes in target.")
        try:
            with open(os.path.join(tipo_dir, f"{tipo}_classification_skipped.json"), "w", encoding="utf-8") as f:
                json.dump({"run_time": run_time, "tipo": tipo, "reason": "insufficient distinct classes"}, f)
        except Exception as e:
            print(f"Warn: could not save classification skipped info: {e}")

    # ------------------------------------------------------------
    # Regression: predict continuous variable (y_cont_col)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # Linear
    lin_reg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )
    lin_reg.fit(Xr_train, yr_train)
    yr_pred_lin = lin_reg.predict(Xr_test)

    # Decision Tree Regressor
    tree_reg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(max_depth=6, random_state=42)),
        ]
    )
    tree_reg.fit(Xr_train, yr_train)
    yr_pred_tree = tree_reg.predict(Xr_test)

    # Metrics
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    r2_lin = r2_score(yr_test, yr_pred_lin)
    rmse_lin = rmse(yr_test, yr_pred_lin)
    r2_tree = r2_score(yr_test, yr_pred_tree)
    rmse_tree = rmse(yr_test, yr_pred_tree)

    print(f"Linear Regression -> R2: {r2_lin:.3f} | RMSE: {rmse_lin:.4f}")
    print(f"Tree Regression   -> R2: {r2_tree:.3f} | RMSE: {rmse_tree:.4f}")

    # Save regression metrics
    reg_metrics = pd.DataFrame(
        [
            {"run_time": run_time, "tipo": tipo, "target": y_cont_col, "model": "linear", "r2": r2_lin, "rmse": rmse_lin, "n_test": int(len(yr_test))},
            {"run_time": run_time, "tipo": tipo, "target": y_cont_col, "model": "tree", "r2": r2_tree, "rmse": rmse_tree, "n_test": int(len(yr_test))},
        ]
    )
    try:
        reg_metrics.to_csv(os.path.join(tipo_dir, f"{tipo}_regression_metrics.csv"), index=False)
    except Exception as e:
        print(f"Warn: could not save regression metrics: {e}")

    # Save predictions (regression)
    try:
        pred_df = pd.DataFrame(
            {
                f"y_true_{y_cont_col}": yr_test.values,
                "y_pred_linear": yr_pred_lin,
                "y_pred_tree": yr_pred_tree,
            }
        )
        pred_df.to_csv(os.path.join(tipo_dir, f"{tipo}_regression_predictions.csv"), index=False)
    except Exception as e:
        print(f"Warn: could not save regression predictions: {e}")

    # Save tree feature importances (regression)
    try:
        importances_reg = tree_reg.named_steps["model"].feature_importances_
        feat_imp_reg = pd.DataFrame({"feature": Xr_train.columns, "importance": importances_reg}).sort_values("importance", ascending=False)
        feat_imp_reg.to_csv(os.path.join(tipo_dir, f"{tipo}_reg_tree_feature_importances.csv"), index=False)
    except Exception as e:
        print(f"Warn: could not save regression feature importances: {e}")

    # Plot predicted vs actual
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300, facecolor='w', edgecolor='k')
    fig.suptitle(f"Predicted vs Actual \n ({tipo}) \n", fontsize=18, y=0.95)

    ax.scatter(yr_test, yr_pred_lin, s=150, edgecolor='w', zorder=3, alpha=0.6, label="Linear")
    ax.scatter(yr_test, yr_pred_tree, s=150, edgecolor='w', zorder=3, alpha=0.6, label="Tree")

    m = [min(yr_test.min(), yr_pred_lin.min(), yr_pred_tree.min()),
         max(yr_test.max(), yr_pred_lin.max(), yr_pred_tree.max())]

    ax.plot(m, m, "k--", lw=1)

    ax.set_xlabel(f"Actual ({y_cont_col})", fontsize=18)
    ax.set_ylabel("Predicted", fontsize=18)

    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    ax.legend()
    # ax.tight_layout()

    try:
        plt.savefig(os.path.join(dirimg, f"{tipo}_reg_pred_vs_real.png"))
    finally:
        plt.close()

