import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import joblib
import dagshub

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 0. CONFIG DAGSHUB MLFLOW
# ==========================================
RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
N_TRIALS = 20

# 1. Inisialisasi DagsHub (Ganti dengan Username dan Nama Repo Anda)
# Baris ini otomatis akan mengatur mlflow.set_tracking_uri ke DagsHub
dagshub.init(repo_owner='wriyadi5', repo_name='student_performance', mlflow=True)

# 2. Set Eksperimen
mlflow.set_experiment("Student_CGPA_Nested_CV")

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv('Students_Performance_dataset.csv')

target = 'What is your current CGPA?'
X = df.drop(columns=[target])
y = df[target]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ]
)

# ==========================================
# 2. NESTED CV
# ==========================================
def run_nested_cv():
    outer_cv = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    outer_results = []
    best_params_all_folds = []

    with mlflow.start_run(run_name="Nested_CV_RF"):

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
            print(f"\n🔥 Outer Fold {fold}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # ==========================================
            # INNER OPTUNA
            # ==========================================
            def objective(trial):

                n_components = trial.suggest_int('pca_n_components', 5, min(25, X_train.shape[1]))
                rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300, step=50)
                rf_max_depth = trial.suggest_int('rf_max_depth', 3, 15)

                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('pca', PCA(n_components=n_components)),
                    ('rf', RandomForestRegressor(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    ))
                ])

                score = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=N_INNER_SPLITS,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                ).mean()

                return np.sqrt(-score)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=N_TRIALS)

            best_params = study.best_trial.params
            best_params_all_folds.append(best_params)

            # ==========================================
            # TRAIN BEST MODEL
            # ==========================================
            best_model = Pipeline([
                ('preprocessor', preprocessor),
                ('pca', PCA(n_components=best_params['pca_n_components'])),
                ('rf', RandomForestRegressor(
                    n_estimators=best_params['rf_n_estimators'],
                    max_depth=best_params['rf_max_depth'],
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ])

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            smape = np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "rmse": rmse,
                "mae": mae,
                "smape": smape,
                "r2": r2
            }

            outer_results.append(metrics)

            mlflow.log_metrics({f"{k}_fold_{fold}": v for k, v in metrics.items()})
            mlflow.log_params({f"{k}_fold_{fold}": v for k, v in best_params.items()})

            print(f"Fold {fold} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        # ==========================================
        # FINAL METRICS
        # ==========================================
        avg_metrics = {
            "avg_rmse": np.mean([m["rmse"] for m in outer_results]),
            "avg_mae": np.mean([m["mae"] for m in outer_results]),
            "avg_smape": np.mean([m["smape"] for m in outer_results]),
            "avg_r2": np.mean([m["r2"] for m in outer_results]),
        }

        mlflow.log_metrics(avg_metrics)

        print("\n=== FINAL RESULT ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

    return best_params_all_folds

# ==========================================
# 3. FINAL MODEL (NO DATA LEAKAGE)
# ==========================================
def train_final_model(best_params_all_folds):

    print("\n🚀 Training FINAL model (Optuna on FULL data)...")

    def objective_final(trial):

        n_components = trial.suggest_int('pca_n_components', 5, min(25, X.shape[1]))
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300, step=50)
        rf_max_depth = trial.suggest_int('rf_max_depth', 3, 15)

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=n_components)),
            ('rf', RandomForestRegressor(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        score = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        ).mean()

        return np.sqrt(-score)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_final, n_trials=30)

    best_params = study.best_trial.params

    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=best_params['pca_n_components'])),
        ('rf', RandomForestRegressor(
            n_estimators=best_params['rf_n_estimators'],
            max_depth=best_params['rf_max_depth'],
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    final_model.fit(X, y)

    joblib.dump(final_model, "final_model.pkl")

    mlflow.sklearn.log_model(
    sk_model=final_model,
    name="final_model")

    print("✅ Model final saved to DagsHub & Local!")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    best_params_all_folds = run_nested_cv()
    train_final_model(best_params_all_folds)