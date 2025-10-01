# scripts/03_run_model.py
"""
Fit a Bayesian hierarchical GLMM for fantasy points (World Cup dataset).
- Loads: data/processed/player_match_wc.csv
- Produces: data/model_output/wc_model.nc (ArviZ InferenceData)
- Saves posterior predictive samples to data/model_output/ppc_samples.npy (numpy array)
Notes:
- Requires 'pymc' and 'arviz' installed in your venv.
- Adjust sampling settings (draws/tune/chains) for speed or thoroughness.
"""
import os
import sys
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

# Config
DATA_CSV = "data/processed/player_match_wc.csv"
MODEL_DIR = "data/model_output"
os.makedirs(MODEL_DIR, exist_ok=True)
INFERENCE_DATA_PATH = os.path.join(MODEL_DIR, "wc_model.nc")
PPC_SAVE_PATH = os.path.join(MODEL_DIR, "ppc_samples.npy")

# Sampling configuration (adjust to your machine)
SAMPLE_CONFIG = {
    "draws": 1000,     # posterior draws to keep (after tuning)
    "tune": 1000,      # tuning steps
    "chains": 4,
    "target_accept": 0.9,
    # "cores": 4        # leave default (PyMC will pick available cores)
}

def load_and_preprocess(path=DATA_CSV, min_minutes=0):
    """
    Load CSV and prepare arrays for PyMC model.

    - By default min_minutes=0 so we do not drop players with 0 minutes (World Cup dataset had many zero/missing).
    - Fills missing opponent_strength with tournament median if available, else 0.0.
    - Returns: df, counts, home, opp_strength, pos_idx, player_idx, team_idx, meta
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found: {path}. Run feature engineering first.")
    df = pd.read_csv(path)
    print("Loaded rows:", len(df))

    # Ensure fantasy_points exists and coerce to integer counts
    if "fantasy_points" not in df.columns:
        raise KeyError("fantasy_points column not found in processed CSV.")
    df = df[pd.notna(df["fantasy_points"])]
    df["fantasy_points"] = df["fantasy_points"].astype(float).round().astype(int)

    # Fill missing minutes with 0 (defensive)
    if "minutes" not in df.columns:
        df["minutes"] = 0
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)

    # Optionally filter by minutes (default min_minutes=0 keeps all)
    if min_minutes > 0:
        before = len(df)
        df = df[df["minutes"] >= min_minutes]
        print(f"Filtered minutes < {min_minutes}: {before} -> {len(df)}")

    # Handle opponent_strength: fill with tournament median if possible, else 0.0
    if "opponent_strength" in df.columns:
        if df["opponent_strength"].dropna().shape[0] > 0:
            med = df["opponent_strength"].median()
            df["opponent_strength"] = df["opponent_strength"].fillna(med)
        else:
            df["opponent_strength"] = 0.0
    else:
        df["opponent_strength"] = 0.0

    # If after filters the dataframe is empty, stop with a clear message
    if df.shape[0] == 0:
        raise RuntimeError("No rows remain after preprocessing. Check `minutes` filtering or your processed CSV.")

    # Encode IDs and positions
    df["player_id_str"] = df["player_id"].astype(str)
    df["team_str"] = df["team"].astype(str)

    df["position_filled"] = df["position"].fillna("M").astype(str).str.upper().str[0]

    df["player_idx"] = df["player_id_str"].astype("category").cat.codes
    df["team_idx"] = df["team_str"].astype("category").cat.codes
    df["pos_idx"] = df["position_filled"].astype("category").cat.codes

    player_map = dict(enumerate(df["player_id_str"].astype("category").cat.categories))
    team_map = dict(enumerate(df["team_str"].astype("category").cat.categories))
    pos_map = dict(enumerate(df["position_filled"].astype("category").cat.categories))

    # Standardize opponent_strength for numerical stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # scaler.fit_transform expects at least 1 row; df is non-empty here
    df["opp_strength_scaled"] = scaler.fit_transform(df[["opponent_strength"]].astype(float))

    counts = df["fantasy_points"].values
    home = df["home"].fillna(0).astype(int).values
    opp_strength = df["opp_strength_scaled"].astype(float).values
    pos_idx = df["pos_idx"].astype(int).values

    meta = {
        "n_players": int(df["player_idx"].nunique()),
        "n_teams": int(df["team_idx"].nunique()),
        "n_pos": int(df["pos_idx"].nunique()),
        "n_obs": len(df),
        "player_map": player_map,
        "team_map": team_map,
        "pos_map": pos_map,
        "scaler_mean": float(scaler.mean_[0]),
        "scaler_scale": float(scaler.scale_[0])
    }

    return df, counts, home, opp_strength, pos_idx, df["player_idx"].astype(int).values, df["team_idx"].astype(int).values, meta

def check_overdispersion(counts):
    mean = counts.mean()
    var = counts.var()
    print(f"Counts mean = {mean:.3f}, variance = {var:.3f}")
    if var > mean + 1e-6:
        print("Data appears overdispersed (variance > mean). Negative-Binomial is recommended.")
        return True
    else:
        print("Poisson may be adequate (variance <= mean).")
        return False

def build_and_sample(counts, player_idx, team_idx, home, opp_strength, pos_idx, meta, use_nb=True):
    n_players = meta["n_players"]
    n_teams = meta["n_teams"]
    n_pos = meta["n_pos"]

    with pm.Model() as model:
        # Hyperpriors for random effects scales
        sigma_player = pm.HalfNormal("sigma_player", sigma=1.0)
        sigma_team = pm.HalfNormal("sigma_team", sigma=1.0)

        # Player and team random intercepts (non-centered)
        player_offset = pm.Normal("player_offset", mu=0.0, sigma=1.0, shape=n_players)
        player_effect = pm.Deterministic("player_effect", player_offset * sigma_player)

        team_offset = pm.Normal("team_offset", mu=0.0, sigma=1.0, shape=n_teams)
        team_effect = pm.Deterministic("team_effect", team_offset * sigma_team)

        # Global intercept and fixed effects
        intercept = pm.Normal("intercept", mu=0.0, sigma=5.0)
        beta_home = pm.Normal("beta_home", mu=0.0, sigma=1.0)
        beta_opp = pm.Normal("beta_opp", mu=0.0, sigma=1.0)
        # position effects as a small vector
        beta_pos = pm.Normal("beta_pos", mu=0.0, sigma=1.0, shape=n_pos)

        # linear predictor on log scale
        eta = (
            intercept
            + beta_home * home
            + beta_opp * opp_strength
            + beta_pos[pos_idx]
            + player_effect[player_idx]
            + team_effect[team_idx]
        )
        mu = pm.math.exp(eta)

        if use_nb:
            # Negative binomial parametrization with alpha (overdispersion)
            alpha = pm.Exponential("alpha", 1.0)
            obs = pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=counts)
        else:
            obs = pm.Poisson("obs", mu=mu, observed=counts)

        print("Starting sampling with config:", SAMPLE_CONFIG)
        idata = pm.sample(**SAMPLE_CONFIG, return_inferencedata=True)
        # posterior predictive
            # posterior predictive
        # posterior predictive
    ppc = pm.sample_posterior_predictive(idata, model=model, var_names=["obs"], random_seed=42)


    # Try multiple ways to obtain obs_samples (n_draws x n_obs)
    obs_samples = None

    # 1) ppc dict from pm.sample_posterior_predictive() often returns dict with key 'obs'
    if isinstance(ppc, dict) and "obs" in ppc:
        obs_samples = np.array(ppc["obs"])  # shape (n_draws, n_obs) or (n_chains, n_draws, n_obs) depending on pymc version
        # Flatten chain dimension if present
        if obs_samples.ndim == 3:
            # (chains, draws, n_obs) -> (chains*draws, n_obs)
            obs_samples = obs_samples.reshape(-1, obs_samples.shape[-1])
        print("Posterior predictive obtained from ppc dict; shape:", obs_samples.shape)

    # 2) If ppc doesn't have 'obs', check idata.posterior_predictive (ArviZ InferenceData)
    elif hasattr(idata, "posterior_predictive") and "obs" in idata.posterior_predictive:
        # idata.posterior_predictive['obs'] shape is (chain, draw, obs) or (draw, obs)
        arr = idata.posterior_predictive["obs"].values
        if arr.ndim == 3:
            obs_samples = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            obs_samples = arr
        print("Posterior predictive obtained from idata.posterior_predictive; shape:", obs_samples.shape)

    # 3) If still none, try to salvage any array in ppc dict (take first array-like value)
    elif isinstance(ppc, dict) and len(ppc) > 0:
        # pick the first key
        first_key = list(ppc.keys())[0]
        try:
            arr = np.array(ppc[first_key])
            if arr.ndim == 3:
                obs_samples = arr.reshape(-1, arr.shape[-1])
            elif arr.ndim == 2:
                obs_samples = arr
            print(f"Posterior predictive taken from ppc['{first_key}']; shape:", obs_samples.shape)
        except Exception:
            obs_samples = None

    # Save results robustly (if any)
    if obs_samples is not None:
        np.save(PPC_SAVE_PATH, obs_samples)
        print("Saved posterior predictive samples to:", PPC_SAVE_PATH, " shape:", obs_samples.shape)
        ppc_mean = float(obs_samples.mean())
    else:
        print("Posterior predictive didn't return usable samples (no 'obs' key and idata.posterior_predictive absent).")
        obs_samples = None
        ppc_mean = float("nan")

    return model, idata, ppc

def main():
    print("Loading & preprocessing data...")
    df, counts, home, opp_strength, pos_idx, player_idx, team_idx, meta = load_and_preprocess(DATA_CSV)
    print("Meta:", {k: v for k, v in meta.items() if k not in ('player_map','team_map','pos_map')})
    print("unique players:", meta["n_players"], "unique teams:", meta["n_teams"], "unique positions:", meta["n_pos"])

    use_nb = check_overdispersion(counts)

    print("Building model and sampling...")
    model, idata, ppc = build_and_sample(counts, player_idx, team_idx, home, opp_strength, pos_idx, meta, use_nb=use_nb)

    # Save inference data
    print("Saving inference data to:", INFERENCE_DATA_PATH)
    az.to_netcdf(idata, INFERENCE_DATA_PATH)

    # Save posterior predictive obs samples as numpy array (n_draws, n_obs)
    if "obs" in ppc:
        obs_samples = np.array(ppc["obs"])
        np.save(PPC_SAVE_PATH, obs_samples)
        print("Saved posterior predictive samples to:", PPC_SAVE_PATH, " shape:", obs_samples.shape)

    # Print concise diagnostics
    print("\nPosterior summary (selected):")
    print(az.summary(idata, var_names=["intercept", "beta_home", "beta_opp", "sigma_player", "sigma_team", "alpha"], round_to=3))

    # Posterior predictive check: compare observed vs predicted mean
    
    obs_mean = counts.mean()
    if obs_samples is not None:
        ppc_mean = obs_samples.mean()
        print(f"Posterior predictive mean (avg over draws) = {ppc_mean:.3f}")
    else:
        print("Posterior predictive samples unavailable; skipping PPC mean.")
    
    print(f"\nObserved mean fantasy points = {obs_mean:.3f}")
    print(f"Posterior predictive mean (avg over draws) = {ppc_mean:.3f}")

    # Save a tiny CSV of player random effects (posterior means) for quick ranking
    try:
        player_effect_mean = idata.posterior["player_effect"].mean(dim=["chain", "draw"]).values
        # align with meta player_map keys (player_effect_mean is length n_players)
        pe_df = pd.DataFrame({
            "player_idx": np.arange(len(player_effect_mean)),
            "player_id": [meta["player_map"][i] for i in range(len(player_effect_mean))],
            "player_effect_mean": player_effect_mean
        })
        pe_df = pe_df.sort_values("player_effect_mean", ascending=False)
        pe_df.to_csv(os.path.join(MODEL_DIR, "player_effects_summary.csv"), index=False)
        print("Saved player_effects_summary.csv (posterior means).")
    except Exception as e:
        print("Could not compute/save player_effects_summary:", e)

    print("\nModel run complete. Files saved under", MODEL_DIR)
    print("To inspect results interactively, open a Python REPL and run:\nimport arviz as az\nidata = az.from_netcdf('" + INFERENCE_DATA_PATH + "')\naz.plot_trace(idata)\naz.plot_ppc(idata)\n")

if __name__ == "__main__":
    main()
