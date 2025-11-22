import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.calibration import CalibratedClassifierCV


# =====================================================
# 0) CONFIGURACIÓN GENERAL
# =====================================================
# Rutas a tus CSV (ajusta según tus archivos)
CSV_AAPL = r"aapl.csv"
CSV_SPY  = r"spy.csv"
CSV_QQQ  = r"qqq.csv"
CSV_VIX  = r"vix.csv"
CSV_TNX  = r"tnx.csv"

START_WF = 300     # mínimo histórico para empezar walk-forward
EPS_3D   = 0.0025  # 0.25% para horizonte 3 días
EPS_5D   = 0.0040  # 0.40% para horizonte 5 días

ADX_THR_BASE = 20  # umbral básico de ADX para considerar tendencia


# =====================================================
# 1) INDICADORES BÁSICOS (SIN LEAKAGE)
# =====================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = gain_ema / (loss_ema + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(h, l, c):
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(h, l, c, window=14):
    tr = true_range(h, l, c)
    return tr.rolling(window).mean()

def adx(h, l, c, window=14):
    up_move   = h.diff()
    down_move = -l.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(h, l, c)
    tr_n = tr.rolling(window).sum()
    plus_di  = 100 * (pd.Series(plus_dm, index=h.index).rolling(window).sum() / (tr_n + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=h.index).rolling(window).sum() / (tr_n + 1e-12))

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12) ) * 100
    adx_val = dx.rolling(window).mean()
    return adx_val

def bollinger_bands(c, window=20, num_std=2):
    ma = c.rolling(window).mean()
    std = c.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / (ma + 1e-12)
    return ma, upper, lower, width

def obv(c, v):
    direction = np.sign(c.diff().fillna(0))
    obv_series = (direction * v).cumsum()
    return obv_series

def stochastic_kd(h, l, c, window=14):
    """
    Calcula el Stochastic Oscillator %K y %D

    h  : pd.Series -> precios máximos
    l  : pd.Series -> precios mínimos
    c  : pd.Series -> precios de cierre
    window : int    -> período (usualmente 14 días)

    return : (stochastic_k, stochastic_d) -> %K y %D
    """
    # Cálculo de %K con protección contra NaN
    lowest_low = l.rolling(window).min()  # Minimo de los lows en el periodo de `window`
    highest_high = h.rolling(window).max()  # Máximo de los highs en el periodo de `window`
    
    # Evitar NaN dividiendo por cero
    stochastic_k = 100 * (c - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)

    # Cálculo de %D (media móvil de %K, suaviza el %K)
    stochastic_d = stochastic_k.rolling(3).mean()  # Usualmente se utiliza una media de 3 días

    return stochastic_k, stochastic_d


# =====================================================
# 2) CARGA SERIES MACRO (SPY, QQQ, VIX, TNX)
# =====================================================

def load_macro_series():
    def _load_one(path, col_close, new_name):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No existe CSV: {p}")
        df = pd.read_csv(p, parse_dates=["Date"])
        df = df.rename(columns={
            "date": "Date", "Date": "Date",
            "Adj Close": col_close, "Close": col_close, "close": col_close
        })
        df = df[["Date", col_close]].copy()
        df = df.sort_values("Date").reset_index(drop=True)
        df = df.rename(columns={col_close: new_name})
        return df

    spy = _load_one(CSV_SPY, "Close", "spy_close")
    qqq = _load_one(CSV_QQQ, "Close", "qqq_close")
    vix = _load_one(CSV_VIX, "Close", "vix_close")
    tnx = _load_one(CSV_TNX, "Close", "tnx_close")

    return spy, qqq, vix, tnx


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    spy, qqq, vix, tnx = load_macro_series()

    out = df.merge(spy, on="Date", how="left")
    out = out.merge(qqq, on="Date", how="left")
    out = out.merge(vix, on="Date", how="left")
    out = out.merge(tnx, on="Date", how="left")

    # SPY trend & volatility
    out["spy_ema200"] = ema(out["spy_close"], 200)
    out["spy_trend"] = (out["spy_close"] - out["spy_ema200"]) / (out["spy_ema200"] + 1e-12)
    out["spy_vol20"] = out["spy_close"].pct_change().rolling(20).std()

    # QQQ trend
    out["qqq_ema200"] = ema(out["qqq_close"], 200)
    out["qqq_trend"] = (out["qqq_close"] - out["qqq_ema200"]) / (out["qqq_ema200"] + 1e-12)

    # VIX trend
    out["vix_ema20"] = ema(out["vix_close"], 20)
    out["vix_trend"] = (out["vix_close"] - out["vix_ema20"]) / (out["vix_ema20"] + 1e-12)

    # TNX (10Y yield) trend
    out["tnx_ema20"] = ema(out["tnx_close"], 20)
    out["tnx_trend"] = (out["tnx_close"] - out["tnx_ema20"]) / (out["tnx_ema20"] + 1e-12)

    # Lagg de todo lo macro para evitar leak
    macro_cols_to_lag = [
        "spy_trend", "spy_vol20",
        "qqq_trend",
        "vix_trend",
        "tnx_trend"
    ]
    for col in macro_cols_to_lag:
        out[col] = out[col].shift(1)

    return out


# =====================================================
# 3) FEATURES DE AAPL (PRECIO)
# =====================================================

def compute_features_aapl(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    v = df["Volume"]

    # Rangos y retornos
    hl_range = h - l
    ret1 = o.pct_change(1)
    ret5 = o.pct_change(5)
    ret10 = o.pct_change(10)

    # Volatilidad y volumen
    atr14 = atr(h, l, c, 14)
    std10 = ret1.rolling(10).std()
    std20 = ret1.rolling(20).std()
    z_volume = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)

    # Tendencia EMAs
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    ema20_slope = ema20.diff()

    # MACD
    macd_line = ema(c, 12) - ema(c, 26)
    macd_signal = ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    # Momentum
    rsi14 = rsi(c, 14)
    stoch_k, stoch_d = stochastic_kd(h, l, c, 14)  # Aquí se calculan %K y %D
    roc10 = c.pct_change(10)

    # Bollinger
    bb_ma, bb_upper, bb_lower, bb_width = bollinger_bands(c, 20, 2)

    # Volumen OBV
    obv_series = obv(c, v)
    obv_change = obv_series.diff()

    # ADX
    adx14 = adx(h, l, c, 14)

    # Features especiales AAPL: distancia a EMAs
    dist_ema20 = (c - ema20) / (ema20 + 1e-12)
    dist_ema50 = (c - ema50) / (ema50 + 1e-12)

    # TODO laggeado (shift(1))
    feats = pd.DataFrame({
        "ret1": ret1.shift(1),
        "ret5": ret5.shift(1),
        "ret10": ret10.shift(1),
        "hl_range": hl_range.shift(1),
        "atr14": atr14.shift(1),
        "std10": std10.shift(1),
        "std20": std20.shift(1),
        "z_volume": z_volume.shift(1),

        "ema20": ema20.shift(1),
        "ema50": ema50.shift(1),
        "ema200": ema200.shift(1),
        "ema20_slope": ema20_slope.shift(1),

        "macd_line": macd_line.shift(1),
        "macd_signal": macd_signal.shift(1),
        "macd_hist": macd_hist.shift(1),

        "rsi14": rsi14.shift(1),
        "stoch_k": stoch_k.shift(1),  #
        "stoch_d": stoch_d.shift(1),  #
        "roc10": roc10.shift(1),

        "bb_ma": bb_ma.shift(1),
        "bb_width": bb_width.shift(1),

        "obv": obv_series.shift(1),
        "obv_change": obv_change.shift(1),

        "adx14": adx14.shift(1),
        "dist_ema20": dist_ema20.shift(1),
        "dist_ema50": dist_ema50.shift(1),
    }, index=df.index)

    return feats


# =====================================================
# 4) LABELS MULTI-HORIZONTE (3D y 5D)
# =====================================================

def compute_labels_multi(df: pd.DataFrame,
                         eps_3d: float = EPS_3D,
                         eps_5d: float = EPS_5D) -> pd.DataFrame:
    o = df["Open"]
    ret_3d = (o.shift(-3) - o) / o
    ret_5d = (o.shift(-5) - o) / o
    ret_next = (o.shift(-1) - o) / o

    def dir_from_ret(ret, eps):
        y = pd.Series(0, index=ret.index)
        y[ret > eps] = 1
        y[ret < -eps] = -1
        return y

    y_dir_3d = dir_from_ret(ret_3d, eps_3d)
    y_dir_5d = dir_from_ret(ret_5d, eps_5d)

    labels = pd.DataFrame({
        "ret_3d": ret_3d,
        "y_dir_3d": y_dir_3d,
        "ret_5d": ret_5d,
        "y_dir_5d": y_dir_5d,
        "ret_next": ret_next
    }, index=df.index)

    return labels


# =====================================================
# 5) CARGA AAPL + FEATURES + MACRO + LABELS
# =====================================================

def load_and_prepare_aapl() -> pd.DataFrame:
    p = Path(CSV_AAPL)
    if not p.exists():
        raise FileNotFoundError(f"No existe archivo AAPL: {p}")

    df = pd.read_csv(p, parse_dates=["Date"])
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
        "Adj Close": "Close"
    })
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    feats_price = compute_features_aapl(df)
    labels = compute_labels_multi(df)

    out = pd.concat([df, feats_price, labels], axis=1)
    out = add_macro_features(out)

    # quitar filas sin futuro a 5 días
    out = out.iloc[:-5].copy()
    out = out.dropna().reset_index(drop=True)

    return out


# =====================================================
# 6) ENCODE + WALK-FORWARD CATBOOST TRINARIO
# =====================================================

def encode_trinary(y_dir: pd.Series):
    mapping = {-1: 0, 0: 1, 1: 2}
    return y_dir.map(mapping).astype(int), mapping

def walk_forward_catboost(X, y_trinary, start=300, model_save_path="catboost_model.cbm"):
    n = len(y_trinary)
    proba = np.full((n, 3), np.nan)

    for t in tqdm(range(start, n), desc="WF CatBoost"):
        X_tr, y_tr = X[:t], y_trinary[:t]
        X_te = X[t:t+1]

        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
            task_type="GPU",  # GPU
            devices="0",       # Número de la GPU, "0" para la primera GPU
            early_stopping_rounds=50,
            bagging_temperature=0.2,
        )
        model.fit(X_tr, y_tr)

        # Guardar el modelo
        model.save_model(model_save_path)  # guardará el modelo entrenado

        p = model.predict_proba(X_te)[0]
        proba[t, :] = p

    return proba


# =====================================================
# 7) BUSCAR UMBRALES (OOS) PARA UP/DOWN
# =====================================================

def best_thr_binary(y_true_binary, p, lo=0.3, hi=0.7, steps=81):
    grid = np.linspace(lo, hi, steps)
    best_t, best_mcc = 0.5, -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        mcc = matthews_corrcoef(y_true_binary, pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t, best_mcc


# =====================================================
# 8) RÉGIMEN DE MERCADO + TAMAÑO DE POSICIÓN
# =====================================================

def classify_regime(row):
    """
    Simple régimen:
      0 → risk-off    (SPY muy debajo de EMA200 o VIX alto)
      1 → neutral
      2 → risk-on     (SPY en tendencia alcista y VIX tranquilo)
    """
    spy_tr = row["spy_trend"]
    vix_tr = row["vix_trend"]
    qqq_tr = row["qqq_trend"]

    if np.isnan(spy_tr) or np.isnan(vix_tr):
        return 1  # neutral

    # risk-off: SPY muy por debajo de EMA200 o VIX muy arriba de EMA20
    if spy_tr < -0.03 or vix_tr > 0.20:
        return 0

    # risk-on: SPY y QQQ por encima, VIX tranquilo
    if spy_tr > 0.01 and qqq_tr > 0.01 and vix_tr < 0.05:
        return 2

    return 1

def position_size(row):
    """
    Tamaño dinámico según ADX (tendencia) y ATR relativo (volatilidad).
    """
    adx = row["adx14"]
    atr_rel = abs(row["atr14"] / row["Close"]) if row["Close"] != 0 else 0.0
    regime = row["regime"]

    if np.isnan(adx) or np.isnan(atr_rel):
        return 0.0

    # sin régimen favorable → muy defensivo
    if regime == 0:
        return 0.0  # risk-off, no operar
    elif regime == 1:
        adx_offset = 0  # neutral
    else:  # regime == 2 (risk-on)
        adx_offset = 3  # bajamos umbral ADX un poco (más agresivo)

    # escalado por ADX
    adx_eff = adx - adx_offset

    if adx_eff < ADX_THR_BASE:
        base = 0.0
    elif adx_eff < ADX_THR_BASE + 4:
        base = 0.3
    elif adx_eff < ADX_THR_BASE + 8:
        base = 0.5
    elif adx_eff < ADX_THR_BASE + 12:
        base = 0.8
    else:
        base = 1.0

    # penalizar volatilidad extrema
    if atr_rel > 0.04:
        base *= 0.5

    return base


# =====================================================
# 9) BACKTEST + CÁLCULO DE MÉTRICAS
# =====================================================

def backtest_with_signals(df_valid):
    # clasificar régimen
    df_valid["regime"] = df_valid.apply(classify_regime, axis=1)

    # tamaño de posición dinámica
    df_valid["position_size"] = df_valid.apply(position_size, axis=1)

    # basamos la trading rule en ret_next (1 día) y señal t-1
    df_valid["strategy_return_dyn"] = (
        df_valid["signal_final"].shift(1) *
        df_valid["position_size"].shift(1) *
        df_valid["ret_next"]
    )

    df_bt = df_valid.dropna(subset=["strategy_return_dyn"]).copy()
    df_bt["equity"] = df_bt["strategy_return_dyn"].cumsum()

    win_rate = (df_bt["strategy_return_dyn"] > 0).mean()
    profit_factor = df_bt[df_bt["strategy_return_dyn"] > 0]["strategy_return_dyn"].sum() / \
                    abs(df_bt[df_bt["strategy_return_dyn"] < 0]["strategy_return_dyn"].sum() + 1e-12)

    df_bt["peak"] = df_bt["equity"].cummax()
    df_bt["dd"] = df_bt["equity"] - df_bt["peak"]
    max_dd = df_bt["dd"].min()

    sharpe = df_bt["strategy_return_dyn"].mean() / (df_bt["strategy_return_dyn"].std() + 1e-12) * np.sqrt(252)
    total_ret = df_bt["strategy_return_dyn"].sum() * 100
    n_trades = (df_bt["signal_final"] != 0).sum()

    print(f"\nRetorno total estrategia dyn (approx, %) = {total_ret:.2f}%")
    print(f"Win rate                                = {win_rate:.3f}")
    print(f"Profit factor                           = {profit_factor:.3f}")
    print(f"Max drawdown (%)                        = {max_dd*100:.2f}%")
    print(f"Sharpe ratio                            = {sharpe:.3f}")
    print(f"Nº de operaciones (no neutral)          = {n_trades}")

    # Equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(df_bt["Date"], df_bt["equity"])
    plt.title("Equity Curve - Estrategia AAPL v3 (pos. dinámica)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nÚltimas filas backtest:")
    print(df_bt[["Date", "p_up_ens", "p_down_ens", "signal_final",
                 "regime", "position_size", "strategy_return_dyn", "equity"]].tail())


# =====================================================
# 10) MAIN: WF 3D + WF 5D + ENSEMBLE + SPLIT 70/30 + OOS UMBRALES
# =====================================================

def main():
    df = load_and_prepare_aapl()
    print(f"[OK] Dataset AAPL preparado. Filas: {len(df)}")

    feature_cols = [
        "ret1", "ret5", "ret10",
        "hl_range", "atr14", "std10", "std20", "z_volume",
        "ema20", "ema50", "ema200", "ema20_slope",
        "macd_line", "macd_signal", "macd_hist",
        "rsi14", "stoch_k", "stoch_d", "roc10",
        "bb_ma", "bb_width",
        "obv", "obv_change",
        "adx14", "dist_ema20", "dist_ema50",
        "spy_trend", "spy_vol20",
        "qqq_trend",
        "vix_trend",
        "tnx_trend"
    ]

    X = df[feature_cols].values.astype("float32")

    # Trinario 3d y 5d
    y3_trinary, _ = encode_trinary(df["y_dir_3d"])
    y5_trinary, _ = encode_trinary(df["y_dir_5d"])

    # Walk-forward 3d
    print("\n=== Walk-forward 3D ===")
    proba3 = walk_forward_catboost(X, y3_trinary, start=START_WF)

    # Walk-forward 5d
    print("\n=== Walk-forward 5D ===")
    proba5 = walk_forward_catboost(X, y5_trinary, start=START_WF)

    df["p3_down"] = proba3[:, 0]
    df["p3_neutral"] = proba3[:, 1]
    df["p3_up"] = proba3[:, 2]

    df["p5_down"] = proba5[:, 0]
    df["p5_neutral"] = proba5[:, 1]
    df["p5_up"] = proba5[:, 2]

    # Ensemble
    W3 = 0.45
    W5 = 0.55

    df["p_up_ens"] = W3 * df["p3_up"] + W5 * df["p5_up"]
    df["p_down_ens"] = W3 * df["p3_down"] + W5 * df["p5_down"]

    # Trabajamos solo desde START_WF (donde hay proba válida)
    df_valid = df.iloc[START_WF:].copy()
    df_valid = df_valid.dropna(subset=["p_up_ens", "p_down_ens"]).reset_index(drop=True)

    # Split temporal 70/30 para calibrar umbrales fuera de muestra
    split_idx = int(len(df_valid) * 0.7)
    df_thr_train = df_valid.iloc[:split_idx].copy()
    df_thr_test = df_valid.iloc[split_idx:].copy()

    # UP: objetivo = y_dir_5d == 1
    y_up_train = (df_thr_train["y_dir_5d"] == 1).astype(int)
    thr_up, mcc_up = best_thr_binary(y_up_train, df_thr_train["p_up_ens"].values)
    print(f"\n[TRAIN] Mejor umbral UP (ens): {thr_up:.3f} | MCC = {mcc_up:.4f}")

    # DOWN: objetivo = y_dir_5d == -1
    y_down_train = (df_thr_train["y_dir_5d"] == -1).astype(int)
    thr_down, mcc_down = best_thr_binary(y_down_train, df_thr_train["p_down_ens"].values)
    print(f"[TRAIN] Mejor umbral DOWN (ens): {thr_down:.3f} | MCC = {mcc_down:.4f}")

    # Definimos señal usando SOLO estos umbrales (fijos) en TODO df_valid

    def signal_row(row):
        p_up = row["p_up_ens"]
        p_down = row["p_down_ens"]
        adx_val = row["adx14"]

        if np.isnan(p_up) or np.isnan(p_down) or np.isnan(adx_val):
            return 0

        if adx_val < ADX_THR_BASE:  # sin tendencia básica → no operar
            return 0

        if (p_up >= thr_up) and (p_up >= p_down):
            return 1
        if (p_down >= thr_down) and (p_down > p_up):
            return -1
        return 0

    df_valid["signal_final"] = df_valid.apply(signal_row, axis=1)

    # Evaluamos clasificación de dirección a 1 día (ret_next)
    df_valid["truth_1d"] = 0
    df_valid.loc[df_valid["ret_next"] > 0, "truth_1d"] = 1
    df_valid.loc[df_valid["ret_next"] < 0, "truth_1d"] = -1

    df_valid["hit_1d"] = (df_valid["signal_final"] == df_valid["truth_1d"]).astype(int)
    acc_1d = df_valid["hit_1d"].mean()
    print(f"\nAccuracy señal_final vs truth_1d (en df_valid completo) = {acc_1d:.4f}")

    # Backtest con posición dinámica + régimen
    backtest_with_signals(df_valid)


if __name__ == "__main__":
    main()