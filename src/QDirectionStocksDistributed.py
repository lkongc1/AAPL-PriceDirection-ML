import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import json

# ML
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef

# Dask
import dask
dask.config.set({
    "dataframe.query-planning": False,
    "distributed.worker.memory.target": 0.8,
    "distributed.worker.memory.spill": 0.85,
    "distributed.worker.memory.pause": 0.90,
})

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress, wait
from dask import delayed, compute

# =====================================================
# CONFIG
# =====================================================
class Config:
    """Configuración centralizada y serializable"""
    CSV_AAPL = r"aapl.csv"
    CSV_SPY  = r"spy.csv"
    CSV_QQQ  = r"qqq.csv"
    CSV_VIX  = r"vix.csv"
    CSV_TNX  = r"tnx.csv"
    
    START_WF = 300
    EPS_3D = 0.0025
    EPS_5D = 0.0040
    ADX_THR_BASE = 20
    
    # Configuración de paralelización
    N_WORKERS = 4
    THREADS_PER_WORKER = 2
    MEMORY_PER_WORKER = "4GB"
    
    # Walk-forward paralelo: tamaño de ventana para cada tarea
    WF_WINDOW_SIZE = 100  # Cada worker procesa 100 iteraciones
    
    # Usar GPU
    USE_GPU = False
    
    # Para escalar a múltiples máquinas
    SCHEDULER_ADDRESS = "tcp://ip:port" # None : localhost 


# =====================================================
# INDICADORES (sin cambios, pero como funciones puras)
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
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(h, l, c, window=14):
    tr = true_range(h, l, c)
    return tr.rolling(window).mean()

def adx(h, l, c, window=14):
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(h, l, c)
    tr_n = tr.rolling(window).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=h.index).rolling(window).sum() / (tr_n + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=h.index).rolling(window).sum() / (tr_n + 1e-12))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)) * 100
    return dx.rolling(window).mean()

def bollinger_bands(c, window=20, num_std=2):
    ma = c.rolling(window).mean()
    std = c.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / (ma + 1e-12)
    return ma, upper, lower, width

def obv(c, v):
    direction = np.sign(c.diff().fillna(0))
    return (direction * v).cumsum()

def stochastic_kd(h, l, c, window=14):
    lowest_low = l.rolling(window).min()
    highest_high = h.rolling(window).max()
    stochastic_k = 100 * (c - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stochastic_d = stochastic_k.rolling(3).mean()
    return stochastic_k, stochastic_d


# =====================================================
# FEATURE ENGINEERING DISTRIBUIDO
# =====================================================
def compute_features_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features para una partición - función pura para map_partitions"""
    if len(df) == 0:
        return df
    
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    
    # Todos los features con shift(1) para evitar look-ahead
    feats = pd.DataFrame(index=df.index)
    
    # Retornos
    feats["ret1"] = o.pct_change(1).shift(1)
    feats["ret5"] = o.pct_change(5).shift(1)
    feats["ret10"] = o.pct_change(10).shift(1)
    
    # Volatilidad
    feats["hl_range"] = (h - l).shift(1)
    feats["atr14"] = atr(h, l, c, 14).shift(1)
    feats["std10"] = o.pct_change(1).rolling(10).std().shift(1)
    feats["std20"] = o.pct_change(1).rolling(20).std().shift(1)
    
    # Volumen
    feats["z_volume"] = ((v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)).shift(1)
    
    # EMAs
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    feats["ema20"] = ema20.shift(1)
    feats["ema50"] = ema50.shift(1)
    feats["ema200"] = ema200.shift(1)
    feats["ema20_slope"] = ema20.diff().shift(1)
    
    # MACD
    macd_line = ema(c, 12) - ema(c, 26)
    macd_signal = ema(macd_line, 9)
    feats["macd_line"] = macd_line.shift(1)
    feats["macd_signal"] = macd_signal.shift(1)
    feats["macd_hist"] = (macd_line - macd_signal).shift(1)
    
    # Osciladores
    feats["rsi14"] = rsi(c, 14).shift(1)
    stoch_k, stoch_d = stochastic_kd(h, l, c, 14)
    feats["stoch_k"] = stoch_k.shift(1)
    feats["stoch_d"] = stoch_d.shift(1)
    feats["roc10"] = c.pct_change(10).shift(1)
    
    # Bollinger
    bb_ma, bb_upper, bb_lower, bb_width = bollinger_bands(c, 20, 2)
    feats["bb_ma"] = bb_ma.shift(1)
    feats["bb_width"] = bb_width.shift(1)
    
    # OBV
    obv_series = obv(c, v)
    feats["obv"] = obv_series.shift(1)
    feats["obv_change"] = obv_series.diff().shift(1)
    
    # ADX y distancias
    feats["adx14"] = adx(h, l, c, 14).shift(1)
    feats["dist_ema20"] = ((c - ema20) / (ema20 + 1e-12)).shift(1)
    feats["dist_ema50"] = ((c - ema50) / (ema50 + 1e-12)).shift(1)
    
    return pd.concat([df[["Open", "High", "Low", "Close", "Volume"]], feats], axis=1)


def compute_labels_partition(df: pd.DataFrame, eps_3d: float, eps_5d: float) -> pd.DataFrame:
    """Calcula labels para una partición"""
    if len(df) == 0:
        return df
    
    o = df["Open"]
    
    # Retornos futuros
    ret_3d = (o.shift(-3) - o) / o
    ret_5d = (o.shift(-5) - o) / o
    ret_next = (o.shift(-1) - o) / o
    
    df = df.copy()
    df["ret_3d"] = ret_3d
    df["ret_5d"] = ret_5d
    df["ret_next"] = ret_next
    
    # Direcciones
    df["y_dir_3d"] = 0
    df.loc[ret_3d > eps_3d, "y_dir_3d"] = 1
    df.loc[ret_3d < -eps_3d, "y_dir_3d"] = -1
    
    df["y_dir_5d"] = 0
    df.loc[ret_5d > eps_5d, "y_dir_5d"] = 1
    df.loc[ret_5d < -eps_5d, "y_dir_5d"] = -1
    
    return df


# =====================================================
# WALK-FORWARD PARALELO - EL CAMBIO CLAVE
# =====================================================
@delayed
def train_model_window(
    X: np.ndarray,
    y: np.ndarray,
    window_start: int,
    window_end: int,
    model_type: str = "catboost",
    use_gpu: bool = False,
):
    """
    Entrena de forma secuencial un modelo walk‑forward en el rango
    [window_start, window_end) y devuelve las probabilidades
    para cada t de esa ventana.
    """
    import tempfile
    import numpy as np
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    n_classes = 3  # porque y ya viene de encode_trinary
    proba_window = np.full((window_end - window_start, n_classes), np.nan, dtype=np.float32)

    for local_i, t in enumerate(range(window_start, window_end)):
        # Datos hasta t para entrenar, t para test
        X_train, y_train = X[:t], y[:t]
        X_test = X[t:t+1]

        # -------------------------
        # Construir el modelo
        # -------------------------
        if model_type == "catboost":
            params = dict(
                loss_function="MultiClass",
                iterations=300,
                depth=6,
                learning_rate=0.05,
                verbose=False,
                train_dir=tempfile.mkdtemp(prefix="catboost_dask_"),
            )
            # Solo aquí respetamos use_gpu
            params["task_type"] = "CPU"
            model = CatBoostClassifier(**params)

        elif model_type == "xgboost":
            # FORZAMOS CPU: tu xgboost no soporta 'gpu_hist'
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                objective="multi:softprob",
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=-1,
                tree_method="hist",  # SIEMPRE CPU
            )

        elif model_type == "lightgbm":
            # Para evitar los problemas de boost_compute en GPU/OpenCL
            model = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                objective="multiclass",
                num_class=n_classes,
                n_jobs=-1,
                device_type="cpu",  # FORZAMOS CPU
                verbose=-1,
            )
        else:
            raise ValueError(f"Modelo no soportado: {model_type}")

        # -------------------------
        # Entrenar y predecir
        # -------------------------
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[0]

        # Alinear por si falta alguna clase en la ventana
        if proba.shape[0] != n_classes and hasattr(model, "classes_"):
            tmp = np.zeros(n_classes, dtype=np.float32)
            for cls_idx, cls_label in enumerate(model.classes_):
                tmp[int(cls_label)] = proba[cls_idx]
            proba = tmp

        proba_window[local_i, :] = proba

    # Esta tupla es lo que esperan walk_forward_parallel y train_ensemble_parallel
    return window_start, window_end, proba_window


def walk_forward_parallel(
    X: np.ndarray,
    y: np.ndarray,
    start: int = 300,
    window_size: int = 100,
    model_type: str = "catboost",
    use_gpu: bool = False,
    client: Client = None
) -> np.ndarray:
    """
    Walk-forward paralelizado por ventanas.
    
    En lugar de hacer t=300, t=301, t=302... secuencialmente,
    dividimos en ventanas y procesamos en paralelo:
    - Worker 1: t=300-399
    - Worker 2: t=400-499
    - Worker 3: t=500-599
    - etc.
    """
    n = len(y)
    proba = np.full((n, 3), np.nan)
    
    # Crear tareas para cada ventana
    tasks = []
    for window_start in range(start, n, window_size):
        window_end = min(window_start + window_size, n)
        task = train_model_window(
            X, y, window_start, window_end, model_type, use_gpu
        )
        tasks.append(task)
    
    print(f"[{model_type.upper()}] Creadas {len(tasks)} tareas paralelas")
    
    # Ejecutar en paralelo
    if client:
        futures = client.compute(tasks)
        progress(futures)
        results = client.gather(futures)
    else:
        results = compute(*tasks)
    
    # Combinar resultados
    for start_idx, end_idx, proba_window in results:
        proba[start_idx:end_idx, :] = proba_window
    
    return proba


# =====================================================
# ENSEMBLE PARALELO - MÚLTIPLES MODELOS
# =====================================================
def train_ensemble_parallel(
    X: np.ndarray,
    y: np.ndarray,
    start: int = 300,
    window_size: int = 100,
    use_gpu: bool = False,
    client: Client = None
) -> Dict[str, np.ndarray]:
    """
    Entrena múltiples modelos en paralelo y devuelve sus predicciones.
    """
    model_types = ["catboost", "xgboost", "lightgbm"]
    
    # Crear todas las tareas para todos los modelos
    all_tasks = {}
    for model_type in model_types:
        tasks = []
        n = len(y)
        for window_start in range(start, n, window_size):
            window_end = min(window_start + window_size, n)
            task = train_model_window(
                X, y, window_start, window_end, model_type, use_gpu
            )
            tasks.append(task)
        all_tasks[model_type] = tasks
    
    # Aplanar todas las tareas
    flat_tasks = []
    task_mapping = []  # (model_type, task_index)
    for model_type, tasks in all_tasks.items():
        for i, task in enumerate(tasks):
            flat_tasks.append(task)
            task_mapping.append((model_type, i))
    
    print(f"[ENSEMBLE] Total de tareas: {len(flat_tasks)}")
    
    # Ejecutar todas en paralelo
    if client:
        futures = client.compute(flat_tasks)
        progress(futures)
        results = client.gather(futures)
    else:
        results = compute(*flat_tasks)
    
    # Organizar resultados por modelo
    n = len(y)
    model_proba = {mt: np.full((n, 3), np.nan) for mt in model_types}
    
    for (model_type, _), (start_idx, end_idx, proba_window) in zip(task_mapping, results):
        model_proba[model_type][start_idx:end_idx, :] = proba_window
    
    return model_proba


# =====================================================
# DATA LOADING DISTRIBUIDO
# =====================================================
class DistributedDataLoader:
    """Carga y procesa datos de forma distribuida"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_csv_partitioned(self, path: str, n_partitions: int = 4) -> dd.DataFrame:
        """Carga CSV y lo particiona para procesamiento paralelo"""
        # Primero cargamos para saber el tamaño
        df = pd.read_csv(path, parse_dates=["Date"])
        cols = [c for c in df.columns if c in ["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df[cols].set_index("Date")
        
        # Convertir a dask con particiones
        ddf = dd.from_pandas(df, npartitions=n_partitions)
        return ddf
    
    def load_all_data(self, client: Client) -> pd.DataFrame:
        """Carga todos los datos y los prepara para ML"""
        
        print("[1/5] Cargando datos...")
        aapl = self.load_csv_partitioned(self.config.CSV_AAPL)
        spy = self.load_csv_partitioned(self.config.CSV_SPY)
        qqq = self.load_csv_partitioned(self.config.CSV_QQQ)
        vix = self.load_csv_partitioned(self.config.CSV_VIX)
        tnx = self.load_csv_partitioned(self.config.CSV_TNX)
        
        print("[2/5] Calculando features en paralelo...")
        
        # Definir meta para map_partitions
        sample_df = aapl.head(5)
        sample_feats = compute_features_partition(sample_df)
        meta = sample_feats
        
        # Procesar cada asset en paralelo
        aapl_f = aapl.map_partitions(compute_features_partition, meta=meta)
        spy_f = spy.map_partitions(compute_features_partition, meta=meta)
        qqq_f = qqq.map_partitions(compute_features_partition, meta=meta)
        vix_f = vix.map_partitions(compute_features_partition, meta=meta)
        tnx_f = tnx.map_partitions(compute_features_partition, meta=meta)
        
        # Persist para mantener en memoria distribuida
        aapl_f, spy_f, qqq_f, vix_f, tnx_f = client.persist([aapl_f, spy_f, qqq_f, vix_f, tnx_f])
        wait([aapl_f, spy_f, qqq_f, vix_f, tnx_f])
        
        print("[3/5] Combinando datos...")
        
        # Compute individual dataframes
        aapl_pd = aapl_f.compute().reset_index()
        spy_pd = spy_f.compute().reset_index()
        qqq_pd = qqq_f.compute().reset_index()
        vix_pd = vix_f.compute().reset_index()
        tnx_pd = tnx_f.compute().reset_index()
        
        # Renombrar columnas macro
        def rename_cols(df, prefix):
            rename_map = {c: f"{prefix}_{c}" if c != "Date" else c for c in df.columns}
            return df.rename(columns=rename_map)
        
        spy_pd = rename_cols(spy_pd[["Date", "Close", "ret1", "std20", "ema200"]], "spy")
        qqq_pd = rename_cols(qqq_pd[["Date", "Close", "ret1", "ema200"]], "qqq")
        vix_pd = rename_cols(vix_pd[["Date", "Close", "ret1"]], "vix")
        tnx_pd = rename_cols(tnx_pd[["Date", "Close", "ret1"]], "tnx")
        
        # Merge
        df = aapl_pd.copy()
        df = df.merge(spy_pd, on="Date", how="left")
        df = df.merge(qqq_pd, on="Date", how="left")
        df = df.merge(vix_pd, on="Date", how="left")
        df = df.merge(tnx_pd, on="Date", how="left")
        
        print("[4/5] Calculando features macro...")
        
        # Features macro adicionales
        df["spy_trend"] = ((df["spy_Close"] - df["spy_ema200"]) / (df["spy_ema200"] + 1e-12)).shift(1)
        df["qqq_trend"] = ((df["qqq_Close"] - df["qqq_ema200"]) / (df["qqq_ema200"] + 1e-12)).shift(1)
        df["vix_ma20"] = df["vix_Close"].rolling(20).mean()
        df["vix_trend"] = ((df["vix_Close"] - df["vix_ma20"]) / (df["vix_ma20"] + 1e-12)).shift(1)
        
        print("[5/5] Calculando labels...")
        
        # Labels
        df = compute_labels_partition(df, self.config.EPS_3D, self.config.EPS_5D)
        
        # Limpiar
        df = df.ffill().bfill()
        df = df.iloc[:-5].dropna().reset_index(drop=True)
        
        return df


# =====================================================
# EVALUACIÓN Y BACKTEST
# =====================================================
def encode_trinary(y_dir: pd.Series) -> np.ndarray:
    mapping = {-1: 0, 0: 1, 1: 2}
    return y_dir.map(mapping).values.astype(int)


def best_threshold(y_true_binary: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Encuentra el mejor umbral para MCC"""
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.3, 0.7, 81):
        pred = (p >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        mcc = matthews_corrcoef(y_true_binary, pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return best_t, best_mcc


def backtest(df: pd.DataFrame, thr_up: float, thr_down: float, config: Config):
    """Ejecuta backtest con las señales"""
    df = df.copy()
    
    # Generar señales
    def get_signal(row):
        if np.isnan(row["p_up_ens"]) or np.isnan(row["adx14"]):
            return 0
        if row["adx14"] < config.ADX_THR_BASE:
            return 0
        if row["p_up_ens"] >= thr_up and row["p_up_ens"] >= row["p_down_ens"]:
            return 1
        if row["p_down_ens"] >= thr_down and row["p_down_ens"] > row["p_up_ens"]:
            return -1
        return 0
    
    df["signal"] = df.apply(get_signal, axis=1)
    df["strategy_return"] = df["signal"].shift(1) * df["ret_next"]
    
    # Métricas
    df_bt = df.dropna(subset=["strategy_return"])
    df_bt["equity"] = df_bt["strategy_return"].cumsum()
    
    total_return = df_bt["strategy_return"].sum() * 100
    win_rate = (df_bt["strategy_return"] > 0).mean()
    sharpe = df_bt["strategy_return"].mean() / (df_bt["strategy_return"].std() + 1e-12) * np.sqrt(252)
    
    df_bt["peak"] = df_bt["equity"].cummax()
    max_dd = (df_bt["equity"] - df_bt["peak"]).min() * 100
    
    print(f"\n{'='*50}")
    print("RESULTADOS BACKTEST")
    print(f"{'='*50}")
    print(f"Retorno total: {total_return:.2f}%")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Sharpe ratio: {sharpe:.3f}")
    print(f"Max drawdown: {max_dd:.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_bt["Date"], df_bt["equity"])
    plt.title("Equity Curve - Estrategia Distribuida")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return df_bt


# =====================================================
# MAIN - ORQUESTACIÓN DISTRIBUIDA
# =====================================================
def create_cluster(config: Config) -> Client:
    """Crea el cluster - local o remoto"""
    if config.SCHEDULER_ADDRESS:
        # Conectar a cluster existente (múltiples máquinas)
        print(f"Conectando a cluster: {config.SCHEDULER_ADDRESS}")
        client = Client(config.SCHEDULER_ADDRESS)
    else:
        # Crear cluster local
        print("Creando cluster local...")
        cluster = LocalCluster(
            n_workers=config.N_WORKERS,
            threads_per_worker=config.THREADS_PER_WORKER,
            memory_limit=config.MEMORY_PER_WORKER,
            dashboard_address=":8787"  # Dashboard en http://localhost:8787
        )
        client = Client(cluster)
    
    print(f"Dashboard: {client.dashboard_link}")
    print(client)
    return client


def main():
    config = Config()
    
    # 1. Crear/conectar cluster
    client = create_cluster(config)
    
    try:
        # 2. Cargar datos de forma distribuida
        loader = DistributedDataLoader(config)
        df = loader.load_all_data(client)
        print(f"\n[OK] Dataset listo. Shape: {df.shape}")
        
        # 3. Preparar features y labels
        feature_cols = [
            "ret1", "ret5", "ret10", "hl_range", "atr14", "std10", "std20", "z_volume",
            "ema20", "ema50", "ema200", "ema20_slope",
            "macd_line", "macd_signal", "macd_hist",
            "rsi14", "stoch_k", "stoch_d", "roc10",
            "bb_ma", "bb_width", "obv", "obv_change",
            "adx14", "dist_ema20", "dist_ema50",
            "spy_trend", "qqq_trend", "vix_trend"
        ]
        
        # Asegurar que todas las columnas existan
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        
        X = df[feature_cols].values.astype("float32")
        y3 = encode_trinary(df["y_dir_3d"])
        y5 = encode_trinary(df["y_dir_5d"])
        
        # 4. Walk-forward PARALELO con ensemble
        print("\n" + "="*50)
        print("ENTRENAMIENTO PARALELO")
        print("="*50)
        
        # Opción A: Un solo modelo paralelo
        # proba3 = walk_forward_parallel(
        #     X, y3, config.START_WF, config.WF_WINDOW_SIZE,
        #     "catboost", config.USE_GPU, client
        # )
        
        # Opción B: Ensemble de modelos en paralelo (más distribuido)
        print("\n[3D] Entrenando ensemble...")
        model_proba_3d = train_ensemble_parallel(
            X, y3, config.START_WF, config.WF_WINDOW_SIZE,
            config.USE_GPU, client
        )
        
        print("\n[5D] Entrenando ensemble...")
        model_proba_5d = train_ensemble_parallel(
            X, y5, config.START_WF, config.WF_WINDOW_SIZE,
            config.USE_GPU, client
        )
        
        # 5. Combinar predicciones del ensemble
        weights = {"catboost": 0.4, "xgboost": 0.3, "lightgbm": 0.3}
        
        proba3 = sum(w * model_proba_3d[m] for m, w in weights.items())
        proba5 = sum(w * model_proba_5d[m] for m, w in weights.items())
        
        # Combinar 3D y 5D
        W3, W5 = 0.45, 0.55
        df["p_down_ens"] = W3 * proba3[:, 0] + W5 * proba5[:, 0]
        df["p_neutral_ens"] = W3 * proba3[:, 1] + W5 * proba5[:, 1]
        df["p_up_ens"] = W3 * proba3[:, 2] + W5 * proba5[:, 2]
        
        # 6. Encontrar umbrales óptimos
        df_valid = df.iloc[config.START_WF:].dropna(subset=["p_up_ens"]).reset_index(drop=True)
        split = int(len(df_valid) * 0.7)
        df_train = df_valid.iloc[:split]
        
        thr_up, mcc_up = best_threshold(
            (df_train["y_dir_5d"] == 1).astype(int).values,
            df_train["p_up_ens"].values
        )
        thr_down, mcc_down = best_threshold(
            (df_train["y_dir_5d"] == -1).astype(int).values,
            df_train["p_down_ens"].values
        )
        
        print(f"\nUmbral UP: {thr_up:.3f} (MCC: {mcc_up:.4f})")
        print(f"Umbral DOWN: {thr_down:.3f} (MCC: {mcc_down:.4f})")
        
        # 7. Backtest
        backtest(df_valid, thr_up, thr_down, config)
        
    finally:
        client.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()