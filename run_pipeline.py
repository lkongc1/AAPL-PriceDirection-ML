# run_pipeline.py
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
from catboost import CatBoostClassifier

# ==============================
# CONFIGURACIÓN DE RUTAS
# ==============================
BASE_PATH = Path(__file__).resolve().parent / "csv"
MODEL_PATH = Path("catboost_model.cbm")

TICKERS = {
    "AAPL": "aapl_clean.csv",
    "SPY":  "spy_clean.csv",
    "QQQ":  "qqq_clean.csv",
    "^VIX": "vix_clean.csv",
    "^TNX": "tnx_clean.csv",
}

# PARA LA LIMPIEZA DE LOS CSV
def clean_csv_file(filename: str):
    """
    Limpia un CSV:
    - Elimina filas sin fecha válida.
    - (Opcional) Fuerza OHLCV a numérico y elimina filas sin datos válidos.
    """
    file_path = BASE_PATH / filename
    if not file_path.exists():
        return

    df = pd.read_csv(file_path)

    # Asegurar columna Date y convertirla a datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # Quitar filas sin fecha válida (aquí cae la fila ',AAPL,AAPL,...')
        df = df.dropna(subset=["Date"])
    else:
        # Si no tiene Date, no nos sirve nada
        return

    # Forzar numérico en OHLCV por si hubiera texto raro
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Quitar filas donde todo OHLCV sea NaN
    if all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
        mask_valid = df[["Open", "High", "Low", "Close", "Volume"]].notna().any(axis=1)
        df = df[mask_valid]

    df.to_csv(file_path, index=False)

def clean_all_csvs():
    for _, fname in TICKERS.items():
        clean_csv_file(fname)
# =====================================================
# 0) ACTUALIZACIÓN AUTOMÁTICA DESDE YFINANCE (DESCARGA COMPLETA)
# =====================================================

def update_ticker(ticker: str, filename: str, start_date="2005-01-01"):
    file_path = BASE_PATH / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.today().date()
    end_date = today + timedelta(days=1)  # 'end' en yfinance es exclusivo

    print(f"{ticker}: descargando desde {start_date} hasta {today} ...")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"No se obtuvieron datos para {ticker}")
        return

    # Pasar el índice a columna
    df = df.reset_index()

    # Renombrar la primera columna a 'Date' si se llama distinto
    first_col = df.columns[0]
    if first_col != "Date":
        df = df.rename(columns={first_col: "Date"})

    # Asegurar columnas base
    cols_base = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for c in cols_base:
        if c not in df.columns:
            df[c] = 0 if c == "Volume" else np.nan

    df = df[cols_base]

    # Añadir Name y asegurar orden
    df["Name"] = ticker
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Name"]]

    df.to_csv(file_path, index=False)
    print(
        f"  Guardado {file_path.name} "
        f"(desde {df['Date'].min().date()} hasta {df['Date'].max().date()})"
    )


def update_all_tickers():
    print("=== ACTUALIZANDO DATOS DE YFINANCE (DESCARGA COMPLETA) ===")
    for t, fname in TICKERS.items():
        update_ticker(t, fname)
    print("=== ACTUALIZACIÓN COMPLETA ===\n")


# =====================================================
# 1) INDICADORES TÉCNICOS
# =====================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    # Robustez: convertir a numérico por si viniera como texto
    series = pd.to_numeric(series, errors="coerce")
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(h, l, c):
    h = pd.to_numeric(h, errors="coerce")
    l = pd.to_numeric(l, errors="coerce")
    c = pd.to_numeric(c, errors="coerce")
    prev_close = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(h, l, c, window=14):
    return true_range(h, l, c).rolling(window).mean()

def adx(h, l, c, window=14):
    h = pd.to_numeric(h, errors="coerce")
    l = pd.to_numeric(l, errors="coerce")
    c = pd.to_numeric(c, errors="coerce")

    tr = true_range(h, l, c)
    up_move = h.diff()
    down_move = l.diff().mul(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_smooth = pd.Series(plus_dm, index=h.index).ewm(alpha=1/window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=h.index).ewm(alpha=1/window, adjust=False).mean()
    tr_smooth = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-12)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_val = dx.ewm(alpha=1/window, adjust=False).mean()
    return adx_val

def stochastic_kd(h, l, c, window=14):
    h = pd.to_numeric(h, errors="coerce")
    l = pd.to_numeric(l, errors="coerce")
    c = pd.to_numeric(c, errors="coerce")
    lowest_low = l.rolling(window).min()
    highest_high = h.rolling(window).max()
    stochastic_k = 100 * (c - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stochastic_d = stochastic_k.rolling(3).mean()
    return stochastic_k, stochastic_d

def bollinger_bands(c, window=20, num_std=2):
    c = pd.to_numeric(c, errors="coerce")
    ma = c.rolling(window).mean()
    std = c.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / (ma + 1e-12)
    return ma, upper, lower, width

def obv(c, v):
    c = pd.to_numeric(c, errors="coerce")
    v = pd.to_numeric(v, errors="coerce")
    direction = np.sign(c.diff().fillna(0))
    return (direction * v).cumsum()

feature_cols = [
    "ret1", "ret5", "ret10", "hl_range", "atr14", "std10", "std20", "z_volume",
    "ema20", "ema50", "ema200", "ema20_slope", "macd_line", "macd_signal", "macd_hist",
    "rsi14", "stoch_k", "stoch_d", "roc10", "bb_ma", "bb_width", "obv", "obv_change",
    "adx14", "dist_ema20", "dist_ema50", "spy_trend", "spy_vol20", "qqq_trend",
    "vix_trend", "tnx_trend"
]

# =====================================================
# 2) CARGA DE SERIES MACRO Y FEATURES
# =====================================================

def load_macro_series():
    files = {
        'spy_clean.csv': 'spy_close',
        'qqq_clean.csv': 'qqq_close', 
        'vix_clean.csv': 'vix_close',
        'tnx_clean.csv': 'tnx_close'
    }
    series = {}
    for file, new_name in files.items():
        try:
            p = BASE_PATH / file
            if p.exists():
                df = pd.read_csv(p, parse_dates=["Date"])
                # En nuestros CSV, la columna de cierre se llama "Close"
                price_col = "Close"
                if price_col not in df.columns:
                    price_col = next((col for col in ['Close', 'Adj Close', 'close'] if col in df.columns), None)
                if price_col:
                    df = df[["Date", price_col]].copy()
                    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
                    df = df.dropna(subset=[price_col])
                    df = df.rename(columns={price_col: new_name})
                    df = df.sort_values("Date").reset_index(drop=True)
                    series[new_name] = df
            else:
                print(f"Archivo no encontrado: {file}")
        except Exception as e:
            print(f"Error cargando {file}: {e}")
    return series

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    macro_series = load_macro_series()
    for name, macro_df in macro_series.items():
        df = df.merge(macro_df, on="Date", how="left")
        # name: 'spy_close', 'qqq_close', 'vix_close', 'tnx_close'
        if 'spy' in name or 'qqq' in name:
            ema_col = f"{name.split('_')[0]}_ema200"   # spy_ema200, qqq_ema200
            trend_col = f"{name.split('_')[0]}_trend"  # spy_trend, qqq_trend
            df[ema_col] = ema(df[name], 200)
            df[trend_col] = (df[name] - df[ema_col]) / (df[ema_col] + 1e-12)
            if 'spy' in name:
                df["spy_vol20"] = df[name].pct_change().rolling(20).std()
        else:
            ema_col = f"{name.split('_')[0]}_ema20"    # vix_ema20, tnx_ema20
            trend_col = f"{name.split('_')[0]}_trend"  # vix_trend, tnx_trend
            df[ema_col] = ema(df[name], 20)
            df[trend_col] = (df[name] - df[ema_col]) / (df[ema_col] + 1e-12)

    # Desplazar tendencias y volatilidad para evitar leakage
    macro_cols = [col for col in df.columns if any(x in col for x in ['_trend', '_vol20'])]
    for col in macro_cols:
        df[col] = df[col].shift(1)
    return df

def compute_features_aapl(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar numérico en OHLCV
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    hl_range = h - l
    returns = o.pct_change()

    ema20, ema50, ema200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd_line = ema(c, 12) - ema(c, 26)
    macd_signal = ema(macd_line, 9)
    rsi14 = rsi(c, 14)
    stoch_k, stoch_d = stochastic_kd(h, l, c, 14)
    adx14 = adx(h, l, c, 14)
    bb_ma, _, _, bb_width = bollinger_bands(c, 20, 2)
    obv_series = obv(c, v)

    feats = pd.DataFrame({
        "ret1": returns.shift(1),
        "ret5": o.pct_change(5).shift(1),
        "ret10": o.pct_change(10).shift(1),
        "hl_range": hl_range.shift(1),
        "atr14": atr(h, l, c, 14).shift(1),
        "std10": returns.rolling(10).std().shift(1),
        "std20": returns.rolling(20).std().shift(1),
        "z_volume": ((v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-12)).shift(1),
        "ema20": ema20.shift(1),
        "ema50": ema50.shift(1),
        "ema200": ema200.shift(1),
        "ema20_slope": ema20.diff().shift(1),
        "macd_line": macd_line.shift(1),
        "macd_signal": macd_signal.shift(1),
        "macd_hist": (macd_line - macd_signal).shift(1),
        "rsi14": rsi14.shift(1),
        "stoch_k": stoch_k.shift(1),
        "stoch_d": stoch_d.shift(1),
        "roc10": c.pct_change(10).shift(1),
        "bb_ma": bb_ma.shift(1),
        "bb_width": bb_width.shift(1),
        "obv": obv_series.shift(1),
        "obv_change": obv_series.diff().shift(1),
        "adx14": adx14.shift(1),
        "dist_ema20": ((c - ema20) / (ema20 + 1e-12)).shift(1),
        "dist_ema50": ((c - ema50) / (ema50 + 1e-12)).shift(1),
    }, index=df.index)
    return feats

def load_and_prepare_aapl() -> pd.DataFrame:
    p = BASE_PATH / "aapl_clean.csv"
    if not p.exists():
        raise FileNotFoundError(f"No existe archivo AAPL: {p}")

    df = pd.read_csv(p, parse_dates=["Date"])

    # Quitar espacios en nombres por si acaso
    df.columns = [c.strip() for c in df.columns]

    # Asegurar columnas requeridas
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna requerida faltante en AAPL: {col}")

    df = df[required_cols].copy().sort_values("Date").reset_index(drop=True)

    # Forzar numérico en OHLCV
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Añadir macros y features
    df = add_macro_features(df)
    feats_price = compute_features_aapl(df)

    result = pd.concat([df, feats_price], axis=1).dropna().reset_index(drop=True)

    print(f"Datos AAPL: {len(result)} registros desde {result['Date'].min().strftime('%Y-%m-%d')} hasta {result['Date'].max().strftime('%Y-%m-%d')}")

    columnas_faltantes = [col for col in feature_cols if col not in result.columns]
    if columnas_faltantes:
        print(f"Columnas faltantes en features: {columnas_faltantes}")

    return result

# =====================================================
# 3) PREDICCIÓN
# =====================================================

def get_confidence_analysis(prob_subir, prob_neutral, prob_bajar):
    prob_maxima = max(prob_subir, prob_neutral, prob_bajar)
    diferencia = abs(prob_subir - prob_bajar)

    if prob_maxima == prob_subir:
        direccion = "SUBIR"
        if prob_subir >= 0.75:
            confianza, recomendacion = "MUY ALTA", "FUERTE COMPRA"
        elif prob_subir >= 0.65:
            confianza, recomendacion = "ALTA", "COMPRAR"
        elif prob_subir >= 0.55:
            confianza, recomendacion = "MEDIA", "COMPRA MODERADA"
        else:
            confianza, recomendacion = "BAJA", "ESPERAR"
    elif prob_maxima == prob_bajar:
        direccion = "BAJAR"
        if prob_bajar >= 0.75:
            confianza, recomendacion = "MUY ALTA", "FUERTE VENTA"
        elif prob_bajar >= 0.65:
            confianza, recomendacion = "ALTA", "VENDER"
        elif prob_bajar >= 0.55:
            confianza, recomendacion = "MEDIA", "REDUCIR"
        else:
            confianza, recomendacion = "BAJA", "ESPERAR"
    else:
        direccion = "NEUTRAL"
        if prob_neutral >= 0.70:
            confianza, recomendacion = "ALTA", "MANTENER"
        else:
            confianza, recomendacion = "MEDIA", "OBSERVAR"

    if diferencia > 0.35:
        fuerza_señal = "FUERTE TENDENCIA"
    elif diferencia > 0.20:
        fuerza_señal = "TENDENCIA MODERADA"
    elif diferencia > 0.10:
        fuerza_señal = "SEÑAL DÉBIL"
    else:
        fuerza_señal = "MERCADO INDECISO"

    return direccion, confianza, recomendacion, fuerza_señal, diferencia

def predict_next_day():
    print("INICIANDO PREDICCIÓN AAPL (SIGUIENTE DÍA)")
    print("="*65)
    try:
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))

        df = load_and_prepare_aapl()
        fecha_ultimo_dato = df['Date'].max()
        ultimo_dato = df.iloc[[-1]].copy()

        print(f"Último dato disponible: {fecha_ultimo_dato.strftime('%Y-%m-%d')}")
        print(f"Precio cierre: ${ultimo_dato['Close'].iloc[0]:.2f}")

        columnas_faltantes = [col for col in feature_cols if col not in ultimo_dato.columns]
        if columnas_faltantes:
            print(f"Columnas faltantes: {columnas_faltantes}")
            return None

        X_actual = ultimo_dato[feature_cols].values.astype("float32")
        probabilidades = model.predict_proba(X_actual)[0]
        prob_subir, prob_neutral, prob_bajar = probabilidades[2], probabilidades[1], probabilidades[0]

        direccion, confianza, recomendacion, fuerza_señal, diferencia = get_confidence_analysis(
            prob_subir, prob_neutral, prob_bajar
        )

        print(f"\nPREDICCIÓN: {direccion}")
        print(f"CONFIANZA: {confianza} | SEÑAL: {fuerza_señal}")
        print(f"RECOMENDACIÓN: {recomendacion}")

        print("\nDISTRIBUCIÓN DE PROBABILIDADES:")
        print(f"    SUBIR:   {prob_subir:>6.1%}")
        print(f"    NEUTRAL: {prob_neutral:>6.1%}")
        print(f"    BAJAR:   {prob_bajar:>6.1%}")
        print(f"\nDIFERENCIA SUBIR/BAJAR: {diferencia:.1%}")

        fecha_prediccion = fecha_ultimo_dato + timedelta(days=1)

        return {
            'fecha_prediccion': fecha_prediccion,
            'direccion': direccion,
            'confianza': confianza,
            'recomendacion': recomendacion,
            'probabilidades': {
                'subir': prob_subir,
                'neutral': prob_neutral,
                'bajar': prob_bajar
            },
            'fuerza_senal': fuerza_señal,
            'diferencia': diferencia
        }

    except Exception as e:
        print(f"Error en la predicción: {e}")
        import traceback
        traceback.print_exc()
        return None

# =====================================================
# 4) MAIN
# =====================================================

def main():
    print(" SISTEMA DE PREDICCIÓN AAPL - PIPELINE COMPLETO")
    print("="*65)

    # 1) Actualizar datos (descarga desde 2005 y sobreescribe CSVs)
    update_all_tickers()

    # 1.5) Limpiar CSVs por si hubiera filas basura
    clean_all_csvs()
    
    # 2) Predecir
    resultado = predict_next_day()

    if resultado:
        print("\n" + " RESUMEN EJECUTIVO ".center(65, "="))
        print(f" FECHA PREDICCIÓN (teórica): {resultado['fecha_prediccion'].strftime('%d/%m/%Y')}")
        print(f" PREDICCIÓN: {resultado['direccion']}")
        print(f" CONFIANZA: {resultado['confianza']}")
        print(f" ACCIÓN: {resultado['recomendacion']}")
        print(f" SEÑAL: {resultado['fuerza_senal']}")
        print("="*65)

def run_pipeline_prediction():
    """
    Versión FastAPI:
    - No imprime nada
    - No descarga históricos completos (solo usa CSV ya existentes)
    - Devuelve JSON estructurado
    """
    try:
        model = CatBoostClassifier()
        model.load_model(str(MODEL_PATH))

        # Cargar datos (sin imprimir)
        df = load_and_prepare_aapl()

        fecha_ultimo_dato = df['Date'].max()
        ultimo_dato = df.iloc[[-1]].copy()

        # Features
        X_actual = ultimo_dato[feature_cols].values.astype("float32")

        # Predicción
        probabilidades = model.predict_proba(X_actual)[0]
        prob_subir, prob_neutral, prob_bajar = probabilidades[2], probabilidades[1], probabilidades[0]

        direccion, confianza, recomendacion, fuerza_señal, diferencia = get_confidence_analysis(
            prob_subir, prob_neutral, prob_bajar
        )

        fecha_prediccion = fecha_ultimo_dato + timedelta(days=1)

        return {
            'fecha_prediccion': fecha_prediccion.strftime("%Y-%m-%d"),
            'direccion': direccion,
            'confianza': confianza,
            'recomendacion': recomendacion,
            'probabilidades': {
                'subir': round(prob_subir, 6),
                'neutral': round(prob_neutral, 6),
                'bajar': round(prob_bajar, 6)
            },
            'fuerza_senal': fuerza_señal,
            'diferencia': round(diferencia, 6)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    main()