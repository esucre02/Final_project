import streamlit as st
import pandas as pd
import pickle
import json
import yfinance as yf
import numpy as np

st.set_page_config(page_title="Oráculo Financiero Automático", page_icon="🤖", layout="centered")

# --- CARGAR MODELO Y DICCIONARIO ---
@st.cache_resource
def cargar_archivos():
    with open('models/oraculo_financiero_xgb.pkl', 'rb') as archivo_modelo:
        modelo = pickle.load(archivo_modelo)
    with open('src/diccionario_tickers.json', 'r') as archivo_json:
        diccionario = json.load(archivo_json)
    return modelo, diccionario

modelo, diccionario_tickers = cargar_archivos()

# --- FUNCIÓN PARA DESCARGAR Y CALCULAR INDICADORES EN VIVO ---
def obtener_datos_en_vivo(ticker_symbol):
    # Descargamos los últimos 100 días para tener datos suficientes para las medias móviles
    df = yf.download(ticker_symbol, period="100d", progress=False)
    
    # Calculamos MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    # Calculamos Z-Score (20 días)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    z_score = (df['Close'] - sma20) / std20
    
    # Calculamos RSI (14 días) simple
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Extraemos solo la ÚLTIMA fila (el día de hoy)
    hoy = df.iloc[-1]
    
    # Empaquetamos los datos que necesita el modelo
    datos_hoy = {
        'Volume': float(hoy['Volume']),
        'Std_20': float(std20.iloc[-1]),
        'MACD_Line': float(macd_line.iloc[-1]),
        'MACD_Signal': float(macd_signal.iloc[-1]),
        'MACD_Hist': float(macd_hist.iloc[-1]),
        'RSI_14': float(rsi.iloc[-1]),
        'Momentum_10': float(df['Close'].iloc[-1] - df['Close'].iloc[-11]),
        'Vol_SMA_20': float(df['Volume'].rolling(20).mean().iloc[-1]),
        'Vol_Ratio': float(hoy['Volume'] / df['Volume'].rolling(20).mean().iloc[-1]),
        'Vol_Change': 0.0,
        'High_Low_Spread': float(hoy['High'] - hoy['Low']),
        'Z_Score_20': float(z_score.iloc[-1]),
        'Above_SMA200': 1 if float(hoy['Close']) > float(df['Close'].mean()) else 0,
        'Pct_From_High_50': float((hoy['Close'] - df['High'].rolling(50).max().iloc[-1]) / df['High'].rolling(50).max().iloc[-1])
    }
    return datos_hoy

# --- INTERFAZ DE USUARIO ---
st.title("🤖 Oráculo Financiero en Tiempo Real")
st.markdown("Selecciona una empresa. La IA descargará los datos de Wall Street de hoy y hará la predicción instantáneamente.")
st.divider()

# Extraemos el texto puro del ticker (ej: "AAPL" en lugar de "Apple (AAPL)")
# Para que yfinance sepa a quién buscar. Asumiendo que tu diccionario tiene claves como "AAPL"
empresa_seleccionada = st.selectbox("Selecciona la Empresa", options=list(diccionario_tickers.keys()))

if st.button('🚀 Analizar Mercado en Vivo'):
    with st.spinner(f'Descargando datos financieros de {empresa_seleccionada} en tiempo real...'):
        try:
            # 1. Conseguimos el código interno para el modelo (0, 1, 2...)
            ticker_encoded = diccionario_tickers[empresa_seleccionada]
            
            # 2. Conseguimos los indicadores reales de hoy desde Yahoo Finance
            # (Si tu diccionario tiene formato "Apple (AAPL)", tendrás que adaptar esto para extraer solo "AAPL")
            datos_vivo = obtener_datos_en_vivo(empresa_seleccionada)
            datos_vivo['Ticker_Encoded'] = ticker_encoded
            
            # 3. Convertimos a DataFrame para el modelo
            datos_entrada = pd.DataFrame([datos_vivo])
            
            # 4. Predicción
            prediccion = modelo.predict(datos_entrada)[0]
            probabilidad = modelo.predict_proba(datos_entrada)[0]
            
            # 5. Mostrar resultados y los datos calculados para transparencia
            st.subheader(f"🔮 Resultados para {empresa_seleccionada}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("RSI Actual", f"{datos_vivo['RSI_14']:.2f}")
            col2.metric("MACD Hist", f"{datos_vivo['MACD_Hist']:.2f}")
            col3.metric("Z-Score", f"{datos_vivo['Z_Score_20']:.2f}")
            
            st.markdown("---")
            if prediccion == 1:
                st.success(f"**¡DÍA POSITIVO ESPERADO PARA MAÑANA! 🟢**")
                st.write(f"Confianza del modelo: **{probabilidad[1]*100:.2f}%**")
                st.balloons()
            else:
                st.error(f"**DÍA NEGATIVO ESPERADO PARA MAÑANA 🔴**")
                st.write(f"Confianza del modelo: **{probabilidad[0]*100:.2f}%**")
                
        except Exception as e:
            st.error("Hubo un error al descargar los datos de Yahoo Finance. Intenta con otra empresa.")
            st.write(e)
