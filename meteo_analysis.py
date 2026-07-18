import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def coletar_dados_meteo(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Coleta dados climáticos horários da API Open-Meteo.
    Parâmetros requeridos: Temperatura (2m), Umidade Relativa (2m), Velocidade do Vento (10m), Radiação Solar.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "auto"
    }
    
    logging.info(f"Iniciando requisição para a API Open-Meteo ({start_date} a {end_date}).")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Converte o JSON de horas e variáveis para DataFrame
        hourly_data = data.get("hourly", {})
        df = pd.DataFrame(hourly_data)
        logging.info("Dados coletados com sucesso.")
        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro na requisição da API: {e}")
        return pd.DataFrame()

def processar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata os dados brutos: ajusta índice de tempo, interpolação de nulos, e calcula médias móveis.
    """
    if df.empty:
        logging.warning("DataFrame vazio na entrada do processamento.")
        return df
        
    logging.info("Iniciando tratamento de dados.")
    # Converte coluna de tempo para datetime e define como índice
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Renomeando as colunas para português (opcional, para melhor leitura)
    df.rename(columns={
        'temperature_2m': 'Temperatura_C',
        'relative_humidity_2m': 'Umidade_Relativa_%',
        'wind_speed_10m': 'Velocidade_Vento_kmh',
        'shortwave_radiation': 'Radiacao_Solar_Wm2'
    }, inplace=True)
    
    # Trata valores ausentes por interpolação linear
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        df.interpolate(method='linear', inplace=True)
        logging.info(f"Foram interpolados {missing_before} valores ausentes.")
        
    # Criação de médias móveis de 24 horas (tendência suavizada)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[f'{col}_MM24h'] = df[col].rolling(window=24, min_periods=1).mean()
            
    logging.info("Tratamento de dados concluído.")
    return df

def gerar_resumo_diario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume os dados horários em médias, máximas e mínimas diárias.
    """
    logging.info("Gerando resumo diário.")
    # Seleciona apenas as colunas base (sem as médias móveis para o resumo)
    colunas_base = ['Temperatura_C', 'Umidade_Relativa_%', 'Velocidade_Vento_kmh', 'Radiacao_Solar_Wm2']
    colunas_disponiveis = [c for c in colunas_base if c in df.columns]
    
    df_diario = df[colunas_disponiveis].resample('D').agg(['mean', 'max', 'min'])
    # Achata o MultiIndex das colunas
    df_diario.columns = ['_'.join(col).strip() for col in df_diario.columns.values]
    return df_diario

def analisar_estatisticas(df: pd.DataFrame) -> dict:
    """
    Calcula estatísticas descritivas, matriz de correlação e o Índice de Calor (Heat Index).
    """
    logging.info("Realizando análise matemática e estatística.")
    
    colunas_base = ['Temperatura_C', 'Umidade_Relativa_%', 'Velocidade_Vento_kmh', 'Radiacao_Solar_Wm2']
    colunas_disponiveis = [c for c in colunas_base if c in df.columns]
    df_base = df[colunas_disponiveis]
    
    # Estatísticas Descritivas
    estatisticas = df_base.describe().T
    estatisticas['variancia'] = df_base.var()
    estatisticas['skewness'] = df_base.skew()
    estatisticas['kurtosis'] = df_base.kurtosis()
    estatisticas['mediana'] = df_base.median()
    
    # Matriz de Correlação de Pearson
    correlacao = df_base.corr(method='pearson')
    
    # Cálculo do Índice de Calor (Heat Index) usando fórmula empírica baseada em T (Celsius) e UR (%)
    # Referência simplificada adaptada da NOAA, que geralmente requer T > 27C para ser significativa,
    # mas aplicaremos a fórmula matemática usando NumPy em todo o array para demonstração
    T = df['Temperatura_C'].values
    UR = df['Umidade_Relativa_%'].values
    
    # Constantes da fórmula de Rothfusz (adaptadas para Celsius)
    c1, c2, c3, c4 = -8.78469475556, 1.61139411, 2.33854883889, -0.14611605
    c5, c6, c7, c8 = -0.012308094, -0.0164248277778, 0.002211732, 0.00072546
    c9 = -0.000003582
    
    heat_index = (c1 + (c2 * T) + (c3 * UR) + (c4 * T * UR) + 
                  (c5 * (T**2)) + (c6 * (UR**2)) + (c7 * (T**2) * UR) + 
                  (c8 * T * (UR**2)) + (c9 * (T**2) * (UR**2)))
                  
    # Para temperaturas baixas, o índice de calor é igual à própria temperatura
    heat_index = np.where(T < 27, T, heat_index)
    df['Indice_Calor_C'] = heat_index
    
    return {
        'estatisticas': estatisticas,
        'correlacao': correlacao
    }

def prever_temperatura(df: pd.DataFrame) -> dict:
    """
    Usa Regressão Linear para prever a temperatura com base na temperatura de 24 horas atrás (lag de 24h).
    """
    logging.info("Iniciando modelagem preditiva (Regressão Linear).")
    # Vamos prever a Temperatura_C.
    # Feature (X): Temperatura 24h atrás
    # Target (y): Temperatura atual
    
    # Copia os dados
    df_modelo = df[['Temperatura_C']].copy()
    df_modelo['Temp_Lag_24h'] = df_modelo['Temperatura_C'].shift(24)
    df_modelo.dropna(inplace=True) # Remove as primeiras 24 linhas que ficaram com NaN devido ao shift
    
    X = df_modelo[['Temp_Lag_24h']]
    y = df_modelo['Temperatura_C']
    
    # Divisão simples em treino (80%) e teste (20%)
    split_index = int(len(df_modelo) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Métricas do Modelo - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Retorna também dados para plotagem
    resultados_modelo = {
        'modelo': modelo,
        'mse': mse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test
    }
    
    return resultados_modelo

def exportar_resultados(df: pd.DataFrame, stats_dict: dict, modelo_resultados: dict):
    """
    Salva o DataFrame final em CSV e gera os gráficos solicitados.
    """
    logging.info("Exportando dados e gráficos.")
    
    # 1. Salvar CSV
    df.to_csv('resultados_climaticos.csv', encoding='utf-8')
    logging.info("Arquivo 'resultados_climaticos.csv' salvo com sucesso.")
    
    # Configuração de estilo do Seaborn para gráficos mais bonitos
    sns.set_theme(style="whitegrid")
    
    # 2. Gráfico de Linha: Evolução Temporal
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Temperatura_C'], label='Temperatura (°C)', color='tomato', alpha=0.8)
    plt.plot(df.index, df['Temperatura_C_MM24h'], label='Média Móvel 24h', color='darkred', linestyle='--')
    plt.title("Evolução Temporal da Temperatura")
    plt.xlabel("Data")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('evolucao_temporal.png', dpi=300)
    plt.close()
    
    # 3. Heatmap: Matriz de Correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(stats_dict['correlacao'], annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Matriz de Correlação de Pearson")
    plt.tight_layout()
    plt.savefig('heatmap_correlacao.png', dpi=300)
    plt.close()
    
    # 4. Gráfico de Dispersão: Valores Reais vs Previstos
    y_test = modelo_resultados['y_test']
    y_pred = modelo_resultados['y_pred']
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    
    # Linha de tendência ideal (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Linha de Tendência Ideal')
    
    plt.title(f"Dispersão Previsão de Temperatura (R²: {modelo_resultados['r2']:.2f})")
    plt.xlabel("Temperatura Real (°C)")
    plt.ylabel("Temperatura Prevista (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('scatter_previsao.png', dpi=300)
    plt.close()
    
    logging.info("Gráficos exportados com sucesso (arquivos .png).")

def main():
    # Coordenadas aproximadas de Alegrete-RS (UNIPAMPA)
    lat = -29.789
    lon = -55.795
    # Últimos 30 dias de histórico, por exemplo
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    df_bruto = coletar_dados_meteo(lat, lon, start_date, end_date)
    if df_bruto.empty:
        logging.error("O processo foi interrompido devido à falha na coleta.")
        return
        
    df_tratado = processar_dados(df_bruto)
    df_diario = gerar_resumo_diario(df_tratado)
    
    stats_dict = analisar_estatisticas(df_tratado)
    
    # Mostrando algumas estatísticas no console
    print("\n--- Estatísticas Descritivas (Amostra) ---")
    print(stats_dict['estatisticas'][['mean', 'std', 'min', 'max', 'skewness']])
    
    modelo_resultados = prever_temperatura(df_tratado)
    
    exportar_resultados(df_tratado, stats_dict, modelo_resultados)
    
    logging.info("Pipeline de Engenharia de Dados finalizado com sucesso.")

if __name__ == "__main__":
    main()
