#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Engenharia de Dados Científicos para Climatologia
================================================================

Projeto para coleta, tratamento, análise matemática/estatística e
modelagem preditiva de dados climáticos horários, utilizando a API
pública Open-Meteo (https://open-meteo.com/).

O pipeline é organizado nas seguintes etapas, cada uma isolada em
funções bem definidas (single responsibility), seguindo a PEP 8:

    1. Coleta de dados       -> coletar_dados_climaticos()
    2. Manipulação/tratamento -> tratar_dados(), calcular_medias_moveis(),
                                  resumir_dados_diarios()
    3. Análise matemática     -> calcular_estatisticas_descritivas(),
                                  calcular_matriz_correlacao(),
                                  calcular_eto_hargreaves(),
                                  calcular_indice_calor()
    4. Modelagem preditiva    -> prever_variavel_regressao_linear()
    5. Exportação e gráficos  -> salvar_csv(), gerar_grafico_linha_temporal(),
                                  gerar_heatmap_correlacao(),
                                  gerar_grafico_dispersao_previsao()

Variáveis coletadas (dados horários): temperatura do ar a 2m,
umidade relativa a 2m, velocidade do vento a 10m e radiação solar
de ondas curtas (shortwave radiation).

Autora: Projeto estruturado para uso acadêmico (Mestrado em Engenharia -
UNIPAMPA). Ano: 2026. Licença: MIT.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica, adequado para geração de arquivos .png
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =============================================================================
# 0. CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Endpoint da API de arquivo histórico do Open-Meteo (dados horários já
# consolidados). Para previsões futuras em tempo real, a Open-Meteo
# disponibiliza também o endpoint "https://api.open-meteo.com/v1/forecast".
API_URL_ARQUIVO_HISTORICO = "https://archive-api.open-meteo.com/v1/archive"

# Variáveis horárias obrigatórias, conforme especificação do projeto.
VARIAVEIS_HORARIAS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]

# Nomes amigáveis das variáveis, usados em títulos de gráficos e relatórios.
NOMES_AMIGAVEIS = {
    "temperature_2m": "Temperatura do ar a 2m (°C)",
    "relative_humidity_2m": "Umidade Relativa a 2m (%)",
    "wind_speed_10m": "Velocidade do Vento a 10m (km/h)",
    "shortwave_radiation": "Radiação Solar (W/m²)",
}

# Diretório de saída para os artefatos gerados (CSV e gráficos .png).
DIRETORIO_SAIDA = Path("output")

# Constante solar utilizada no cálculo de radiação extraterrestre (FAO-56).
CONSTANTE_SOLAR_MJ_MIN = 0.0820  # MJ * m^-2 * min^-1

# Configuração do logger da aplicação: registra o andamento do pipeline
# tanto no console quanto em um arquivo de log local.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_climatico.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline_climatico")


# =============================================================================
# ESTRUTURA AUXILIAR PARA RESULTADOS DA MODELAGEM PREDITIVA
# =============================================================================

@dataclass
class ResultadoPrevisao:
    """Agrupa os artefatos produzidos pela modelagem preditiva.

    Atributos:
        variavel: nome da variável meteorológica prevista.
        modelo: instância treinada de LinearRegression.
        mse: Erro Quadrático Médio calculado no conjunto de teste.
        r2: Coeficiente de Determinação (R²) no conjunto de teste.
        y_teste_real: valores reais do conjunto de teste (últimas horas
            conhecidas), usados para validar o modelo.
        y_teste_previsto: valores previstos pelo modelo para o mesmo
            período do conjunto de teste.
        df_previsao_futura: DataFrame com o horizonte de previsão futura
            (horas ainda não observadas no histórico coletado).
    """

    variavel: str
    modelo: LinearRegression
    mse: float
    r2: float
    y_teste_real: np.ndarray
    y_teste_previsto: np.ndarray
    df_previsao_futura: pd.DataFrame


# =============================================================================
# 1. COLETA DE DADOS (API OPEN-METEO)
# =============================================================================

def coletar_dados_climaticos(
    latitude: float,
    longitude: float,
    data_inicio: str,
    data_fim: str,
    variaveis: Optional[list[str]] = None,
    tentativas: int = 3,
    timeout_segundos: int = 30,
) -> pd.DataFrame:
    """Coleta dados climáticos horários históricos na API Open-Meteo.

    Realiza uma requisição HTTP GET ao endpoint de arquivo histórico da
    Open-Meteo e converte a resposta JSON diretamente em um DataFrame
    do Pandas.

    Args:
        latitude: latitude do ponto de interesse (graus decimais).
        longitude: longitude do ponto de interesse (graus decimais).
        data_inicio: data inicial no formato "AAAA-MM-DD".
        data_fim: data final no formato "AAAA-MM-DD".
        variaveis: lista de variáveis horárias a solicitar; usa
            VARIAVEIS_HORARIAS por padrão caso não seja informada.
        tentativas: número máximo de tentativas em caso de falha de rede.
        timeout_segundos: tempo limite (em segundos) para a requisição.

    Returns:
        DataFrame do Pandas com uma coluna "time" e uma coluna para
        cada variável meteorológica solicitada.

    Raises:
        requests.exceptions.RequestException: se todas as tentativas de
            requisição HTTP falharem.
        ValueError: se a resposta da API não contiver os dados horários
            esperados.
    """
    variaveis = variaveis or VARIAVEIS_HORARIAS
    parametros = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": data_inicio,
        "end_date": data_fim,
        "hourly": ",".join(variaveis),
        "timezone": "auto",
        "wind_speed_unit": "kmh",
    }

    ultimo_erro: Optional[Exception] = None
    for tentativa_atual in range(1, tentativas + 1):
        try:
            logger.info(
                "Solicitando dados climáticos à Open-Meteo (tentativa %d/%d)...",
                tentativa_atual,
                tentativas,
            )
            resposta = requests.get(
                API_URL_ARQUIVO_HISTORICO, params=parametros, timeout=timeout_segundos
            )
            resposta.raise_for_status()  # levanta exceção para códigos HTTP 4xx/5xx
            dados_json = resposta.json()

            if "hourly" not in dados_json:
                raise ValueError(
                    "A resposta da API não contém a seção 'hourly' esperada. "
                    f"Resposta recebida: {dados_json}"
                )

            dataframe_bruto = pd.DataFrame(dados_json["hourly"])
            logger.info(
                "Coleta concluída com sucesso: %d registros horários obtidos.",
                len(dataframe_bruto),
            )
            return dataframe_bruto

        except requests.exceptions.RequestException as erro_requisicao:
            ultimo_erro = erro_requisicao
            logger.warning(
                "Falha na tentativa %d/%d de requisição à API: %s",
                tentativa_atual,
                tentativas,
                erro_requisicao,
            )
        except ValueError as erro_valor:
            # Erro de formato de resposta não deve ser reprocessado.
            logger.error("Resposta inválida da API Open-Meteo: %s", erro_valor)
            raise

    # Caso todas as tentativas tenham falhado, propaga o último erro.
    logger.error("Não foi possível coletar os dados após %d tentativas.", tentativas)
    raise requests.exceptions.RequestException(
        f"Falha na comunicação com a API Open-Meteo: {ultimo_erro}"
    )


# =============================================================================
# 2. MANIPULAÇÃO E TRATAMENTO DE DADOS (PANDAS/NUMPY)
# =============================================================================

def tratar_dados(dataframe_bruto: pd.DataFrame) -> pd.DataFrame:
    """Prepara o DataFrame bruto coletado da API para análise.

    Converte a coluna "time" para datetime, define-a como índice
    temporal e trata valores ausentes (NaN) por interpolação linear.

    Args:
        dataframe_bruto: DataFrame retornado por coletar_dados_climaticos().

    Returns:
        DataFrame tratado, indexado por datetime, sem valores ausentes
        (quando possível interpolar).
    """
    df = dataframe_bruto.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    quantidade_nan_antes = int(df.isna().sum().sum())
    if quantidade_nan_antes > 0:
        logger.info(
            "Detectados %d valores ausentes; aplicando interpolação linear.",
            quantidade_nan_antes,
        )
        # Interpolação linear ao longo do tempo, com preenchimento nas
        # extremidades (limit_direction="both") para eventuais lacunas
        # no início ou no fim da série.
        df = df.interpolate(method="linear", limit_direction="both")

    quantidade_nan_depois = int(df.isna().sum().sum())
    if quantidade_nan_depois > 0:
        logger.warning(
            "Ainda restam %d valores ausentes após a interpolação.",
            quantidade_nan_depois,
        )

    return df


def calcular_medias_moveis(df: pd.DataFrame, janela_horas: int = 24) -> pd.DataFrame:
    """Adiciona colunas de média móvel para suavização de tendências.

    Args:
        df: DataFrame tratado, indexado por datetime.
        janela_horas: tamanho da janela da média móvel, em horas.

    Returns:
        DataFrame original acrescido de uma coluna "<variavel>_media_movel_24h"
        para cada variável meteorológica numérica presente.
    """
    df_resultado = df.copy()
    colunas_numericas = df_resultado.select_dtypes(include=[np.number]).columns

    for coluna in colunas_numericas:
        nome_nova_coluna = f"{coluna}_media_movel_24h"
        df_resultado[nome_nova_coluna] = (
            df_resultado[coluna].rolling(window=janela_horas, min_periods=1).mean()
        )

    logger.info("Médias móveis de %dh calculadas para %d variáveis.", janela_horas, len(colunas_numericas))
    return df_resultado


def resumir_dados_diarios(df: pd.DataFrame, variaveis: Optional[list[str]] = None) -> pd.DataFrame:
    """Resume os dados horários em agregados diários (média, máximo e mínimo).

    Args:
        df: DataFrame horário tratado, indexado por datetime.
        variaveis: variáveis originais a agregar; usa VARIAVEIS_HORARIAS
            por padrão.

    Returns:
        DataFrame indexado por dia, com colunas "<variavel>_media",
        "<variavel>_maximo" e "<variavel>_minimo".
    """
    variaveis = variaveis or VARIAVEIS_HORARIAS
    variaveis_presentes = [v for v in variaveis if v in df.columns]

    agregacao = df[variaveis_presentes].resample("D").agg(["mean", "max", "min"])
    # Achata o índice hierárquico de colunas (ex.: ("temperature_2m", "mean"))
    # para nomes legíveis (ex.: "temperature_2m_media").
    traducao_estatistica = {"mean": "media", "max": "maximo", "min": "minimo"}
    agregacao.columns = [
        f"{variavel}_{traducao_estatistica[estatistica]}"
        for variavel, estatistica in agregacao.columns
    ]
    agregacao.index.name = "data"

    logger.info("Resumo diário gerado com %d dias.", len(agregacao))
    return agregacao


# =============================================================================
# 3. ANÁLISE MATEMÁTICA E ESTATÍSTICA
# =============================================================================

def calcular_estatisticas_descritivas(
    df: pd.DataFrame, variaveis: Optional[list[str]] = None
) -> pd.DataFrame:
    """Calcula estatísticas descritivas completas para as variáveis informadas.

    Args:
        df: DataFrame contendo as variáveis meteorológicas.
        variaveis: colunas a analisar; usa VARIAVEIS_HORARIAS por padrão.

    Returns:
        DataFrame com uma linha por variável e colunas: média, mediana,
        desvio_padrao, variancia, assimetria (skewness) e curtose (kurtosis).
    """
    variaveis = variaveis or VARIAVEIS_HORARIAS
    variaveis_presentes = [v for v in variaveis if v in df.columns]

    estatisticas = pd.DataFrame(
        {
            "media": df[variaveis_presentes].mean(),
            "mediana": df[variaveis_presentes].median(),
            "desvio_padrao": df[variaveis_presentes].std(),
            "variancia": df[variaveis_presentes].var(),
            # Assimetria (skewness) e curtose (kurtosis) medem,
            # respectivamente, a simetria e o "achatamento" da distribuição
            # em relação à distribuição normal.
            "assimetria_skewness": df[variaveis_presentes].skew(),
            "curtose_kurtosis": df[variaveis_presentes].kurtosis(),
        }
    )
    logger.info("Estatísticas descritivas calculadas para %d variáveis.", len(variaveis_presentes))
    return estatisticas


def calcular_matriz_correlacao(
    df: pd.DataFrame, variaveis: Optional[list[str]] = None
) -> pd.DataFrame:
    """Calcula a matriz de correlação de Pearson entre variáveis meteorológicas.

    Args:
        df: DataFrame contendo as variáveis meteorológicas.
        variaveis: colunas a correlacionar; usa VARIAVEIS_HORARIAS por padrão.

    Returns:
        DataFrame quadrado com os coeficientes de correlação de Pearson
        (valores entre -1 e 1) entre cada par de variáveis.
    """
    variaveis = variaveis or VARIAVEIS_HORARIAS
    variaveis_presentes = [v for v in variaveis if v in df.columns]
    matriz_correlacao = df[variaveis_presentes].corr(method="pearson")
    logger.info("Matriz de correlação de Pearson calculada.")
    return matriz_correlacao


def calcular_radiacao_extraterrestre(latitude_graus: float, dia_do_ano: np.ndarray) -> np.ndarray:
    """Calcula a radiação solar extraterrestre diária (Ra), em MJ/m²/dia.

    Implementa as equações padrão da FAO-56 (Allen et al., 1998), que
    dependem apenas da latitude do local e do dia juliano do ano (1 a 366).
    Este valor de Ra é utilizado como insumo para o cálculo da
    evapotranspiração potencial pelo método de Hargreaves-Samani.

    Args:
        latitude_graus: latitude do local, em graus decimais.
        dia_do_ano: array com os dias julianos (1 a 366) de cada observação.

    Returns:
        Array numpy com a radiação extraterrestre (Ra) em MJ/m²/dia,
        para cada dia informado.
    """
    latitude_rad = np.radians(latitude_graus)

    # Distância relativa inversa Terra-Sol (dr) e declinação solar (delta).
    distancia_relativa_inversa = 1 + 0.033 * np.cos(2 * np.pi * dia_do_ano / 365)
    declinacao_solar = 0.409 * np.sin(2 * np.pi * dia_do_ano / 365 - 1.39)

    # Ângulo horário do nascer do sol (ws), limitado ao intervalo válido
    # de arco-cosseno para evitar erros numéricos em latitudes extremas.
    argumento_ws = np.clip(
        -np.tan(latitude_rad) * np.tan(declinacao_solar), -1.0, 1.0
    )
    angulo_horario_por_do_sol = np.arccos(argumento_ws)

    # Equação de radiação extraterrestre diária (FAO-56, Eq. 21).
    radiacao_extraterrestre = (
        (24 * 60 / np.pi)
        * CONSTANTE_SOLAR_MJ_MIN
        * distancia_relativa_inversa
        * (
            angulo_horario_por_do_sol * np.sin(latitude_rad) * np.sin(declinacao_solar)
            + np.cos(latitude_rad) * np.cos(declinacao_solar) * np.sin(angulo_horario_por_do_sol)
        )
    )
    return radiacao_extraterrestre


def calcular_eto_hargreaves(df_diario: pd.DataFrame, latitude_graus: float) -> pd.Series:
    """Calcula a Evapotranspiração Potencial (ETo) diária simplificada.

    Utiliza o método empírico de Hargreaves-Samani (1985), que estima a
    ETo (em mm/dia) a partir apenas de temperaturas máxima, mínima e
    média do ar e da radiação extraterrestre (Ra) — sem exigir dados de
    vento ou pressão de vapor, por isso é considerado um método
    "simplificado" em relação ao Penman-Monteith completo (FAO-56).

    Fórmula: ETo = 0,0023 * (Tmed + 17,8) * sqrt(Tmax - Tmin) * Ra_mm

    onde Ra_mm é a radiação extraterrestre convertida para equivalente
    de evaporação em mm/dia (Ra_mm = 0,408 * Ra[MJ/m²/dia]).

    Args:
        df_diario: DataFrame diário contendo as colunas
            "temperature_2m_media", "temperature_2m_maximo" e
            "temperature_2m_minimo" (gerado por resumir_dados_diarios()).
        latitude_graus: latitude do local, em graus decimais.

    Returns:
        Series indexada por data com a ETo estimada, em mm/dia.
    """
    dia_do_ano = df_diario.index.dayofyear.values
    radiacao_extraterrestre_mj = calcular_radiacao_extraterrestre(latitude_graus, dia_do_ano)
    radiacao_extraterrestre_mm = 0.408 * radiacao_extraterrestre_mj

    temperatura_media = df_diario["temperature_2m_media"].values
    temperatura_maxima = df_diario["temperature_2m_maximo"].values
    temperatura_minima = df_diario["temperature_2m_minimo"].values

    # A amplitude térmica (Tmax - Tmin) nunca deve ser negativa; valores
    # muito pequenos são mantidos, mas negativos indicariam erro de dados.
    amplitude_termica = np.clip(temperatura_maxima - temperatura_minima, a_min=0, a_max=None)

    eto_mm_dia = (
        0.0023
        * (temperatura_media + 17.8)
        * np.sqrt(amplitude_termica)
        * radiacao_extraterrestre_mm
    )

    serie_eto = pd.Series(eto_mm_dia, index=df_diario.index, name="eto_hargreaves_mm_dia")
    logger.info("ETo (Hargreaves-Samani) calculada para %d dias.", len(serie_eto))
    return serie_eto


def calcular_indice_calor(temperatura_c: np.ndarray, umidade_relativa: np.ndarray) -> np.ndarray:
    """Calcula o Índice de Calor (sensação térmica) a partir de temperatura e umidade.

    Implementa a regressão de Rothfusz (National Weather Service, EUA),
    amplamente utilizada para estimar a sensação térmica em condições de
    calor. A fórmula original opera em graus Fahrenheit; por isso, os
    dados são convertidos antes e depois do cálculo.

    Observação: a regressão de Rothfusz é uma aproximação válida
    principalmente para temperaturas aproximadamente acima de 27 °C; para
    valores mais baixos, o resultado tende à própria temperatura do ar,
    o que é aceitável para fins de suavização/visualização da série.

    Args:
        temperatura_c: array de temperaturas do ar, em graus Celsius.
        umidade_relativa: array de umidade relativa do ar, em percentual (0-100).

    Returns:
        Array numpy com o índice de calor estimado, em graus Celsius.
    """
    temperatura_f = temperatura_c * 9 / 5 + 32
    umidade = umidade_relativa

    indice_calor_f = (
        -42.379
        + 2.04901523 * temperatura_f
        + 10.14333127 * umidade
        - 0.22475541 * temperatura_f * umidade
        - 0.00683783 * temperatura_f**2
        - 0.05481717 * umidade**2
        + 0.00122874 * temperatura_f**2 * umidade
        + 0.00085282 * temperatura_f * umidade**2
        - 0.00199788 * temperatura_f**2 * umidade**2
    )

    # Para temperaturas baixas, a fórmula de Rothfusz perde validade física;
    # nesses casos, adota-se a própria temperatura do ar como aproximação
    # do índice de calor (comportamento padrão de referências meteorológicas).
    indice_calor_f = np.where(temperatura_f < 80, temperatura_f, indice_calor_f)

    indice_calor_c = (indice_calor_f - 32) * 5 / 9
    return indice_calor_c


# =============================================================================
# 4. MODELAGEM E PREVISÃO (ANÁLISE PREDITIVA)
# =============================================================================

def _construir_features_temporais(indice_datetime: pd.DatetimeIndex, inicio_serie: pd.Timestamp) -> np.ndarray:
    """Constrói a matriz de atributos (features) usada pelo modelo de regressão.

    Combina uma tendência linear (número de horas desde o início da série)
    com componentes senoidais que representam o ciclo diurno de 24 horas,
    permitindo que o modelo capture tanto a tendência quanto a
    periodicidade diária típica de temperatura e radiação solar.

    Args:
        indice_datetime: índice de datas/horas para o qual gerar as features.
        inicio_serie: timestamp de referência (t=0) usado para calcular a
            tendência temporal linear.

    Returns:
        Array numpy de shape (n_amostras, 3): [horas_desde_inicio, seno_hora, cosseno_hora].
    """
    horas_desde_inicio = (indice_datetime - inicio_serie) / pd.Timedelta(hours=1)
    hora_do_dia = indice_datetime.hour + indice_datetime.minute / 60
    ciclo_diario_seno = np.sin(2 * np.pi * hora_do_dia / 24)
    ciclo_diario_cosseno = np.cos(2 * np.pi * hora_do_dia / 24)

    return np.column_stack(
        [horas_desde_inicio.to_numpy(), ciclo_diario_seno, ciclo_diario_cosseno]
    )


def prever_variavel_regressao_linear(
    df: pd.DataFrame,
    variavel: str = "temperature_2m",
    horas_previsao: int = 24,
) -> ResultadoPrevisao:
    """Projeta uma variável meteorológica para as próximas horas via Regressão Linear.

    O modelo utiliza como atributos (features) a tendência temporal linear
    e componentes senoidais do ciclo diurno de 24 horas, permitindo captar
    tanto a tendência de médio prazo quanto a variação típica entre dia e
    noite. As últimas `horas_previsao` observações do histórico são
    reservadas como conjunto de teste (validação fora da amostra,
    respeitando a ordem cronológica dos dados).

    Args:
        df: DataFrame horário tratado (índice datetime), contendo a
            coluna `variavel`.
        variavel: nome da coluna a ser prevista (ex.: "temperature_2m").
        horas_previsao: quantidade de horas futuras a projetar, e também
            o tamanho do conjunto de teste usado na validação do modelo.

    Returns:
        ResultadoPrevisao com o modelo treinado, métricas de erro (MSE, R²)
        e os DataFrames de previsão futura.

    Raises:
        ValueError: se não houver dados suficientes para treinar e validar
            o modelo com o horizonte solicitado.
    """
    if variavel not in df.columns:
        raise ValueError(f"A variável '{variavel}' não existe no DataFrame informado.")

    serie = df[variavel].dropna()
    if len(serie) <= horas_previsao * 2:
        raise ValueError(
            "Dados insuficientes para treinar e validar o modelo com o "
            f"horizonte de {horas_previsao}h. São necessárias pelo menos "
            f"{horas_previsao * 2 + 1} observações; há apenas {len(serie)}."
        )

    inicio_serie = serie.index[0]
    matriz_features = _construir_features_temporais(serie.index, inicio_serie)
    vetor_alvo = serie.to_numpy()

    # Divisão cronológica (sem embaralhar): as últimas `horas_previsao`
    # observações conhecidas formam o conjunto de teste, simulando a
    # validação do modelo em um cenário real de previsão de curto prazo.
    indice_corte = len(serie) - horas_previsao
    x_treino, x_teste = matriz_features[:indice_corte], matriz_features[indice_corte:]
    y_treino, y_teste = vetor_alvo[:indice_corte], vetor_alvo[indice_corte:]

    modelo = LinearRegression()
    modelo.fit(x_treino, y_treino)

    y_teste_previsto = modelo.predict(x_teste)
    erro_quadratico_medio = mean_squared_error(y_teste, y_teste_previsto)
    coeficiente_determinacao = r2_score(y_teste, y_teste_previsto)

    logger.info(
        "Modelo de regressão linear treinado para '%s': MSE=%.4f | R²=%.4f",
        variavel,
        erro_quadratico_medio,
        coeficiente_determinacao,
    )

    # Geração do horizonte de previsão futura (horas ainda não observadas).
    ultimo_timestamp = serie.index[-1]
    datas_futuras = pd.date_range(
        start=ultimo_timestamp + pd.Timedelta(hours=1), periods=horas_previsao, freq="h"
    )
    matriz_features_futuras = _construir_features_temporais(datas_futuras, inicio_serie)
    valores_previstos_futuros = modelo.predict(matriz_features_futuras)

    df_previsao_futura = pd.DataFrame(
        {f"{variavel}_previsto": valores_previstos_futuros}, index=datas_futuras
    )
    df_previsao_futura.index.name = "time"

    return ResultadoPrevisao(
        variavel=variavel,
        modelo=modelo,
        mse=float(erro_quadratico_medio),
        r2=float(coeficiente_determinacao),
        y_teste_real=y_teste,
        y_teste_previsto=y_teste_previsto,
        df_previsao_futura=df_previsao_futura,
    )


# =============================================================================
# 5. EXPORTAÇÃO DE DADOS E GERAÇÃO DE GRÁFICOS
# =============================================================================

def salvar_csv(df: pd.DataFrame, caminho_arquivo: Path) -> None:
    """Salva um DataFrame em formato CSV, com codificação UTF-8.

    Args:
        df: DataFrame a ser exportado.
        caminho_arquivo: caminho completo do arquivo .csv de destino.
    """
    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_arquivo, encoding="utf-8", index=True)
    logger.info("Arquivo CSV salvo em: %s", caminho_arquivo)


def gerar_grafico_linha_temporal(
    df: pd.DataFrame, variaveis: list[str], caminho_arquivo: Path
) -> None:
    """Gera e salva um gráfico de linha com a evolução temporal das variáveis.

    Cada variável é plotada em um subgráfico próprio (eixo Y independente),
    já que possuem unidades de medida distintas.

    Args:
        df: DataFrame horário tratado, indexado por datetime.
        variaveis: lista de colunas a plotar.
        caminho_arquivo: caminho completo do arquivo .png de destino.
    """
    sns.set_theme(style="whitegrid")
    figura, eixos = plt.subplots(len(variaveis), 1, figsize=(12, 3 * len(variaveis)), sharex=True)
    if len(variaveis) == 1:
        eixos = [eixos]

    for eixo, variavel in zip(eixos, variaveis):
        eixo.plot(df.index, df[variavel], color="#2563eb", linewidth=1)
        eixo.set_ylabel(NOMES_AMIGAVEIS.get(variavel, variavel))
        eixo.set_title(NOMES_AMIGAVEIS.get(variavel, variavel), fontsize=10)

    eixos[-1].set_xlabel("Data/Hora")
    figura.suptitle("Evolução Temporal das Variáveis Climáticas", fontsize=14, y=1.0)
    figura.tight_layout()

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho_arquivo, dpi=150, bbox_inches="tight")
    plt.close(figura)
    logger.info("Gráfico de linha temporal salvo em: %s", caminho_arquivo)


def gerar_heatmap_correlacao(matriz_correlacao: pd.DataFrame, caminho_arquivo: Path) -> None:
    """Gera e salva um heatmap da matriz de correlação de Pearson.

    Args:
        matriz_correlacao: DataFrame quadrado com os coeficientes de
            correlação (saída de calcular_matriz_correlacao()).
        caminho_arquivo: caminho completo do arquivo .png de destino.
    """
    sns.set_theme(style="white")
    figura, eixo = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matriz_correlacao,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=eixo,
    )
    eixo.set_title("Matriz de Correlação de Pearson entre Variáveis Climáticas")
    figura.tight_layout()

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho_arquivo, dpi=150, bbox_inches="tight")
    plt.close(figura)
    logger.info("Heatmap de correlação salvo em: %s", caminho_arquivo)


def gerar_grafico_dispersao_previsao(
    resultado: ResultadoPrevisao, caminho_arquivo: Path
) -> None:
    """Gera um gráfico de dispersão (real vs. previsto) com linha de tendência.

    Compara os valores reais e previstos do conjunto de teste (últimas
    horas conhecidas do histórico), permitindo avaliar visualmente a
    qualidade do ajuste do modelo de regressão linear.

    Args:
        resultado: objeto ResultadoPrevisao retornado por
            prever_variavel_regressao_linear().
        caminho_arquivo: caminho completo do arquivo .png de destino.
    """
    sns.set_theme(style="whitegrid")
    figura, eixo = plt.subplots(figsize=(7, 6))

    eixo.scatter(
        resultado.y_teste_real,
        resultado.y_teste_previsto,
        color="#2563eb",
        alpha=0.7,
        label="Observações (teste)",
    )

    # Linha de tendência ideal (y = x): quanto mais próximos os pontos
    # estiverem dessa linha, melhor a qualidade da previsão do modelo.
    valor_minimo = min(resultado.y_teste_real.min(), resultado.y_teste_previsto.min())
    valor_maximo = max(resultado.y_teste_real.max(), resultado.y_teste_previsto.max())
    eixo.plot(
        [valor_minimo, valor_maximo],
        [valor_minimo, valor_maximo],
        color="#dc2626",
        linestyle="--",
        label="Ajuste perfeito (y = x)",
    )

    nome_variavel = NOMES_AMIGAVEIS.get(resultado.variavel, resultado.variavel)
    eixo.set_xlabel(f"{nome_variavel} — Valor real")
    eixo.set_ylabel(f"{nome_variavel} — Valor previsto")
    eixo.set_title(
        f"Previsão via Regressão Linear — {nome_variavel}\n"
        f"MSE = {resultado.mse:.3f} | R² = {resultado.r2:.3f}"
    )
    eixo.legend()
    figura.tight_layout()

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(caminho_arquivo, dpi=150, bbox_inches="tight")
    plt.close(figura)
    logger.info("Gráfico de dispersão da previsão salvo em: %s", caminho_arquivo)


# =============================================================================
# ORQUESTRAÇÃO DO PIPELINE (FUNÇÃO PRINCIPAL)
# =============================================================================

def analisar_argumentos_linha_comando() -> argparse.Namespace:
    """Define e interpreta os argumentos de linha de comando do script.

    Os valores padrão apontam para Bagé/RS (sede da UNIPAMPA) e para um
    intervalo recente de 30 dias, respeitando a defasagem típica de
    consolidação de dados históricos da Open-Meteo (cerca de 5 dias).

    Returns:
        Namespace com os argumentos latitude, longitude, data_inicio e
        data_fim já validados/preenchidos.
    """
    data_fim_padrao = date.today() - timedelta(days=5)
    data_inicio_padrao = data_fim_padrao - timedelta(days=30)

    analisador = argparse.ArgumentParser(
        description="Pipeline de coleta, análise e previsão de dados climáticos (Open-Meteo)."
    )
    analisador.add_argument("--latitude", type=float, default=-31.33, help="Latitude do local (padrão: Bagé/RS).")
    analisador.add_argument("--longitude", type=float, default=-54.11, help="Longitude do local (padrão: Bagé/RS).")
    analisador.add_argument(
        "--data-inicio",
        type=str,
        default=data_inicio_padrao.isoformat(),
        help="Data inicial no formato AAAA-MM-DD.",
    )
    analisador.add_argument(
        "--data-fim",
        type=str,
        default=data_fim_padrao.isoformat(),
        help="Data final no formato AAAA-MM-DD.",
    )
    return analisador.parse_args()


def main() -> None:
    """Executa o pipeline completo: coleta, tratamento, análise, modelagem e exportação."""
    argumentos = analisar_argumentos_linha_comando()
    logger.info(
        "Iniciando pipeline climático | lat=%.4f lon=%.4f período=%s a %s",
        argumentos.latitude,
        argumentos.longitude,
        argumentos.data_inicio,
        argumentos.data_fim,
    )

    # ---------------------------------------------------------------
    # Etapa 1: Coleta de dados
    # ---------------------------------------------------------------
    try:
        df_bruto = coletar_dados_climaticos(
            latitude=argumentos.latitude,
            longitude=argumentos.longitude,
            data_inicio=argumentos.data_inicio,
            data_fim=argumentos.data_fim,
        )
    except (requests.exceptions.RequestException, ValueError) as erro:
        logger.error("Pipeline interrompido: falha na coleta de dados. Detalhe: %s", erro)
        sys.exit(1)

    # ---------------------------------------------------------------
    # Etapa 2: Manipulação e tratamento
    # ---------------------------------------------------------------
    df_tratado = tratar_dados(df_bruto)
    df_com_medias_moveis = calcular_medias_moveis(df_tratado, janela_horas=24)
    df_diario = resumir_dados_diarios(df_tratado)

    # ---------------------------------------------------------------
    # Etapa 3: Análise matemática e estatística
    # ---------------------------------------------------------------
    estatisticas_descritivas = calcular_estatisticas_descritivas(df_tratado)
    matriz_correlacao = calcular_matriz_correlacao(df_tratado)

    serie_eto = calcular_eto_hargreaves(df_diario, latitude_graus=argumentos.latitude)
    df_diario["eto_hargreaves_mm_dia"] = serie_eto

    df_com_medias_moveis["indice_calor_c"] = calcular_indice_calor(
        df_com_medias_moveis["temperature_2m"].to_numpy(),
        df_com_medias_moveis["relative_humidity_2m"].to_numpy(),
    )

    print("\n=== Estatísticas Descritivas ===")
    print(estatisticas_descritivas.round(3))
    print("\n=== Matriz de Correlação de Pearson ===")
    print(matriz_correlacao.round(3))
    print("\n=== ETo (Hargreaves-Samani) — últimos dias ===")
    print(serie_eto.round(3).tail())

    # ---------------------------------------------------------------
    # Etapa 4: Modelagem preditiva
    # ---------------------------------------------------------------
    try:
        resultado_previsao_temperatura = prever_variavel_regressao_linear(
            df_tratado, variavel="temperature_2m", horas_previsao=24
        )
    except ValueError as erro:
        logger.error("Não foi possível gerar a previsão: %s", erro)
        sys.exit(1)

    print("\n=== Modelagem Preditiva (Regressão Linear) — Temperatura ===")
    print(f"MSE (Erro Quadrático Médio): {resultado_previsao_temperatura.mse:.4f}")
    print(f"R² (Coeficiente de Determinação): {resultado_previsao_temperatura.r2:.4f}")
    print(resultado_previsao_temperatura.df_previsao_futura.round(2).head())

    # ---------------------------------------------------------------
    # Etapa 5: Exportação de dados e geração de gráficos
    # ---------------------------------------------------------------
    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)

    df_final_exportacao = df_com_medias_moveis.copy()
    salvar_csv(df_final_exportacao, DIRETORIO_SAIDA / "dados_climaticos_processados.csv")
    salvar_csv(df_diario, DIRETORIO_SAIDA / "resumo_diario.csv")
    salvar_csv(estatisticas_descritivas, DIRETORIO_SAIDA / "estatisticas_descritivas.csv")
    salvar_csv(matriz_correlacao, DIRETORIO_SAIDA / "matriz_correlacao.csv")
    salvar_csv(
        resultado_previsao_temperatura.df_previsao_futura,
        DIRETORIO_SAIDA / "previsao_temperatura_24h.csv",
    )

    gerar_grafico_linha_temporal(
        df_tratado, VARIAVEIS_HORARIAS, DIRETORIO_SAIDA / "grafico_linha_temporal.png"
    )
    gerar_heatmap_correlacao(matriz_correlacao, DIRETORIO_SAIDA / "heatmap_correlacao.png")
    gerar_grafico_dispersao_previsao(
        resultado_previsao_temperatura, DIRETORIO_SAIDA / "dispersao_previsao_temperatura.png"
    )

    logger.info("Pipeline concluído com sucesso. Artefatos disponíveis em: %s", DIRETORIO_SAIDA.resolve())


if __name__ == "__main__":
    main()