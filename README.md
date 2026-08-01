# Pipeline Climático — Open-Meteo

Pipeline de Engenharia de Dados Científicos para coleta, tratamento,
análise matemática/estatística e modelagem preditiva de dados
climáticos horários, utilizando a API pública [Open-Meteo](https://open-meteo.com/).

## Funcionalidades

- **Coleta**: requisição HTTP à Archive API do Open-Meteo (temperatura a
  2m, umidade relativa a 2m, velocidade do vento a 10m e radiação
  solar), com tentativas automáticas em caso de falha de rede.
- **Tratamento**: conversão da coluna de tempo para `datetime` (índice),
  interpolação linear de valores ausentes, médias móveis de 24h e
  resumo diário (média, máximo e mínimo).
- **Análise matemática e estatística**:
  - Estatísticas descritivas completas (média, mediana, desvio padrão,
    variância, assimetria e curtose);
  - Matriz de correlação de Pearson entre todas as variáveis;
  - Evapotranspiração Potencial (ETo) pelo método de
    Hargreaves-Samani, com cálculo da radiação extraterrestre (Ra)
    via equações FAO-56;
  - Índice de Calor (regressão de Rothfusz).
- **Modelagem preditiva**: Regressão Linear (scikit-learn) com features
  de tendência temporal e ciclo diurno (seno/cosseno de 24h), projetando
  as próximas 24 horas de temperatura, com métricas MSE e R².
- **Exportação**: CSVs (UTF-8) dos dados processados, resumo diário,
  estatísticas, correlação e previsão; gráficos em `.png` (linha
  temporal, heatmap de correlação e dispersão da previsão).

## Estrutura do Projeto

```text
clima_unipampa/
├── main.py            # todo o pipeline (coleta, tratamento, análise, modelagem, exportação)
├── requirements.txt   # dependências do projeto
└── README.md
```

Todo o código está organizado em `main.py` por seções numeradas
(coleta, manipulação, análise, modelagem, exportação), com funções
isoladas, type hints e docstrings em português.

Os artefatos gerados na execução (CSVs e gráficos) são salvos em:

```text
output/
├── dados_climaticos_processados.csv
├── resumo_diario.csv
├── estatisticas_descritivas.csv
├── matriz_correlacao.csv
├── previsao_temperatura_24h.csv
├── grafico_linha_temporal.png
├── heatmap_correlacao.png
└── dispersao_previsao_temperatura.png
```

O log de execução é salvo em `pipeline_climatico.log`, na raiz do projeto.

## Como executar

Requisitos: Python 3.11+ e conexão com a internet (para consultar a API).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Por padrão, o script coleta os últimos 30 dias de dados para
Bagé/RS (sede da UNIPAMPA). É possível informar outro local e período
via argumentos de linha de comando:

```bash
python main.py \
  --latitude -30.03 \
  --longitude -51.23 \
  --data-inicio 2026-06-01 \
  --data-fim 2026-06-30
```

| Argumento        | Descrição                              | Padrão                     |
|-------------------|-----------------------------------------|-----------------------------|
| `--latitude`      | Latitude do local (graus decimais)      | `-31.33` (Bagé/RS)          |
| `--longitude`     | Longitude do local (graus decimais)     | `-54.11` (Bagé/RS)          |
| `--data-inicio`   | Data inicial (AAAA-MM-DD)               | Hoje - 35 dias               |
| `--data-fim`      | Data final (AAAA-MM-DD)                 | Hoje - 5 dias                |

> A data final padrão considera uma margem de 5 dias em relação à data
> atual, respeitando a defasagem típica de consolidação dos dados
> históricos na Open-Meteo.

## Limitações

- O modelo de regressão linear é um baseline simples (tendência +
  ciclo diurno); não substitui modelos de séries temporais mais
  robustos (ex.: SARIMA, Prophet) para previsões de médio/longo prazo.
- A ETo por Hargreaves-Samani é uma estimativa simplificada em relação
  ao método de referência Penman-Monteith (FAO-56), por não considerar
  vento e pressão de vapor.
- O Índice de Calor (Rothfusz) é uma aproximação válida principalmente
  para temperaturas acima de ~27 °C.

## Melhorias futuras

- Modelos de séries temporais com sazonalidade (SARIMA, Prophet).
- Cálculo de ETo pelo método completo de Penman-Monteith (FAO-56).
- Interface de linha de comando mais completa (ex.: seleção de
  variável a prever, horizonte de previsão configurável).
- Testes automatizados (pytest) para as funções de cálculo.

---

**Autora:** Ariane Fernandes · **Licença:** MIT · **Ano:** 2026