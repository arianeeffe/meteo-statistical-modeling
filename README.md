# meteo-statistical-modeling

Este projeto em Python foi criado para automatizar a **coleta, manipulação, análise estatística e modelagem preditiva** de dados climáticos horários. 

Desenvolvido no contexto do Mestrado em Engenharia pela **UNIPAMPA**, o pipeline de Engenharia de Dados Científicos consome dados meteorológicos em tempo real da API [Open-Meteo](https://open-meteo.com/) e analisa variáveis importantes como Temperatura, Umidade Relativa, Velocidade do Vento e Radiação Solar.

## 🚀 Funcionalidades

- 🌤️ **Coleta de Dados:** Requisição de dados climáticos via integração com API.
- ⚙️ **Tratamento de Dados:** Ajuste de séries temporais, interpolação linear para lidar com valores ausentes e aplicação de médias móveis usando `Pandas` e `NumPy`.
- 📊 **Análise Matemática:** Geração de estatística descritiva completa, cálculo da Matriz de Correlação de Pearson e cômputo do **Índice de Calor (Heat Index)**.
- 🤖 **Modelagem Preditiva:** Aplicação de Regressão Linear (`Scikit-Learn`) para previsão climática de curto prazo com avaliação de métricas (MSE e R²).
- 📈 **Exportação e Dataviz:** Salvamento automático da base final em formato `.csv` e geração de gráficos analíticos (Evolução Temporal, Heatmap de Correlação e Dispersão) via `Matplotlib/Seaborn`.

## Executando

```bash
rm -rf .venv/

python3.14 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

python meteo_analysis.py
```