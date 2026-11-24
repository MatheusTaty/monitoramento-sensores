
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Configuração da página
st.set_page_config(
    page_title="Monitoramento de Sensores",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sistema de Monitoramento de Sensores")
st.write(
    "Aplicação de análise de dados e inteligência artificial "
    "para previsão de falhas em equipamentos."
)

# Localização da base
ARQUIVO = Path("data") / "sensores.csv"

# Carregamento dos dados
@st.cache_data
def carregar_dados():
    dados = pd.read_csv(ARQUIVO)

    colunas = [
        "timestamp",
        "temperatura",
        "umidade",
        "vibracao",
        "falha"
    ]

    if not all(coluna in dados.columns for coluna in colunas):
        raise ValueError("O CSV não possui todas as colunas necessárias.")

    dados["falha"] = pd.to_numeric(dados["falha"], errors="coerce")

    dados = dados.dropna(subset=colunas).copy()

    return dados


try:
    df = carregar_dados()
except Exception as erro:
    st.error(f"Não foi possível carregar a base de dados: {erro}")
    st.stop()

# Preparação do modelo
X = df[["temperatura", "umidade", "vibracao"]]
y = df["falha"].astype(int)

if y.nunique() < 2:
    st.error("A base precisa conter leituras normais e leituras de falha.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)

# Métricas
acuracia = accuracy_score(y_test, previsoes)
precisao = precision_score(y_test, previsoes, zero_division=0)
recall = recall_score(y_test, previsoes, zero_division=0)
f1 = f1_score(y_test, previsoes, zero_division=0)

# Criação das quatro abas
aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Dados dos sensores",
    "🤖 Modelo de IA",
    "🆕 Nova leitura",
    "⚠️ Limitações e cuidados"
])

# ABA 1 - DADOS DOS SENSORES
with aba1:
    st.header("Dados dos sensores")

    st.write(
        "Visualização das leituras registradas no arquivo sensores.csv."
    )

    total = len(df)
    normais = int((df["falha"] == 0).sum())
    falhas = int((df["falha"] == 1).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total de leituras", total)
    col2.metric("Funcionamento normal", normais)
    col3.metric("Leituras com falha", falhas)

    st.subheader("Tabela de leituras")
    st.dataframe(df, use_container_width=True)

    st.subheader("Gráfico dos sensores")

    sensor = st.selectbox(
        "Escolha o sensor para visualizar:",
        ["temperatura", "umidade", "vibracao"]
    )

    grafico = px.line(
        df,
        x="timestamp",
        y=sensor,
        color="falha",
        title=f"Evolução de {sensor} ao longo do tempo",
        labels={
            "timestamp": "Data e hora",
            sensor: sensor.capitalize(),
            "falha": "Falha (0 = normal, 1 = falha)"
        },
        color_discrete_map={
            0: "green",
            1: "red"
        }
    )

    st.plotly_chart(grafico, use_container_width=True)

    st.subheader("Distribuição das leituras")

    distribuicao = pd.DataFrame({
        "Situação": ["Normal", "Falha"],
        "Quantidade": [normais, falhas]
    })

    grafico_pizza = px.pie(
        distribuicao,
        names="Situação",
        values="Quantidade",
        title="Leituras normais e com falha",
        color="Situação",
        color_discrete_map={
            "Normal": "green",
            "Falha": "red"
        }
    )

    st.plotly_chart(grafico_pizza, use_container_width=True)


# ABA 2 - MODELO DE IA
with aba2:
    st.header("Modelo de Inteligência Artificial")

    st.write(
        "Foi utilizado o algoritmo Random Forest para identificar "
        "padrões associados a falhas nos equipamentos."
    )

    st.write(
        "As variáveis utilizadas foram temperatura, umidade e vibração. "
        "A variável que o modelo tenta prever é falha."
    )

    st.subheader("Configuração do treinamento")

    st.write(f"Total de registros: {len(df)}")
    st.write(f"Registros para treinamento: {len(X_train)}")
    st.write(f"Registros para teste: {len(X_test)}")
    st.write("Divisão: 80% treinamento e 20% teste.")

    st.subheader("Resultados do modelo")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Acurácia", f"{acuracia:.1%}")
    m2.metric("Precisão", f"{precisao:.1%}")
    m3.metric("Recall", f"{recall:.1%}")
    m4.metric("F1-score", f"{f1:.1%}")

    st.subheader("O que significam as métricas?")

    st.markdown("""
    - **Acurácia:** mostra a porcentagem de previsões corretas
      em relação ao total de previsões.

    - **Precisão:** mostra, entre as previsões de falha,
      quantas realmente eram falhas.

    - **Recall:** mostra quantas das falhas reais o modelo
      conseguiu identificar.

    - **F1-score:** combina precisão e recall em uma única
      medida, equilibrando as duas.
    """)

    st.info(
        "Em manutenção de equipamentos, o recall é especialmente "
        "importante, pois ajuda a avaliar quantas falhas reais "
        "o modelo conseguiu detectar."
    )

    st.subheader("Matriz de confusão")

    matriz = confusion_matrix(y_test, previsoes, labels=[0, 1])

    matriz_df = pd.DataFrame(
        matriz,
        index=["Real normal", "Real falha"],
        columns=["Previsto normal", "Previsto falha"]
    )

    st.dataframe(matriz_df, use_container_width=True)

    st.subheader("Importância das variáveis")

    importancia = pd.DataFrame({
        "Sensor": X.columns,
        "Importância": modelo.feature_importances_
    }).sort_values("Importância", ascending=False)

    grafico_importancia = px.bar(
        importancia,
        x="Sensor",
        y="Importância",
        title="Importância de cada sensor para o modelo"
    )

    st.plotly_chart(grafico_importancia, use_container_width=True)


# ABA 3 - NOVA LEITURA
with aba3:
    st.header("Previsão de uma nova leitura")

    st.write(
        "Informe os valores dos sensores para verificar a previsão "
        "do modelo de inteligência artificial."
    )

    with st.form("formulario_sensores"):

        temperatura = st.number_input(
            "Temperatura (°C)",
            min_value=-50.0,
            max_value=200.0,
            value=30.0,
            step=0.5
        )

        umidade = st.number_input(
            "Umidade (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.5
        )

        vibracao = st.number_input(
            "Vibração",
            min_value=0.0,
            max_value=1000.0,
            value=2.0,
            step=0.1
        )

        enviar = st.form_submit_button("Prever funcionamento")

    if enviar:

        nova_leitura = pd.DataFrame({
            "temperatura": [temperatura],
            "umidade": [umidade],
            "vibracao": [vibracao]
        })

        previsao = int(modelo.predict(nova_leitura)[0])

        probabilidades = modelo.predict_proba(nova_leitura)[0]
        indice_falha = list(modelo.classes_).index(1)
        probabilidade_falha = probabilidades[indice_falha]

        st.subheader("Resultado da previsão")

        st.metric(
            "Probabilidade estimada de falha",
            f"{probabilidade_falha:.1%}"
        )

        if previsao == 1:
            st.error("⚠️ O modelo identificou possibilidade de falha.")
        else:
            st.success("✅ O modelo classificou a leitura como normal.")

        st.subheader("Ação recomendada")

        if probabilidade_falha >= 0.70:
            st.error(
                "PARAR E AVALIAR: recomenda-se uma avaliação "
                "imediata por um profissional responsável, seguindo "
                "os procedimentos de segurança da empresa."
            )

        elif probabilidade_falha >= 0.40:
            st.warning(
                "INSPECIONAR: recomenda-se verificar o equipamento "
                "e acompanhar os sensores."
            )

        else:
            st.info(
                "MONITORAR: continuar acompanhando as leituras "
                "e realizar as inspeções preventivas programadas."
            )

        st.caption(
            "A recomendação é uma orientação experimental baseada no "
            "modelo, não substitui inspeção técnica ou sistemas "
            "de segurança do equipamento."
        )


# ABA 4 - LIMITAÇÕES E CUIDADOS

# ABA 4 - LIMITAÇÕES E CUIDADOS
with aba4:
    st.header("Limitações e cuidados")

    st.write(
        "Apesar de ajudar na identificação de possíveis falhas, "
        "o modelo possui limitações que precisam ser consideradas."
    )

    st.subheader("1. Base de dados simulada")

    st.write(
        "As 800 leituras são simuladas e podem não representar "
        "completamente as condições reais de uma máquina. "
        "O desempenho em equipamentos reais pode ser diferente."
    )

    st.success(
        "Como reduzir esse risco: testar o modelo com dados reais "
        "de equipamentos e atualizar a base periodicamente."
    )

    st.subheader("2. Possibilidade de previsões incorretas")

    st.write(
        "O modelo pode indicar funcionamento normal quando existe "
        "uma falha ou indicar falha quando o equipamento está normal."
    )

    st.success(
        "Como reduzir esse risco: acompanhar as métricas, "
        "principalmente o recall, analisar a matriz de confusão "
        "e realizar inspeções técnicas."
    )

    st.subheader("3. Poucas variáveis analisadas")

    st.write(
        "O modelo utiliza somente temperatura, umidade e vibração. "
        "Outros fatores, como pressão, ruído, desgaste e velocidade, "
        "também podem influenciar o funcionamento do equipamento."
    )

    st.success(
        "Como reduzir esse risco: incluir outros sensores e variáveis "
        "relevantes para melhorar a análise do equipamento."
    )

    st.subheader("4. Mudanças nas condições de operação")

    st.write(
        "Se o equipamento passar a operar em condições diferentes "
        "das utilizadas no treinamento, o modelo poderá perder "
        "parte da sua capacidade de previsão."
    )

    st.success(
        "Como reduzir esse risco: acompanhar o desempenho do modelo "
        "e realizar novos treinamentos com dados atualizados."
    )

    st.subheader("5. Cuidados antes de tomar decisões")

    st.write(
        "As previsões devem ser utilizadas como apoio à manutenção, "
        "e não como única fonte de decisão."
    )

    st.warning(
        "Nunca dependa exclusivamente da previsão da inteligência "
        "artificial para manter um equipamento em funcionamento "
        "quando houver risco à segurança. A avaliação de um "
        "profissional responsável é indispensável."
    )

    st.success(
        "Conclusão: a inteligência artificial pode ajudar na "
        "manutenção preventiva, desde que suas limitações sejam "
        "consideradas e exista acompanhamento profissional."
    )


st.divider()

st.caption(
    "Projeto acadêmico - Monitoramento de sensores e previsão de falhas."
)
