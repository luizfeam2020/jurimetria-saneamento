"""
Dashboard Streamlit — Jurimetria aplicada ao saneamento básico

Aplicação demonstrativa, autocontida e baseada exclusivamente em dados sintéticos.
Os resultados não representam processos reais, não constituem decisão jurídica e
não substituem análise profissional, contraditório ou revisão humana.

Execução no Windows, a partir da raiz do projeto:
    streamlit run notebooks\\05_dashboard_streamlit.py

Caso este arquivo esteja na raiz:
    streamlit run 05_dashboard_streamlit.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import shap
except ImportError:  # SHAP é opcional para manter o dashboard executável.
    shap = None

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "valor_causa", "numero_audiencias", "numero_pericias",
    "numero_recursos", "ano_distribuicao", "mes_distribuicao",
]
CATEGORICAL_FEATURES = [
    "classe", "assunto_principal", "tribunal", "comarca",
    "tipo_parte_autora", "tipo_parte_reu", "tipo_demanda",
    "assunto_saneamento", "regiao_geografica",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_data

def gerar_dados_sinteticos(n_processos: int = 3000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Gera uma base artificial reproduzível para demonstração do fluxo."""
    rng = np.random.default_rng(random_state)
    assuntos = ["AGUA", "ESGOTO", "DRENAGEM", "RESIDUOS", "TARIFA", "QUALIDADE", "INTERRUPCAO", "DANO_AMBIENTAL"]
    datas = pd.date_range("2015-01-01", "2024-12-31", periods=n_processos)
    df = pd.DataFrame({
        "id_processo": np.arange(1, n_processos + 1),
        "classe": rng.choice(["Procedimento Comum", "Procedimento Sumário", "Mandado de Segurança", "Ação Civil Pública", "Recurso"], n_processos),
        "assunto_principal": rng.choice(assuntos, n_processos),
        "valor_causa": rng.lognormal(8, 1.5, n_processos).round(2),
        "data_distribuicao": datas,
        "tribunal": rng.choice(["TJSC", "STJ", "STF", "TRF4"], n_processos),
        "comarca": rng.choice(["Florianópolis", "Joinville", "Blumenau", "São José", "Criciúma", "Lages", "Chapecó", "Itajaí"], n_processos),
        "tipo_parte_autora": rng.choice(["CONSUMIDOR", "EMPRESA", "ORGAO_PUBLICO", "MP"], n_processos, p=[.6, .2, .1, .1]),
        "tipo_parte_reu": rng.choice(["CONCESSIONARIA", "MUNICIPIO", "ESTADO"], n_processos, p=[.7, .2, .1]),
        "tipo_demanda": rng.choice(["INDIVIDUAL", "COLETIVA", "ACAO_CIVIL_PUBLICA", "MANDADO_SEGURANCA"], n_processos, p=[.5, .3, .1, .1]),
        "assunto_saneamento": rng.choice(assuntos, n_processos),
        "regiao_geografica": rng.choice(["GRANDE_FLORIANOPOLIS", "NORTE", "SUL", "OESTE", "VALE_ITAJAI"], n_processos),
        "tempo_total_dias": rng.exponential(365, n_processos).astype(int),
        "numero_audiencias": rng.poisson(2, n_processos),
        "numero_pericias": rng.poisson(.5, n_processos),
        "numero_recursos": rng.poisson(1, n_processos),
    })
    df["ano_distribuicao"] = df["data_distribuicao"].dt.year
    df["mes_distribuicao"] = df["data_distribuicao"].dt.month
    sinal = (.8 * (df["assunto_principal"] == "DANO_AMBIENTAL") + .6 * (df["tipo_parte_autora"] == "MP") + .4 * (df["tipo_demanda"] == "ACAO_CIVIL_PUBLICA") + .25 * (df["valor_causa"] > df["valor_causa"].median()) + rng.normal(0, .7, n_processos))
    df["target_procedencia"] = (sinal > .65).astype(int)
    df["resultado_sentenca"] = np.where(df["target_procedencia"].eq(1), "PROCEDENTE", "IMPROCEDENTE")
    return df


@st.cache_resource

def treinar_modelo(df: pd.DataFrame) -> Pipeline:
    """Treina um modelo demonstrativo sem usar o resultado como preditor."""
    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessamento", preprocessor),
        ("modelo", RandomForestClassifier(n_estimators=180, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")),
    ])
    pipeline.fit(df[FEATURE_COLUMNS], df["target_procedencia"])
    return pipeline


def avaliar_modelo(df: pd.DataFrame, pipeline: Pipeline) -> dict[str, float]:
    pred = pipeline.predict(df[FEATURE_COLUMNS])
    prob = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return {
        "Acurácia": accuracy_score(df["target_procedencia"], pred),
        "Precisão": precision_score(df["target_procedencia"], pred, zero_division=0),
        "Recall": recall_score(df["target_procedencia"], pred, zero_division=0),
        "F1-score": f1_score(df["target_procedencia"], pred, zero_division=0),
        "AUC-ROC": roc_auc_score(df["target_procedencia"], prob),
    }


def formatar_reais(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros da análise")
    tribunais = st.sidebar.multiselect("Tribunal", sorted(df["tribunal"].unique()), default=sorted(df["tribunal"].unique()))
    assuntos = st.sidebar.multiselect("Assunto", sorted(df["assunto_principal"].unique()), default=sorted(df["assunto_principal"].unique()))
    regioes = st.sidebar.multiselect("Região", sorted(df["regiao_geografica"].unique()), default=sorted(df["regiao_geografica"].unique()))
    anos = st.sidebar.slider("Ano de distribuição", int(df["ano_distribuicao"].min()), int(df["ano_distribuicao"].max()), (int(df["ano_distribuicao"].min()), int(df["ano_distribuicao"].max())))
    filtrado = df[df["tribunal"].isin(tribunais) & df["assunto_principal"].isin(assuntos) & df["regiao_geografica"].isin(regioes) & df["ano_distribuicao"].between(*anos)].copy()
    st.sidebar.caption(f"{len(filtrado):,} processos selecionados")
    return filtrado


def aba_visao_geral(df: pd.DataFrame) -> None:
    st.subheader("Visão geral da base")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processos", f"{len(df):,}")
    c2.metric("Valor médio da causa", formatar_reais(df["valor_causa"].mean()))
    c3.metric("Taxa sintética de procedência", f"{100 * df['target_procedencia'].mean():.1f}%")
    c4.metric("Tempo mediano", f"{df['tempo_total_dias'].median():.0f} dias")
    col1, col2 = st.columns(2)
    with col1:
        contagem = df["assunto_principal"].value_counts().rename_axis("Assunto").reset_index(name="Processos")
        st.plotly_chart(px.bar(contagem, x="Processos", y="Assunto", orientation="h", title="Processos por assunto"), use_container_width=True)
    with col2:
        serie = df.groupby("ano_distribuicao", as_index=False)["id_processo"].count().rename(columns={"id_processo": "Processos"})
        st.plotly_chart(px.line(serie, x="ano_distribuicao", y="Processos", markers=True, title="Distribuição anual"), use_container_width=True)


def aba_perfil(df: pd.DataFrame) -> None:
    st.subheader("Perfil dos processos")
    col1, col2 = st.columns(2)
    with col1:
        tabela = pd.crosstab(df["tribunal"], df["target_procedencia"], normalize="index").rename(columns={0: "Não procedente", 1: "Procedente"}).reset_index()
        st.plotly_chart(px.bar(tabela, x="tribunal", y=["Não procedente", "Procedente"], barmode="stack", title="Resultado sintético por tribunal"), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(df, x="target_procedencia", y="valor_causa", color="target_procedencia", title="Distribuição do valor da causa"), use_container_width=True)
    st.dataframe(df[["tribunal", "assunto_principal", "regiao_geografica", "target_procedencia"]].head(100), use_container_width=True, hide_index=True)


def aba_modelagem(df: pd.DataFrame, pipeline: Pipeline) -> None:
    st.subheader("Modelagem preditiva")
    st.info("A previsão é experimental e foi treinada com dados sintéticos. Não use o resultado como decisão jurídica.")
    metricas = avaliar_modelo(df, pipeline)
    cols = st.columns(len(metricas))
    for col, (nome, valor) in zip(cols, metricas.items()):
        col.metric(nome, f"{valor:.3f}")
    prob = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    hist = pd.DataFrame({"Probabilidade sintética": prob})
    st.plotly_chart(px.histogram(hist, x="Probabilidade sintética", nbins=25, title="Distribuição das probabilidades previstas"), use_container_width=True)
    st.caption("As métricas são calculadas sobre a mesma base demonstrativa usada no treinamento e não representam validação de desempenho real.")


def aba_explicabilidade(df: pd.DataFrame, pipeline: Pipeline) -> None:
    st.subheader("Explicabilidade")
    st.info("As explicações indicam associações do modelo; não são justificativas jurídicas nem prova de causalidade.")
    importances = permutation_importance(pipeline, df[FEATURE_COLUMNS], df["target_procedencia"], n_repeats=3, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    imp = pd.DataFrame({"Variável": FEATURE_COLUMNS, "Importância": importances.importances_mean}).sort_values("Importância", ascending=False).head(12)
    st.plotly_chart(px.bar(imp.sort_values("Importância"), x="Importância", y="Variável", orientation="h", title="Importância aproximada por permutação"), use_container_width=True)
    if shap is None:
        st.warning("SHAP não está instalado. A explicabilidade por permutação permanece disponível; para SHAP, instale a dependência com: python -m pip install shap")
    else:
        st.success("A biblioteca SHAP está disponível. Para a execução completa das explicações SHAP detalhadas, utilize também o script 04_explicabilidade_modelos.py.")


def aba_predicao(df: pd.DataFrame, pipeline: Pipeline) -> None:
    st.subheader("Predição experimental de um novo caso")
    st.warning("A saída é apenas demonstrativa, baseada em dados sintéticos, e exige revisão humana qualificada.")
    with st.form("form_predicao"):
        col1, col2 = st.columns(2)
        with col1:
            classe = st.selectbox("Classe", sorted(df["classe"].unique()))
            assunto = st.selectbox("Assunto principal", sorted(df["assunto_principal"].unique()))
            tribunal = st.selectbox("Tribunal", sorted(df["tribunal"].unique()))
            comarca = st.selectbox("Comarca", sorted(df["comarca"].unique()))
            valor = st.number_input("Valor da causa (R$)", min_value=0.0, value=float(df["valor_causa"].median()), step=100.0)
            ano = st.number_input("Ano de distribuição", min_value=2015, max_value=2030, value=2026, step=1)
        with col2:
            autora = st.selectbox("Tipo de parte autora", sorted(df["tipo_parte_autora"].unique()))
            reu = st.selectbox("Tipo de parte ré", sorted(df["tipo_parte_reu"].unique()))
            demanda = st.selectbox("Tipo de demanda", sorted(df["tipo_demanda"].unique()))
            saneamento = st.selectbox("Assunto de saneamento", sorted(df["assunto_saneamento"].unique()))
            regiao = st.selectbox("Região geográfica", sorted(df["regiao_geografica"].unique()))
            audiencias = st.number_input("Número de audiências", min_value=0, value=2, step=1)
            pericias = st.number_input("Número de perícias", min_value=0, value=0, step=1)
            recursos = st.number_input("Número de recursos", min_value=0, value=1, step=1)
        enviar = st.form_submit_button("Calcular previsão")
    if enviar:
        novo = pd.DataFrame([{"valor_causa": valor, "numero_audiencias": audiencias, "numero_pericias": pericias, "numero_recursos": recursos, "ano_distribuicao": ano, "mes_distribuicao": date.today().month, "classe": classe, "assunto_principal": assunto, "tribunal": tribunal, "comarca": comarca, "tipo_parte_autora": autora, "tipo_parte_reu": reu, "tipo_demanda": demanda, "assunto_saneamento": saneamento, "regiao_geografica": regiao}])
        prob = float(pipeline.predict_proba(novo[FEATURE_COLUMNS])[0, 1])
        st.metric("Probabilidade sintética estimada de procedência", f"{100 * prob:.1f}%")
        st.progress(prob)
        st.caption("Esta probabilidade não é uma conclusão sobre o caso e não deve orientar decisão automática.")


def main() -> None:
    st.set_page_config(page_title="Jurimetria — Saneamento", page_icon="⚖️", layout="wide")
    st.title("Framework de Jurimetria para Saneamento Básico")
    st.caption("Dashboard demonstrativo com dados sintéticos — Avanço 2")
    st.sidebar.markdown("### Status metodológico")
    st.sidebar.warning("Demonstração acadêmica. Revisão humana obrigatória.")
    df = gerar_dados_sinteticos()
    filtrado = aplicar_filtros(df)
    pipeline = treinar_modelo(df)
    abas = st.tabs(["Visão geral", "Perfil dos processos", "Modelagem", "Explicabilidade", "Predição"])
    with abas[0]:
        aba_visao_geral(filtrado)
    with abas[1]:
        aba_perfil(filtrado)
    with abas[2]:
        aba_modelagem(filtrado if len(filtrado) > 20 else df, pipeline)
    with abas[3]:
        aba_explicabilidade(filtrado if len(filtrado) > 20 else df, pipeline)
    with abas[4]:
        aba_predicao(df, pipeline)
    st.divider()
    st.caption("Uso responsável: resultados sintéticos, finalidade acadêmica e apoio à análise — nunca decisão automatizada.")


if __name__ == "__main__":
    main()
