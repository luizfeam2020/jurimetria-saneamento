# -*- coding: utf-8 -*-
"""Dashboard demonstrativo de jurimetria aplicada ao saneamento básico.

Dados sintéticos; recomendações orientativas; revisão humana obrigatória.
Execução: streamlit run notebooks\\05_dashboard_streamlit.py
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
NUMERIC_FEATURES = ["valor_causa", "numero_audiencias", "numero_pericias", "numero_recursos", "ano_distribuicao", "mes_distribuicao"]
CATEGORICAL_FEATURES = ["classe", "assunto_principal", "tribunal", "comarca", "tipo_parte_autora", "tipo_parte_reu", "tipo_demanda", "assunto_saneamento", "regiao_geografica"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def carregar_motor() -> Any:
    caminho = Path(__file__).with_name("06_motor_regras_mvp.py")
    spec = importlib.util.spec_from_file_location("motor_regras_mvp", caminho)
    if spec is None or spec.loader is None:
        raise ImportError(f"Motor não encontrado: {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


motor = carregar_motor()


@st.cache_data
def gerar_dados_sinteticos(n_processos: int = 3000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
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
        "numero_audiencias": rng.poisson(2, n_processos), "numero_pericias": rng.poisson(.5, n_processos), "numero_recursos": rng.poisson(1, n_processos),
    })
    df["ano_distribuicao"] = df["data_distribuicao"].dt.year; df["mes_distribuicao"] = df["data_distribuicao"].dt.month
    sinal = (.8 * (df.assunto_principal == "DANO_AMBIENTAL") + .6 * (df.tipo_parte_autora == "MP") + .4 * (df.tipo_demanda == "ACAO_CIVIL_PUBLICA") + .25 * (df.valor_causa > df.valor_causa.median()) + rng.normal(0, .7, n_processos))
    df["target_procedencia"] = (sinal > .65).astype(int); df["resultado_sentenca"] = np.where(df.target_procedencia.eq(1), "PROCEDENTE", "IMPROCEDENTE")
    return df


@st.cache_resource
def treinar_modelo(df: pd.DataFrame) -> Pipeline:
    prep = ColumnTransformer([("num", "passthrough", NUMERIC_FEATURES), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    modelo = Pipeline([("preprocessamento", prep), ("modelo", RandomForestClassifier(n_estimators=180, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))])
    modelo.fit(df[FEATURE_COLUMNS], df["target_procedencia"])
    return modelo


def formatar_reais(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros da análise")
    tribunais = st.sidebar.multiselect("Tribunal", sorted(df.tribunal.unique()), default=sorted(df.tribunal.unique()))
    assuntos = st.sidebar.multiselect("Assunto", sorted(df.assunto_principal.unique()), default=sorted(df.assunto_principal.unique()))
    regioes = st.sidebar.multiselect("Região", sorted(df.regiao_geografica.unique()), default=sorted(df.regiao_geografica.unique()))
    anos = st.sidebar.slider("Ano de distribuição", int(df.ano_distribuicao.min()), int(df.ano_distribuicao.max()), (int(df.ano_distribuicao.min()), int(df.ano_distribuicao.max())))
    resultado = df[df.tribunal.isin(tribunais) & df.assunto_principal.isin(assuntos) & df.regiao_geografica.isin(regioes) & df.ano_distribuicao.between(*anos)].copy()
    st.sidebar.caption(f"{len(resultado):,} processos selecionados")
    return resultado


def avaliar_modelo(df: pd.DataFrame, pipeline: Pipeline) -> dict[str, float]:
    pred = pipeline.predict(df[FEATURE_COLUMNS]); prob = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]; y = df.target_procedencia
    return {"Acurácia": accuracy_score(y, pred), "Precisão": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0), "F1-score": f1_score(y, pred, zero_division=0), "AUC-ROC": roc_auc_score(y, prob)}


def preparar_caso(linha: pd.Series, probabilidade: float) -> dict[str, Any]:
    assunto = str(linha.assunto_principal)
    return {"caso_id": str(linha.id_processo), "valor_causa": float(linha.valor_causa), "impacto_financeiro": "alto" if linha.valor_causa >= 150000 else "medio" if linha.valor_causa >= 50000 else "baixo", "tipologia": assunto, "tipologia_sensivel": any(t in assunto.lower() for t in ("interrup", "tarifa", "agua", "qualidade")), "fase": "recurso" if linha.numero_recursos >= 2 else "instrução", "fase_avancada": bool(linha.numero_recursos >= 2), "tempo_tramitacao_dias": max(0, (date.today() - linha.data_distribuicao.date()).days), "impacto_reputacional": bool(linha.tipo_demanda in ("COLETIVA", "ACAO_CIVIL_PUBLICA")), "probabilidade_modelo": probabilidade}


def renderizar_hitl(resultado: Any) -> None:
    st.subheader("Revisão humana (HITL)")
    st.info("A decisão é obrigatoriamente humana. O índice é preliminar e não substitui análise jurídica.")
    decisao = st.selectbox("Decisão do responsável", ["Aceitar", "Aceitar com ajuste", "Rejeitar"], key=f"decisao_{resultado.caso_id}")
    usuario = st.text_input("Usuário responsável", key=f"usuario_{resultado.caso_id}")
    justificativa = st.text_area("Justificativa obrigatória", key=f"justificativa_{resultado.caso_id}")
    if st.button("Registrar decisão", key=f"registrar_{resultado.caso_id}"):
        if not usuario.strip(): st.error("Informe o usuário responsável.")
        elif not justificativa.strip(): st.error("A justificativa é obrigatória.")
        else:
            registro = motor.registrar_decisao(resultado, decisao, justificativa, usuario, Path("logs_hitl.csv"))
            st.success(f"Decisão registrada para {registro['caso_id']} em {registro['data_hora_utc']}.")


def main() -> None:
    st.set_page_config(page_title="Jurimetria — Saneamento", page_icon="⚖️", layout="wide")
    st.title("Framework de Jurimetria para Saneamento Básico")
    st.caption("Dashboard demonstrativo com dados sintéticos, modelo preditivo e motor de regras MVP.")
    st.sidebar.warning("Demonstração acadêmica. Revisão humana obrigatória.")
    df = gerar_dados_sinteticos(); filtrado = aplicar_filtros(df); pipeline = treinar_modelo(df)
    abas = st.tabs(["Visão geral", "Perfil dos processos", "Modelagem", "Explicabilidade", "Predição e HITL"])
    with abas[0]:
        c1, c2, c3, c4 = st.columns(4); c1.metric("Processos", f"{len(filtrado):,}"); c2.metric("Valor médio", formatar_reais(filtrado.valor_causa.mean())); c3.metric("Procedência", f"{filtrado.target_procedencia.mean():.1%}"); c4.metric("Tempo mediano", f"{filtrado.tempo_total_dias.median():.0f} dias")
        st.plotly_chart(px.histogram(filtrado, x="resultado_sentenca", color="resultado_sentenca", title="Distribuição dos resultados"), use_container_width=True)
    with abas[1]:
        st.dataframe(filtrado.head(100), use_container_width=True, hide_index=True)
        contagem = filtrado.tribunal.value_counts().rename_axis("Tribunal").reset_index(name="Processos")
        st.plotly_chart(px.bar(contagem, x="Tribunal", y="Processos", title="Processos por tribunal"), use_container_width=True)
    with abas[2]:
        metricas = avaliar_modelo(df, pipeline); st.dataframe(pd.DataFrame([metricas]).T.rename(columns={0: "Valor"}).style.format("{:.3f}"), use_container_width=True)
        st.warning("As métricas usam dados sintéticos e não representam desempenho em processos reais.")
    with abas[3]:
        amostra = filtrado if len(filtrado) > 20 else df; imp = permutation_importance(pipeline, amostra[FEATURE_COLUMNS], amostra.target_procedencia, n_repeats=3, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
        tabela = pd.DataFrame({"Variável": FEATURE_COLUMNS, "Importância": imp.importances_mean}).sort_values("Importância", ascending=False).head(12)
        st.plotly_chart(px.bar(tabela.sort_values("Importância"), x="Importância", y="Variável", orientation="h", title="Importância por permutação"), use_container_width=True)
        st.info("A associação do modelo não é causalidade nem justificativa jurídica.")
    with abas[4]:
        st.subheader("Avaliação individual")
        if filtrado.empty:
            st.warning("Nenhum processo atende aos filtros atuais."); return
        indice = st.selectbox("Selecione o caso", filtrado.index, format_func=lambda i: str(filtrado.loc[i, "id_processo"]))
        linha = filtrado.loc[indice]; prob = float(pipeline.predict_proba(pd.DataFrame([linha])[FEATURE_COLUMNS])[0, 1]); resultado = motor.avaliar_caso(preparar_caso(linha, prob))
        a, b, c = st.columns(3); a.metric("Índice de atenção", f"{resultado.indice_atencao}/100"); b.metric("Faixa", resultado.faixa); c.metric("Probabilidade do modelo", f"{prob:.1%}")
        st.write("**Recomendação:**", resultado.recomendacao); st.write("**Qualidade dos dados:**", resultado.qualidade_dados)
        st.dataframe(pd.DataFrame(resultado.fatores), use_container_width=True, hide_index=True); renderizar_hitl(resultado)
        log = Path("logs_hitl.csv")
        if log.exists():
            st.subheader("Trilha de auditoria"); st.dataframe(pd.read_csv(log), use_container_width=True, hide_index=True)
    st.caption("Uso responsável: resultados sintéticos, finalidade acadêmica e apoio à análise — nunca decisão automatizada.")


if __name__ == "__main__":
    main()
