"""
Notebook 2: Modelagem Preditiva - Classificação de Procedência
Framework de Suporte à Decisão - Saneamento Básico
Mestrado Profissionalizante - CASAN

Observação metodológica:
    Este arquivo usa exclusivamente dados sintéticos. Os resultados são
    demonstrativos e não representam desempenho em processos reais.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = Path("resultados_classificacao")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 6)
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 50)


def gerar_dados_sinteticos(n_processos=10_000, random_state=RANDOM_STATE):
    """Gera a base artificial utilizada apenas para demonstração."""
    rng = np.random.default_rng(random_state)
    assuntos = [
        "AGUA", "ESGOTO", "DRENAGEM", "RESIDUOS", "TARIFA",
        "QUALIDADE", "INTERRUPCAO", "DANO_AMBIENTAL",
    ]
    tipos_demanda = [
        "INDIVIDUAL", "COLETIVA", "ACAO_CIVIL_PUBLICA", "MANDADO_SEGURANCA",
    ]
    resultados = [
        "PROCEDENTE", "IMPROCEDENTE", "PARCIALMENTE_PROCEDENTE", "EXTINTO",
    ]

    data = {
        "id_processo": np.arange(1, n_processos + 1),
        "classe": rng.choice(
            [
                "Procedimento Comum", "Procedimento Sumário",
                "Mandado de Segurança", "Ação Civil Pública", "Recurso",
            ], n_processos,
        ),
        "assunto_principal": rng.choice(assuntos, n_processos),
        "valor_causa": rng.lognormal(mean=8, sigma=1.5, size=n_processos),
        "data_distribuicao": pd.date_range(
            start="2015-01-01", end="2024-12-31", periods=n_processos,
        ),
        "tribunal": rng.choice(["TJSC", "STJ", "STF", "TRF4"], n_processos),
        "comarca": rng.choice(
            ["Florianópolis", "Joinville", "Blumenau", "São José",
             "Criciúma", "Lages", "Chapecó", "Itajaí"], n_processos,
        ),
        "tipo_parte_autora": rng.choice(
            ["CONSUMIDOR", "EMPRESA", "ORGAO_PUBLICO", "MP"],
            n_processos, p=[0.6, 0.2, 0.1, 0.1],
        ),
        "tipo_parte_reu": rng.choice(
            ["CONCESSIONARIA", "MUNICIPIO", "ESTADO"],
            n_processos, p=[0.7, 0.2, 0.1],
        ),
        "resultado_sentenca": rng.choice(
            resultados, n_processos, p=[0.35, 0.25, 0.25, 0.15],
        ),
        "tempo_total_dias": rng.exponential(scale=365, size=n_processos).astype(int),
        "numero_audiencias": rng.poisson(lam=2, size=n_processos),
        "numero_pericias": rng.poisson(lam=0.5, size=n_processos),
        "numero_recursos": rng.poisson(lam=1, size=n_processos),
        "tipo_demanda": rng.choice(
            tipos_demanda, n_processos, p=[0.5, 0.3, 0.1, 0.1],
        ),
        "assunto_saneamento": rng.choice(assuntos, n_processos),
        "regiao_geografica": rng.choice(
            ["GRANDE_FLORIANOPOLIS", "NORTE", "SUL", "OESTE", "VALE_ITAJAI"],
            n_processos,
        ),
    }
    df = pd.DataFrame(data)
    df["ano_distribuicao"] = df["data_distribuicao"].dt.year
    df["mes_distribuicao"] = df["data_distribuicao"].dt.month
    df["target_procedencia"] = df["resultado_sentenca"].map(
        {"PROCEDENTE": 1, "IMPROCEDENTE": 0,
         "PARCIALMENTE_PROCEDENTE": 1, "EXTINTO": np.nan}
    )
    return df.dropna(subset=["target_procedencia"]).copy()


def construir_modelos():
    """Cria os quatro algoritmos citados no texto acadêmico."""
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=1_000, random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, verbosity=-1,
        ),
    }


def executar():
    print("=" * 60)
    print("MODELAGEM PREDITIVA - CLASSIFICAÇÃO DE PROCEDÊNCIA")
    print("Framework de Jurimetria para Saneamento Básico")
    print("=" * 60)

    print("\n[1] Gerando dados sintéticos...")
    df = gerar_dados_sinteticos()
    print(f"Dataset gerado: {df.shape[0]} processos, {df.shape[1]} colunas")

    # resultado_sentenca, tempo_total_dias e demais campos posteriores ao
    # desfecho não entram como preditores: isso evita vazamento de informação.
    numeric_features = [
        "valor_causa", "numero_audiencias", "numero_pericias",
        "numero_recursos", "ano_distribuicao", "mes_distribuicao",
    ]
    categorical_features = [
        "classe", "assunto_principal", "tribunal", "comarca",
        "tipo_parte_autora", "tipo_parte_reu", "tipo_demanda",
        "assunto_saneamento", "regiao_geografica",
    ]
    feature_columns = numeric_features + categorical_features
    X = df[feature_columns]
    y = df["target_procedencia"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados = []
    curvas_roc = {}
    modelos_treinados = {}

    print("\n[2] Treinando e avaliando os quatro modelos...")
    for nome, modelo in construir_modelos().items():
        pipeline = Pipeline([
            ("preprocessamento", preprocessor),
            ("modelo", modelo),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        cv_auc = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        curvas_roc[nome] = (fpr, tpr, roc_auc_score(y_test, y_prob))
        modelos_treinados[nome] = pipeline
        resultados.append({
            "Modelo": nome,
            "Acurácia": accuracy_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_test, y_prob),
            "AUC-ROC CV média": cv_auc.mean(),
            "AUC-ROC CV desvio": cv_auc.std(),
        })

    resultados_df = pd.DataFrame(resultados).sort_values("AUC-ROC", ascending=False)
    resultados_df.to_csv(OUTPUT_DIR / "metricas_modelos.csv", index=False)
    print("\n[3] Métricas no conjunto de teste:")
    print(resultados_df.round(4).to_string(index=False))
    melhor_modelo = resultados_df.iloc[0]["Modelo"]
    print(f"\nModelo com maior AUC no cenário sintético: {melhor_modelo}")
    print("Este resultado não permite inferência sobre processos reais.")

    plt.figure(figsize=(12, 7))
    for nome, (fpr, tpr, auc) in curvas_roc.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{nome} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Referência aleatória")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title("Curvas ROC — classificação experimental")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curvas_roc.png", dpi=150)
    plt.close()

    melhor_pipeline = modelos_treinados[melhor_modelo]
    y_pred_melhor = melhor_pipeline.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred_melhor)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Não procedente", "Procedente"],
        yticklabels=["Não procedente", "Procedente"],
    )
    plt.xlabel("Predito")
    plt.ylabel("Observado")
    plt.title(f"Matriz de confusão — {melhor_modelo}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matriz_confusao_melhor_modelo.png", dpi=150)
    plt.close()

    print(f"\nResultados salvos em: {OUTPUT_DIR.resolve()}")
    return resultados_df, modelos_treinados


if __name__ == "__main__":
    executar()
