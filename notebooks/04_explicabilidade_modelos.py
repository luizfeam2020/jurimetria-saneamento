"""
Notebook 4: Explicabilidade dos modelos preditivos
Framework de Suporte à Decisão - Saneamento Básico

Este arquivo usa exclusivamente dados sintéticos para demonstração.
As explicações SHAP e LIME não constituem decisão jurídica e não substituem
análise profissional, contraditório ou revisão humana.

Execução:
    python 04_explicabilidade_modelos.py

Saídas:
    resultados_explicabilidade/
        importancia_shap_global.csv
        importancia_shap_global.png
        shap_summary.png
        shap_dependence_valor_causa.png
        explicacao_shap_individual.csv
        explicacao_lime_individual.html (quando lime estiver instalado)
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

try:
    import shap
except ImportError:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUTPUT_DIR = Path("resultados_explicabilidade")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")


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


def gerar_dados_sinteticos(n_processos=5_000, random_state=RANDOM_STATE):
    """Gera dados artificiais compatíveis com o experimento de classificação."""
    rng = np.random.default_rng(random_state)
    assuntos = [
        "AGUA", "ESGOTO", "DRENAGEM", "RESIDUOS", "TARIFA",
        "QUALIDADE", "INTERRUPCAO", "DANO_AMBIENTAL",
    ]
    df = pd.DataFrame({
        "valor_causa": rng.lognormal(mean=8, sigma=1.5, size=n_processos),
        "numero_audiencias": rng.poisson(2, n_processos),
        "numero_pericias": rng.poisson(0.5, n_processos),
        "numero_recursos": rng.poisson(1, n_processos),
        "ano_distribuicao": rng.integers(2015, 2025, n_processos),
        "mes_distribuicao": rng.integers(1, 13, n_processos),
        "classe": rng.choice([
            "Procedimento Comum", "Procedimento Sumário",
            "Mandado de Segurança", "Ação Civil Pública", "Recurso",
        ], n_processos),
        "assunto_principal": rng.choice(assuntos, n_processos),
        "tribunal": rng.choice(["TJSC", "STJ", "STF", "TRF4"], n_processos),
        "comarca": rng.choice([
            "Florianópolis", "Joinville", "Blumenau", "São José",
            "Criciúma", "Lages", "Chapecó", "Itajaí",
        ], n_processos),
        "tipo_parte_autora": rng.choice(
            ["CONSUMIDOR", "EMPRESA", "ORGAO_PUBLICO", "MP"],
            n_processos, p=[0.6, 0.2, 0.1, 0.1],
        ),
        "tipo_parte_reu": rng.choice(
            ["CONCESSIONARIA", "MUNICIPIO", "ESTADO"],
            n_processos, p=[0.7, 0.2, 0.1],
        ),
        "tipo_demanda": rng.choice(
            ["INDIVIDUAL", "COLETIVA", "ACAO_CIVIL_PUBLICA", "MANDADO_SEGURANCA"],
            n_processos, p=[0.5, 0.3, 0.1, 0.1],
        ),
        "assunto_saneamento": rng.choice(assuntos, n_processos),
        "regiao_geografica": rng.choice([
            "GRANDE_FLORIANOPOLIS", "NORTE", "SUL", "OESTE", "VALE_ITAJAI",
        ], n_processos),
    })
    # Alvo sintético: uma relação artificial, usada somente para testar o fluxo.
    sinal = (
        0.25 * (df["assunto_principal"] == "DANO_AMBIENTAL")
        + 0.20 * (df["tipo_parte_autora"] == "MP")
        + 0.15 * (df["tipo_demanda"] == "ACAO_CIVIL_PUBLICA")
        + 0.10 * (df["valor_causa"] > df["valor_causa"].median())
        + rng.normal(0, 0.15, n_processos)
    )
    df["target_procedencia"] = (sinal > 0.20).astype(int)
    return df


def construir_modelo():
    """Monta o pipeline com codificação segura e XGBoost."""
    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    modelo = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessamento", preprocessor),
        ("modelo", modelo),
    ])


def nomes_variaveis_transformadas(pipeline):
    """Recupera nomes das colunas após o one-hot encoding."""
    preprocessor = pipeline.named_steps["preprocessamento"]
    return list(preprocessor.get_feature_names_out())


def normalizar_shap_values(shap_values):
    """Compatibiliza retornos de versões diferentes do SHAP."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    valores = np.asarray(shap_values)
    if valores.ndim == 3:
        return valores[:, :, -1]
    return valores


def executar():
    print("=" * 60)
    print("EXPLICABILIDADE DOS MODELOS — SHAP E LIME")
    print("Dados exclusivamente sintéticos")
    print("=" * 60)

    if shap is None:
        raise ImportError(
            "A biblioteca SHAP não está instalada. Execute: python -m pip install shap"
        )

    print("\n[1] Gerando dados sintéticos e treinando o modelo...")
    df = gerar_dados_sinteticos()
    X = df[FEATURE_COLUMNS]
    y = df["target_procedencia"]
    pipeline = construir_modelo()
    pipeline.fit(X, y)

    preprocessor = pipeline.named_steps["preprocessamento"]
    modelo = pipeline.named_steps["modelo"]
    X_transformado = preprocessor.transform(X)
    if hasattr(X_transformado, "toarray"):
        X_transformado = X_transformado.toarray()
    nomes = nomes_variaveis_transformadas(pipeline)

    print("[2] Calculando importância global com SHAP...")
    explainer = shap.TreeExplainer(modelo)
    valores_shap = normalizar_shap_values(explainer.shap_values(X_transformado))
    importancia = pd.DataFrame({
        "variavel": nomes,
        "importancia_media_absoluta": np.abs(valores_shap).mean(axis=0),
    }).sort_values("importancia_media_absoluta", ascending=False)
    importancia.to_csv(OUTPUT_DIR / "importancia_shap_global.csv", index=False)

    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=importancia.head(15),
        x="importancia_media_absoluta", y="variavel", color="#2f6f9f",
    )
    plt.title("Importância global — média dos valores absolutos SHAP")
    plt.xlabel("Importância média absoluta")
    plt.ylabel("Variável transformada")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importancia_shap_global.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        valores_shap, X_transformado, feature_names=nomes,
        max_display=15, show=False,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[3] Gerando explicação individual com SHAP...")
    indice = 0
    explicacao_individual = pd.DataFrame({
        "variavel": nomes,
        "valor_transformado": X_transformado[indice],
        "valor_shap": valores_shap[indice],
    })
    explicacao_individual["impacto_absoluto"] = explicacao_individual["valor_shap"].abs()
    explicacao_individual = explicacao_individual.sort_values(
        "impacto_absoluto", ascending=False,
    )
    explicacao_individual.to_csv(
        OUTPUT_DIR / "explicacao_shap_individual.csv", index=False,
    )

    if "num__valor_causa" in nomes:
        coluna = nomes.index("num__valor_causa")
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            coluna, valores_shap, X_transformado, feature_names=nomes,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "shap_dependence_valor_causa.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    print("[4] Gerando explicação local com LIME...")
    if LimeTabularExplainer is None:
        print("Aviso: LIME não está instalado; a etapa LIME foi ignorada.")
    else:
        explainer_lime = LimeTabularExplainer(
            X_transformado,
            feature_names=nomes,
            class_names=["Não procedente", "Procedente"],
            mode="classification",
            discretize_continuous=True,
            random_state=RANDOM_STATE,
        )
        explicacao_lime = explainer_lime.explain_instance(
            X_transformado[indice],
            modelo.predict_proba,
            num_features=15,
        )
        explicacao_lime.save_to_file(
            str(OUTPUT_DIR / "explicacao_lime_individual.html")
        )

    probabilidade = pipeline.predict_proba(X.iloc[[indice]])[0, 1]
    print(f"\nProbabilidade sintética de procedência do caso explicado: {probabilidade:.3f}")
    print("As explicações são indicativas e exigem revisão humana.")
    print(f"Resultados salvos em: {OUTPUT_DIR.resolve()}")
    return pipeline, importancia, explicacao_individual


if __name__ == "__main__":
    executar()
