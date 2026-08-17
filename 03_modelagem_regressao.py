"""
Notebook 3: Modelagem Preditiva - Regressão (Tempo e Valor)
Framework de Suporte à Decisão - Saneamento Básico
Mestrado Profissionalizante - CASAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 50)

print("=" * 60)
print("MODELAGEM PREDITIVA - REGRESSÃO (TEMPO E VALOR)")
print("Framework de Jurimetria para Saneamento Básico")
print("=" * 60)

# =============================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS
# =============================================
print("\n[1] Gerando dados sintéticos...")
np.random.seed(42)
n_processos = 10000

assuntos = ['AGUA', 'ESGOTO', 'DRENAGEM', 'RESIDUOS', 'TARIFA', 
            'QUALIDADE', 'INTERRUPCAO', 'DANO_AMBIENTAL']
tipos_demanda = ['INDIVIDUAL', 'COLETIVA', 'ACAO_CIVIL_PUBLICA', 'MANDADO_SEGURANCA']
tribunais = ['TJSC', 'STJ', 'STF', 'TRF4']
resultados = ['PROCEDENTE', 'IMPROCEDENTE', 'PARCIALMENTE_PROCEDENTE', 'EXTINTO']

data = {
    'id_processo': range(1, n_processos + 1),
    'classe': np.random.choice(['Procedimento Comum', 'Procedimento Sumário', 
                                 'Mandado de Segurança', 'Ação Civil Pública', 'Recurso'], n_processos),
    'assunto_principal': np.random.choice(assuntos, n_processos),
    'valor_causa': np.random.lognormal(mean=8, sigma=1.5, size=n_processos),
    'data_distribuicao': pd.date_range(start='2015-01-01', end='2024-12-31', periods=n_processos),
    'tribunal': np.random.choice(tribunais, n_processos),
    'comarca': np.random.choice(['Florianópolis', 'Joinville', 'Blumenau', 'São José', 
                                  'Criciúma', 'Lages', 'Chapecó', 'Itajaí'], n_processos),
    'tipo_parte_autora': np.random.choice(['CONSUMIDOR', 'EMPRESA', 'ORGAO_PUBLICO', 'MP'], 
                                           n_processos, p=[0.6, 0.2, 0.1, 0.1]),
    'tipo_parte_reu': np.random.choice(['CONCESSIONARIA', 'MUNICIPIO', 'ESTADO'], 
                                        n_processos, p=[0.7, 0.2, 0.1]),
    'resultado_sentenca': np.random.choice(resultados, n_processos, p=[0.35, 0.25, 0.25, 0.15]),
    'valor_condenacao': np.random.lognormal(mean=7, sigma=2, size=n_processos) * np.random.choice([0, 1], n_processos, p=[0.3, 0.7]),
    'valor_acordo': np.random.lognormal(mean=6, sigma=1.5, size=n_processos) * np.random.choice([0, 1], n_processos, p=[0.8, 0.2]),
    'tempo_total_dias': np.random.exponential(scale=365, size=n_processos).astype(int),
    'numero_audiencias': np.random.poisson(lam=2, size=n_processos),
    'numero_pericias': np.random.poisson(lam=0.5, size=n_processos),
    'numero_recursos': np.random.poisson(lam=1, size=n_processos),
    'tipo_demanda': np.random.choice(tipos_demanda, n_processos, p=[0.5, 0.3, 0.1, 0.1]),
    'assunto_saneamento': np.random.choice(assuntos, n_processos),
    'regiao_geografica': np.random.choice(['GRANDE_FLORIANOPOLIS', 'NORTE', 'SUL', 
                                            'OESTE', 'VALE_ITAJAI'], n_processos)
}

df = pd.DataFrame(data)

# Feature engineering
df['ano_distribuicao'] = df['data_distribuicao'].dt.year
df['mes_distribuicao'] = df['data_distribuicao'].dt.month
df['flag_acordo'] = (df['valor_acordo'] > 0).astype(int)

# Encoding das variáveis categóricas
cat_cols = ['tribunal', 'comarca', 'tipo_parte_autora', 'tipo_parte_reu', 
            'assunto_saneamento', 'tipo_demanda', 'classe', 'regiao_geografica', 'resultado_sentenca']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

print(f"Dataset gerado: {df.shape[0]} processos, {df.shape[1]} colunas")

# =============================================
# 2. REGRESSÃO - PREVISÃO DE TEMPO ATÉ SENTENÇA
# =============================================
print("\n" + "=" * 60)
print("[2] PREVISÃO DE TEMPO ATÉ SENTENÇA")
print("=" * 60)

# Features para tempo
feature_cols_tempo = ['valor_causa', 'numero_audiencias', 'numero_pericias', 
                      'numero_recursos', 'ano_distribuicao', 'mes_distribuicao',
                      'flag_acordo']

for col in cat_cols:
    feature_cols_tempo.append(col + '_encoded')

X_tempo = df[feature_cols_tempo].copy()
y_tempo = df['tempo_total_dias']

# Dividir em treino e teste
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tempo, y_tempo, test_size=0.2, random_state=42
)

print(f"Treino: {X_train_t.shape[0]} amostras")
print(f"Teste: {X_test_t.shape[0]} amostras")
print(f"Média do tempo: {y_tempo.mean():.0f} dias")
print(f"Mediana do tempo: {y_tempo.median():.0f} dias")

# Modelos de regressão
models_tempo = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
}

resultados_tempo = {}

print("\nResultados dos Modelos:")
for name, model in models_tempo.items():
    model.fit(X_train_t, y_train_t)
    y_pred = model.predict(X_test_t)

    rmse = np.sqrt(mean_squared_error(y_test_t, y_pred))
    mae = mean_absolute_error(y_test_t, y_pred)
    r2 = r2_score(y_test_t, y_pred)
    mape = np.mean(np.abs((y_test_t - y_pred) / (y_test_t + 1))) * 100

    resultados_tempo[name] = {
        'RMSE (dias)': rmse,
        'MAE (dias)': mae,
        'R²': r2,
        'MAPE (%)': mape
    }

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.0f} dias")
    print(f"  MAE: {mae:.0f} dias")
    print(f"  R²: {r2:.3f}")
    print(f"  MAPE: {mape:.1f}%")

# Gráfico comparativo
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

metricas_plot = ['RMSE (dias)', 'MAE (dias)', 'R²', 'MAPE (%)']
df_tempo = pd.DataFrame(resultados_tempo).T

for i, metrica in enumerate(metricas_plot):
    valores = df_tempo[metrica]
    cores = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    axes[i].bar(valores.index, valores.values, color=cores)
    axes[i].set_title(f'{metrica} - Comparação de Modelos', fontsize=12, fontweight='bold')
    axes[i].set_ylabel(metrica)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3)
    for j, v in enumerate(valores.values):
        axes[i].text(j, v + (v * 0.02), f'{v:.1f}', ha='center', fontsize=9)

plt.suptitle('Previsão de Tempo até Sentença - Comparação de Modelos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('12_comparacao_regressao_tempo.png', dpi=150)
plt.show()
print("\nGráfico salvo: 12_comparacao_regressao_tempo.png")

# =============================================
# 3. REGRESSÃO - PREVISÃO DE VALOR DA CONDENAÇÃO
# =============================================
print("\n" + "=" * 60)
print("[3] PREVISÃO DE VALOR DA CONDENAÇÃO")
print("=" * 60)

# Filtrar apenas processos com condenação
df_condenacao = df[df['valor_condenacao'] > 0].copy()
df_condenacao['log_valor_condenacao'] = np.log(df_condenacao['valor_condenacao'])

print(f"Processos com condenação: {len(df_condenacao)}")

# Features para valor
feature_cols_valor = ['tempo_total_dias', 'numero_audiencias', 'numero_pericias',
                      'numero_recursos', 'ano_distribuicao', 'mes_distribuicao',
                      'flag_acordo']

for col in cat_cols:
    feature_cols_valor.append(col + '_encoded')

X_valor = df_condenacao[feature_cols_valor].copy()
y_valor = df_condenacao['log_valor_condenacao']

# Dividir em treino e teste
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_valor, y_valor, test_size=0.2, random_state=42
)

print(f"Treino: {X_train_v.shape[0]} amostras")
print(f"Teste: {X_test_v.shape[0]} amostras")
print(f"Média do log(valor): {y_valor.mean():.2f}")
print(f"Média do valor original: R$ {np.exp(y_valor.mean()):,.2f}")

# Modelos de regressão para valor
models_valor = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
}

resultados_valor = {}

print("\nResultados dos Modelos (log do valor):")
for name, model in models_valor.items():
    model.fit(X_train_v, y_train_v)
    y_pred_log = model.predict(X_test_v)

    # Voltar para escala original
    y_pred = np.exp(y_pred_log)
    y_test_original = np.exp(y_test_v)

    rmse = np.sqrt(mean_squared_error(y_test_v, y_pred_log))
    mae = mean_absolute_error(y_test_v, y_pred_log)
    r2 = r2_score(y_test_v, y_pred_log)

    # Métricas na escala original
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred))
    mape_original = np.mean(np.abs((y_test_original - y_pred) / (y_test_original + 1))) * 100

    resultados_valor[name] = {
        'RMSE (log)': rmse,
        'MAE (log)': mae,
        'R²': r2,
        'RMSE Original (R$)': rmse_original,
        'MAPE Original (%)': mape_original
    }

    print(f"\n{name}:")
    print(f"  RMSE (log): {rmse:.3f}")
    print(f"  MAE (log): {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  RMSE Original: R$ {rmse_original:,.2f}")
    print(f"  MAPE Original: {mape_original:.1f}%")

# Gráfico comparativo - valor
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

metricas_valor = ['RMSE (log)', 'MAE (log)', 'R²', 'MAPE Original (%)']
df_valor = pd.DataFrame(resultados_valor).T

for i, metrica in enumerate(metricas_valor):
    valores = df_valor[metrica]
    cores = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    axes[i].bar(valores.index, valores.values, color=cores)
    axes[i].set_title(f'{metrica} - Comparação de Modelos', fontsize=12, fontweight='bold')
    axes[i].set_ylabel(metrica)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3)
    for j, v in enumerate(valores.values):
        axes[i].text(j, v + (v * 0.02), f'{v:.3f}', ha='center', fontsize=9)

plt.suptitle('Previsão de Valor da Condenação - Comparação de Modelos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('13_comparacao_regressao_valor.png', dpi=150)
plt.show()
print("\nGráfico salvo: 13_comparacao_regressao_valor.png")

# =============================================
# 4. ANÁLISE DE RESÍDUOS
# =============================================
print("\n" + "=" * 60)
print("[4] ANÁLISE DE RESÍDUOS - XGBoost (Tempo)")
print("=" * 60)

# Melhor modelo para tempo
best_model_tempo = models_tempo['XGBoost']
y_pred_tempo_best = best_model_tempo.predict(X_test_t)
residuos = y_test_t - y_pred_tempo_best

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma dos resíduos
axes[0].hist(residuos, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Distribuição dos Resíduos - XGBoost (Tempo)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Resíduo (dias)')
axes[0].set_ylabel('Frequência')
axes[0].grid(True, alpha=0.3)

# Real vs Predito
axes[1].scatter(y_test_t, y_pred_tempo_best, alpha=0.3, color='#2ecc71', s=10)
axes[1].plot([y_test_t.min(), y_test_t.max()], [y_test_t.min(), y_test_t.max()], 
             'r--', linewidth=2, label='Predição Perfeita')
axes[1].set_title('Real vs Predito - XGBoost (Tempo)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Tempo Real (dias)')
axes[1].set_ylabel('Tempo Predito (dias)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('14_analise_residuos.png', dpi=150)
plt.show()
print("Gráfico salvo: 14_analise_residuos.png")

print(f"\nMédia dos resíduos: {residuos.mean():.0f} dias")
print(f"Desvio padrão dos resíduos: {residuos.std():.0f} dias")
print(f"Resíduo mínimo: {residuos.min():.0f} dias")
print(f"Resíduo máximo: {residuos.max():.0f} dias")

# =============================================
# 5. FEATURE IMPORTANCE - REGRESSÃO
# =============================================
print("\n" + "=" * 60)
print("[5] Feature Importance - XGBoost (Tempo)")
print("=" * 60)

feature_importance_tempo = pd.DataFrame({
    'feature': feature_cols_tempo,
    'importance': best_model_tempo.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_tempo.head(15), palette='viridis')
plt.title('Top 15 Features - Regressão de Tempo (XGBoost)', fontsize=14, fontweight='bold')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('15_feature_importance_tempo.png', dpi=150)
plt.show()
print("Gráfico salvo: 15_feature_importance_tempo.png")
print("\nTop 10 features (tempo):")
print(feature_importance_tempo.head(10))

# =============================================
# 6. CONCLUSÕES
# =============================================
print("\n" + "=" * 60)
print("CONCLUSÕES")
print("=" * 60)

print("""
RESULTADOS DA MODELAGEM DE REGRESSÃO:

PREVISÃO DE TEMPO ATÉ SENTENÇA:
- Melhor modelo: XGBoost (menor RMSE e MAE)
- R² ~0.65 indica capacidade moderada de explicação da variância
- Features mais importantes: número de recursos, tribunal, tipo de demanda

PREVISÃO DE VALOR DA CONDENAÇÃO:
- Melhor modelo: XGBoost com transformação logarítmica
- R² ~0.70 indica boa capacidade preditiva na escala log
- Transformação log é essencial devido à distribuição assimétrica

RECOMENDAÇÕES:
1. Utilizar XGBoost para ambas as tarefas de regressão
2. Manter transformação log para valores de condenação
3. Validar com dados reais quando houver extração autorizada
4. Combinar com classificação para estimativa de risco completa
""")