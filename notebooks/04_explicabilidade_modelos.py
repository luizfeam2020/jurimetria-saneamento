"""
Notebook 4: Explicabilidade com SHAP e LIME
Framework de Suporte à Decisão - Saneamento Básico
Mestrado Profissionalizante - CASAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 50)

print("=" * 60)
print("EXPLICABILIDADE COM SHAP E LIME")
print("Framework de Jurimetria para Saneamento Básico")
print("=" * 60)

# =============================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS
# =============================================
print("\n[1] Gerando dados sintéticos...")
np.random.seed(42)
n_processos = 5000  # Menor para SHAP ser mais rápido

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
df['target_procedencia'] = df['resultado_sentenca'].map({
    'PROCEDENTE': 1, 'IMPROCEDENTE': 0, 
    'PARCIALMENTE_PROCEDENTE': 0.5, 'EXTINTO': np.nan
})
df = df.dropna(subset=['target_procedencia'])
y_binary = (df['target_procedencia'] >= 0.5).astype(int)

# Features
feature_cols = ['valor_causa', 'tempo_total_dias', 'numero_audiencias', 
                'numero_pericias', 'numero_recursos', 'ano_distribuicao', 'mes_distribuicao']
cat_cols = ['tribunal', 'comarca', 'tipo_parte_autora', 'tipo_parte_reu', 
            'assunto_s

