"""
Notebook 2: Modelagem Preditiva - Classificação de Procedência
Framework de Suporte à Decisão - Saneamento Básico
Mestrado Profissionalizante - CASAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 50)

print("=" * 60)
print("MODELAGEM PREDITIVA - CLASSIFICAÇÃO DE PROCEDÊNCIA")
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
    'tempo_total_dias': np.random.exponential(scale=365, size=n_processos).astype(int),
    'numero_audiencias': np.random.poisson(lam=2, size

