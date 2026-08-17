"""
Notebook 1: Análise Exploratória e Jurimetria
Framework de Suporte à Decisão - Saneamento Básico
Mestrado Profissionalizante - CASAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 50)

print("=" * 60)
print("ANÁLISE EXPLORATÓRIA E JURIMETRIA")
print("Framework de Jurimetria para Saneamento Básico")
print("=" * 60)

# 1. GERAÇÃO DE DADOS SINTÉTICOS
print("\n[1] Gerando dados sintéticos...")
np.random.seed(42)
n_processos = 10000

tribunais = ['TJSC', 'STJ', 'STF', 'TRF4']
varas = [f'{t} - {i}ª Vara Cível' for t in tribunais for i in range(1, 6)]
assuntos = ['AGUA', 'ESGOTO', 'DRENAGEM', 'RESIDUOS', 'TARIFA', 
            'QUALIDADE', 'INTERRUPCAO', 'DANO_AMBIENTAL']
tipos_demanda = ['INDIVIDUAL', 'COLETIVA', 'ACAO_CIVIL_PUBLICA', 'MANDADO_SEGURANCA']
resultados = ['PROCEDENTE', 'IMPROCEDENTE', 'PARCIALMENTE_PROCEDENTE', 'EXTINTO']

data = {
    'id_processo': range(1, n_processos + 1),
    'numero_unico': [f'{np.random.randint(1000000, 9999999)}-{np.random.randint(10, 99)}.{np.random.randint(2015, 2024)}.{np.random.choice([4, 8, 24])}.{np.random.randint(1, 9)}.{np.random.randint(1000, 9999)}' for _ in range(n_processos)],
    'classe': np.random.choice(['Procedimento Comum', 'Procedimento Sumário', 
                                 'Mandado de Segurança', 'Ação Civil Pública', 'Recurso'], n_processos),
    'assunto_principal': np.random.choice(assuntos, n_processos),
    'assunto_secundario': np.random.choice(assuntos + ['OUTROS'], n_processos),
    'valor_causa': np.random.lognormal(mean=8, sigma=1.5, size=n_processos),
    'data_distribuicao': pd.date_range(start='2015-01-01', end='2024-12-31', periods=n_processos),
    'tribunal': np.random.choice(tribunais, n_processos),
    'vara': np.random.choice(varas, n_processos),
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

df['ano_distribuicao'] = df['data_distribuicao'].dt.year
df['mes_distribuicao'] = df['data_distribuicao'].dt.month
df['trimestre_distribuicao'] = df['data_distribuicao'].dt.quarter
df['faixa_valor_causa'] = pd.cut(df['valor_causa'], 
                                   bins=[0, 1000, 5000, 10000, 50000, 100000, 500000, 1e6, 1e7], 
                                   labels=['<1k', '1k-5k', '5k-10k', '10k-50k', '50k-100k', 
                                           '100k-500k', '500k-1M', '>1M'])
df['flag_acordo'] = (df['valor_acordo'] > 0).astype(int)
df['target_procedencia'] = df['resultado_sentenca'].map({
    'PROCEDENTE': 1, 'IMPROCEDENTE': 0, 
    'PARCIALMENTE_PROCEDENTE': 0.5, 'EXTINTO': np.nan
})

print(f"Dataset gerado: {df.shape[0]} processos, {df.shape[1]} colunas")
print(df.head())

# 2. INDICADORES JURIMÉTRICOS
print("\n" + "=" * 60)
print("INDICADORES JURIMÉTRICOS")
print("=" * 60)

print("\n[2.1] TAXA DE PROCEDÊNCIA")
taxa_procedencia = df['resultado_sentenca'].value_counts(normalize=True) * 100
print(taxa_procedencia)

print(f"\n[2.2] Taxa de Acordo: {df['flag_acordo'].mean() * 100:.2f}%")
print(f"\n[2.3] Tempo Médio: {df['tempo_total_dias'].mean():.0f} dias")
print(f"Tempo Mediano: {df['tempo_total_dias'].median():.0f} dias")

print("\n[2.4] DISTRIBUIÇÃO POR ASSUNTO DE SANEAMENTO")
print(df['assunto_saneamento'].value_counts())

print("\n[2.5] INDICADORES POR TRIBUNAL")
indicadores_tribunal = df.groupby('tribunal').agg({
    'id_processo': 'count',
    'target_procedencia': 'mean',
    'flag_acordo': 'mean',
    'tempo_total_dias': 'mean',
    'valor_causa': 'mean',
    'valor_condenacao': 'mean'
}).round(2)
indicadores_tribunal.columns = ['Total Processos', 'Taxa Procedência', 'Taxa Acordo', 
                                 'Tempo Médio (dias)', 'Valor Causa Médio', 'Valor Condenação Médio']
print(indicadores_tribunal)

# 3. VISUALIZAÇÕES
print("\n" + "=" * 60)
print("GERANDO VISUALIZAÇÕES...")
print("=" * 60)

plt.figure(figsize=(14, 6))
serie_anual = df.groupby('ano_distribuicao').size()
plt.plot(serie_anual.index, serie_anual.values, marker='o', linewidth=2, markersize=8)
plt.title('Evolução do Número de Processos por Ano', fontsize=14, fontweight='bold')
plt.xlabel('Ano de Distribuição')
plt.ylabel('Número de Processos')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_serie_temporal.png', dpi=150)
plt.show()
print("Gráfico salvo: 01_serie_temporal.png")

plt.figure(figsize=(10, 8))
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
resultado_counts = df['resultado_sentenca'].value_counts()
plt.pie(resultado_counts.values, labels=resultado_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
plt.title('Distribuição dos Resultados das Sentenças', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_distribuicao_resultados.png', dpi=150)
plt.show()
print("Gráfico salvo: 02_distribuicao_resultados.png")

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 60)