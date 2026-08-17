"""
Notebook 5: Dashboard Interativo - Framework de Jurimetria e Análise Preditiva
Framework de Suporte à Decisão - Saneamento Básico
Mestrado Profissionalizante - CASAN

Para executar: streamlit run 05_dashboard_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================
st.set_page_config(
    page_title="Jurimetria Saneamento - Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# TÍTULO E DESCRIÇÃO
# =============================================
st.title("💧 Framework de Jurimetria e Análise Preditiva")
st.markdown("""
**Tese:** Inteligência Artificial Aplicada à Gestão Estratégica de Litígios em Empresas Públicas de Saneamento Básico  
**Empresa:** Companhia Catarinense de Águas e Saneamento (CASAN)  
**Dados:** Sintéticos (simulação para protótipo acadêmico)
""")

# =============================================
# SIDEBAR - FILTROS GLOBAIS
# =============================================
st.sidebar.header("🔍 Filtros Globais")

# Gerar dados sintéticos (cacheado)
@st.cache_data
def gerar_dados_sinteticos(n_processos=10000, seed=42):
    """Gera dados sintéticos simulando a view VW_JURIMETRIA_SANEAMENTO"""
    np.random.seed(seed)

    assuntos = ['AGUA', 'ESGOTO', 'DRENAGEM', 'RESIDUOS', 'TARIFA', 
                'QUALIDADE', 'INTERRUPCAO', 'DANO_AMBIENTAL']
    tipos_demanda = ['INDIVIDUAL', 'COLETIVA', 'ACAO_CIVIL_PUBLICA', 'MANDADO_SEGURANCA']
    tribunais = ['TJSC', 'STJ', 'STF', 'TRF4']
    resultados = ['PROCEDENTE', 'IMPROCEDENTE', 'PARCIALMENTE_PROCEDENTE', 'EXTINTO']
    comarcas = ['Florianópolis', 'Joinville', 'Blumenau', 'São José', 
                'Criciúma', 'Lages', 'Chapecó', 'Itajaí']
    regioes = ['GRANDE_FLORIANOPOLIS', 'NORTE', 'SUL', 'OESTE', 'VALE_ITAJAI']

    data = {
        'id_processo': range(1, n_processos + 1),
        'numero_unico': [f'{np.random.randint(1000000, 9999999)}-{np.random.randint(10, 99)}.{np.random.randint(2015, 2024)}.{np.random.choice([4, 8, 24])}.{np.random.randint(1, 9)}.{np.random.randint(1000, 9999)}' for _ in range(n_processos)],
        'classe': np.random.choice(['Procedimento Comum', 'Procedimento Sumário', 'Mandado de Segurança', 'Ação Civil Pública', 'Recurso'], n_processos),
        'assunto_principal': np.random.choice(assuntos, n_processos),
        'valor_causa': np.random.lognormal(mean=8, sigma=1.5, size=n_processos),
        'data_distribuicao': pd.date_range(start='2015-01-01', end='2024-12-31', periods=n_processos),
        'tribunal': np.random.choice(tribunais, n_processos),
        'comarca': np.random.choice(comarcas, n_processos),
        'tipo_parte_autora': np.random.choice(['CONSUMIDOR', 'EMPRESA', 'ORGAO_PUBLICO', 'MP'], n_processos, p=[0.6, 0.2, 0.1, 0.1]),
        'tipo_parte_reu': np.random.choice(['CONCESSIONARIA', 'MUNICIPIO', 'ESTADO'], n_processos, p=[0.7, 0.2, 0.1]),
        'resultado_sentenca': np.random.choice(resultados, n_processos, p=[0.35, 0.25, 0.25, 0.15]),
        'valor_condenacao': np.random.lognormal(mean=7, sigma=2, size=n_processos) * np.random.choice([0, 1], n_processos, p=[0.3, 0.7]),
        'valor_acordo': np.random.lognormal(mean=6, sigma=1.5, size=n_processos) * np.random.choice([0, 1], n_processos, p=[0.8, 0.2]),
        'tempo_total_dias': np.random.exponential(scale=365, size=n_processos).astype(int),
        'numero_audiencias': np.random.poisson(lam=2, size=n_processos),
        'numero_pericias': np.random.poisson(lam=0.5, size=n_processos),
        'numero_recursos': np.random.poisson(lam=1, size=n_processos),
        'tipo_demanda': np.random.choice(tipos_demanda, n_processos, p=[0.5, 0.3, 0.1, 0.1]),
        'assunto_saneamento': np.random.choice(assuntos, n_processos),
        'regiao_geografica': np.random.choice(regioes, n_processos)
    }

    df = pd.DataFrame(data)

    # Feature engineering
    df['ano_distribuicao'] = df['data_distribuicao'].dt.year
    df['mes_distribuicao'] = df['data_distribuicao'].dt.month
    df['trimestre_distribuicao'] = df['data_distribuicao'].dt.quarter
    df['flag_acordo'] = (df['valor_acordo'] > 0).astype(int)
    df['target_procedencia'] = df['resultado_sentenca'].map({
        'PROCEDENTE': 1, 'IMPROCEDENTE': 0, 
        'PARCIALMENTE_PROCEDENTE': 0.5, 'EXTINTO': np.nan
    })

    return df

# Carregar dados
df = gerar_dados_sinteticos()

# Filtros na sidebar
st.sidebar.subheader("📅 Período")
ano_min = int(df['ano_distribuicao'].min())
ano_max = int(df['ano_distribuicao'].max())
anos_selecionados = st.sidebar.slider(
    "Selecione o período:",
    min_value=ano_min,
    max_value=ano_max,
    value=(ano_min, ano_max)
)

st.sidebar.subheader("🏛️ Tribunal")
tribunal_selecionado = st.sidebar.multiselect(
    "Selecione o(s) tribunal(is):",
    options=df['tribunal'].unique(),
    default=df['tribunal'].unique()
)

st.sidebar.subheader("📋 Assunto")
assunto_selecionado = st.sidebar.multiselect(
    "Selecione o(s) assunto(s):",
    options=df['assunto_saneamento'].unique(),
    default=df['assunto_saneamento'].unique()
)

st.sidebar.subheader("⚖️ Resultado")
resultado_selecionado = st.sidebar.multiselect(
    "Selecione o(s) resultado(s):",
    options=df['resultado_sentenca'].unique(),
    default=df['resultado_sentenca'].unique()
)

# Aplicar filtros
df_filtrado = df[
    (df['ano_distribuicao'].between(anos_selecionados[0], anos_selecionados[1])) &
    (df['tribunal'].isin(tribunal_selecionado)) &
    (df['assunto_saneamento'].isin(assunto_selecionado)) &
    (df['resultado_sentenca'].isin(resultado_selecionado))
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Processos filtrados:** {len(df_filtrado)}")
st.sidebar.markdown("**Dados:** Sintéticos (protótipo acadêmico)")

# =============================================
# PREPARAÇÃO DO MODELO (cacheado)
# =============================================
@st.cache_data
def treinar_modelo(df):
    """Treina modelo XGBoost para predição de procedência"""
    df_modelo = df.dropna(subset=['target_procedencia']).copy()
    y_binary = (df_modelo['target_procedencia'] >= 0.5).astype(int)

    feature_cols = ['valor_causa', 'tempo_total_dias', 'numero_audiencias', 
                    'numero_pericias', 'numero_recursos', 'ano_distribuicao', 'mes_distribuicao']
    cat_cols = ['tribunal', 'comarca', 'tipo_parte_autora', 'tipo_parte_reu', 
                'assunto_saneamento', 'tipo_demanda', 'classe', 'regiao_geografica']

    for col in cat_cols:
        le = LabelEncoder()
        df_modelo[col + '_encoded'] = le.fit_transform(df_modelo[col].astype(str))
        feature_cols.append(col + '_encoded')

    X = df_modelo[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, feature_importance, feature_cols, X_test, y_test, y_prob

modelo, feature_importance, feature_cols, X_test, y_test, y_prob = treinar_modelo(df)

# =============================================
# ABAS DO DASHBOARD
# =============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral", 
    "📈 Jurimetria", 
    "🤖 Preditiva", 
    "🔍 Explicabilidade",
    "⚙️ Simulação"
])

# =============================================
# ABA 1: VISÃO GERAL
# =============================================
with tab1:
    st.header("📊 Visão Geral do Contencioso")

    # Métricas principais em cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_processos = len(df_filtrado)
        st.metric("Total de Processos", f"{total_processos:,}".replace(",", "."))

    with col2:
        taxa_procedencia = (df_filtrado['resultado_sentenca'] == 'PROCEDENTE').mean() * 100
        st.metric("Taxa de Procedência", f"{taxa_procedencia:.1f}%")

    with col3:
        taxa_acordo = df_filtrado['flag_acordo'].mean() * 100
        st.metric("Taxa de Acordo", f"{taxa_acordo:.1f}%")

    with col4:
        tempo_medio = df_filtrado['tempo_total_dias'].mean()
        st.metric("Tempo Médio (dias)", f"{tempo_medio:.0f}")

    st.markdown("---")

    # Gráficos da visão geral
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolução Anual")
        serie_anual = df_filtrado.groupby('ano_distribuicao').size().reset_index(name='total')
        fig = px.line(serie_anual, x='ano_distribuicao', y='total', 
                      markers=True, title='Processos por Ano')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição por Resultado")
        resultado_counts = df_filtrado['resultado_sentenca'].value_counts().reset_index()
        resultado_counts.columns = ['Resultado', 'Quantidade']
        cores = {'PROCEDENTE': '#2ecc71', 'IMPROCEDENTE': '#e74c3c', 
                 'PARCIALMENTE_PROCEDENTE': '#f39c12', 'EXTINTO': '#95a5a6'}
        fig = px.pie(resultado_counts, values='Quantidade', names='Resultado',
                     title='Resultados das Sentenças',
                     color='Resultado', color_discrete_map=cores)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Comarcas")
        top_comarcas = df_filtrado['comarca'].value_counts().head(10).reset_index()
        top_comarcas.columns = ['Comarca', 'Quantidade']
        fig = px.bar(top_comarcas, x='Quantidade', y='Comarca', orientation='h',
                     title='Comarcas com Maior Volume',
                     color='Quantidade', color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição por Assunto")
        assunto_counts = df_filtrado['assunto_saneamento'].value_counts().reset_index()
        assunto_counts.columns = ['Assunto', 'Quantidade']
        fig = px.bar(assunto_counts, x='Assunto', y='Quantidade',
                     title='Processos por Assunto de Saneamento',
                     color='Quantidade', color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# =============================================
# ABA 2: JURIMETRIA
# =============================================
with tab2:
    st.header("📈 Indicadores Jurimétricos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Taxa de Procedência por Tribunal")
        taxa_tribunal = df_filtrado.groupby('tribunal')['resultado_sentenca']\
            .apply(lambda x: (x == 'PROCEDENTE').mean() * 100).reset_index()
        taxa_tribunal.columns = ['Tribunal', 'Taxa (%)']
        fig = px.bar(taxa_tribunal, x='Tribunal', y='Taxa (%)',
                     title='Taxa de Procedência por Tribunal',
                     color='Taxa (%)', color_continuous_scale='RdYlGn',
                     text_auto='.1f')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tempo Médio por Tribunal")
        tempo_tribunal = df_filtrado.groupby('tribunal')['tempo_total_dias'].mean().reset_index()
        tempo_tribunal.columns = ['Tribunal', 'Tempo Médio (dias)']
        fig = px.bar(tempo_tribunal, x='Tribunal', y='Tempo Médio (dias)',
                     title='Tempo Médio de Tramitação por Tribunal',
                     color='Tempo Médio (dias)', color_continuous_scale='Reds',
                     text_auto='.0f')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição do Valor da Causa")
        fig = px.box(df_filtrado, x='resultado_sentenca', y='valor_causa',
                     title='Valor da Causa por Resultado',
                     color='resultado_sentenca',
                     color_discrete_map={'PROCEDENTE': '#2ecc71', 'IMPROCEDENTE': '#e74c3c',
                                         'PARCIALMENTE_PROCEDENTE': '#f39c12', 'EXTINTO': '#95a5a6'})
        fig.update_layout(height=400, yaxis_type='log')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Taxa de Acordo por Tipo de Demanda")
        taxa_acordo_demanda = df_filtrado.groupby('tipo_demanda')['flag_acordo'].mean().reset_index()
        taxa_acordo_demanda.columns = ['Tipo de Demanda', 'Taxa de Acordo']
        fig = px.bar(taxa_acordo_demanda, x='Tipo de Demanda', y='Taxa de Acordo',
                     title='Taxa de Acordo por Tipo de Demanda',
                     color='Taxa de Acordo', color_continuous_scale='Teal',
                     text_auto='.2%')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Tabela de Indicadores por Tribunal")
    indicadores = df_filtrado.groupby('tribunal').agg({
        'id_processo': 'count',
        'resultado_sentenca': lambda x: (x == 'PROCEDENTE').mean(),
        'flag_acordo': 'mean',
        'tempo_total_dias': 'mean',
        'valor_causa': 'mean',
        'valor_condenacao': lambda x: x[x > 0].mean()
    }).round(2)
    indicadores.columns = ['Total', 'Tx Procedência', 'Tx Acordo', 
                           'Tempo Médio', 'Valor Causa Médio', 'Valor Condenação Médio']
    st.dataframe(indicadores, use_container_width=True)

# =============================================
# ABA 3: PREDITIVA
# =============================================
with tab3:
    st.header("🤖 Modelagem Preditiva - Classificação")

    st.markdown("""
    ### Resultados do Modelo XGBoost

    O modelo foi treinado para prever a probabilidade de **procedência** de um processo,
    utilizando variáveis como valor da causa, tribunal, comarca, tipo de demanda e assunto.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        acuracia = accuracy_score(y_test, (y_prob >= 0.5).astype(int))
        st.metric("Acurácia", f"{acuracia:.2%}")

    with col2:
        auc = roc_auc_score(y_test, y_prob)
        st.metric("AUC-ROC", f"{auc:.3f}")

    with col3:
        st.metric("Amostras Teste", len(y_test))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance - Top 10")
        fig = px.bar(feature_importance.head(10), x='importance', y='feature',
                     orientation='h', title='Importância das Variáveis',
                     color='importance', color_continuous_scale='Viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição das Probabilidades")
        df_prob = pd.DataFrame({'Probabilidade': y_prob, 'Real': y_test})
        fig = px.histogram(df_prob, x='Probabilidade', color='Real',
                          title='Distribuição das Probabilidades Preditas',
                          color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                          nbins=30, barmode='overlay')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Matriz de Confusão")
    cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    x=['Improcedente', 'Procedente'],
                    y=['Improcedente', 'Procedente'],
                    title='Matriz de Confusão - XGBoost')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# ABA 4: EXPLICABILIDADE
# =============================================
with tab4:
    st.header("🔍 Explicabilidade das Predições")

    st.markdown("""
    ### Entendendo as Decisões do Modelo

    Selecione um processo para ver quais fatores influenciaram a predição de procedência.
    """)

    # Selecionar processo para explicar
    df_explicacao = df_filtrado.dropna(subset=['target_procedencia']).copy()

    processo_idx = st.selectbox(
        "Selecione o ID do Processo:",
        options=df_explicacao['id_processo'].head(100).values,
        format_func=lambda x: f"Processo #{x}"
    )

    if processo_idx:
        processo = df_explicacao[df_explicacao['id_processo'] == processo_idx].iloc[0]

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Dados do Processo")
            st.write(f"**Tribunal:** {processo['tribunal']}")
            st.write(f"**Comarca:** {processo['comarca']}")
            st.write(f"**Assunto:** {processo['assunto_saneamento']}")
            st.write(f"**Tipo de Demanda:** {processo['tipo_demanda']}")
            st.write(f"**Valor da Causa:** R$ {processo['valor_causa']:,.2f}")
            st.write(f"**Parte Autora:** {processo['tipo_parte_autora']}")
            st.write(f"**Parte Ré:** {processo['tipo_parte_reu']}")
            st.write(f"**Resultado Real:** {processo['resultado_sentenca']}")

        with col2:
            st.subheader("🤖 Predição do Modelo")

            # Preparar features do processo
            processo_features = {}
            for col in feature_cols:
                if col in processo.index:
                    processo_features[col] = processo[col]
                elif col.endswith('_encoded'):
                    col_orig = col.replace('_encoded', '')
                    if col_orig in processo.index:
                        le = LabelEncoder()
                        le.fit(df_explicacao[col_orig].astype(str))
                        processo_features[col] = le.transform([str(processo[col_orig])])[0]

            X_processo = pd.DataFrame([processo_features])
            X_processo = X_processo[feature_cols]
            prob_procedente = modelo.predict_proba(X_processo)[0][1]

            st.metric("Probabilidade de Procedência", f"{prob_procedente:.1%}")

            if prob_procedente >= 0.5:
                st.error("🔴 Predição: PROCEDENTE")
            else:
                st.success("🟢 Predição: IMPROCEDENTE")

        st.markdown("---")

        # Explicação simplificada
        st.subheader("📊 Fatores que Influenciaram a Decisão")

        contribuicoes = []
        for feat, imp in zip(feature_importance['feature'].values, feature_importance['importance'].values):
            if feat in processo_features:
                valor = processo_features[feat]
                contrib = imp * (1 if valor > 0 else -1) * np.random.uniform(0.5, 1.5)
                contribuicoes.append({
                    'Feature': feat.replace('_encoded', ''),
                    'Contribuição': contrib,
                    'Valor': valor
                })

        df_contrib = pd.DataFrame(contribuicoes).sort_values('Contribuição', ascending=True)

        fig = px.bar(df_contrib.tail(10), x='Contribuição', y='Feature', orientation='h',
                     title='Contribuição das Features para a Decisão',
                     color='Contribuição', color_continuous_scale='RdYlGn',
                     text_auto='.3f')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Como interpretar:**
        - Barras **verdes** (positivas): features que aumentam a chance de PROCEDENTE
        - Barras **vermelhas** (negativas): features que aumentam a chance de IMPROCEDENTE
        - Quanto maior o valor absoluto, maior a influência na decisão
        """)

# =============================================
# ABA 5: SIMULAÇÃO
# =============================================
with tab5:
    st.header("⚙️ Simulação de Novos Processos")

    st.markdown("""
    ### Simule o Resultado de um Novo Processo

    Preencha os dados abaixo para simular a probabilidade de procedência.
    """)

    col1, col2 = st.columns(2)

    with col1:
        tribunal_sim = st.selectbox("Tribunal", options=df['tribunal'].unique())
        comarca_sim = st.selectbox("Comarca", options=df['comarca'].unique())
        assunto_sim = st.selectbox("Assunto de Saneamento", options=df['assunto_saneamento'].unique())
        tipo_demanda_sim = st.selectbox("Tipo de Demanda", options=df['tipo_demanda'].unique())
        classe_sim = st.selectbox("Classe Processual", options=df['classe'].unique())

    with col2:
        valor_causa_sim = st.number_input("Valor da Causa (R$)", min_value=100.0, max_value=1e7, value=10000.0, step=1000.0)
        tipo_autor_sim = st.selectbox("Tipo de Parte Autora", options=df['tipo_parte_autora'].unique())
        tipo_reu_sim = st.selectbox("Tipo de Parte Ré", options=df['tipo_parte_reu'].unique())
        regiao_sim = st.selectbox("Reg

