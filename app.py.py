import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings
from PIL import Image # Para carregar as imagens

# Configurações da página
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# FUNÇÃO PARA CARREGAR OS ARQUIVOS
# ---------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline de modelo treinado (4 variáveis)."""
    try:
        modelo = joblib.load('modelo_aluguel_4vars.pkl')
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_aluguel_4vars.pkl' não encontrado.")
        st.info("Certifique-se de que o arquivo .pkl está no repositório GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.info("Verifique se a versão do scikit-learn no requirements.txt é a mesma do Colab.")
        st.stop()

@st.cache_data
def carregar_bairros():
    """Carrega a lista de bairros únicos."""
    try:
        with open('bairros_unicos.json', 'r', encoding='utf-8') as f:
            bairros = json.load(f)
        return bairros
    except FileNotFoundError:
        st.error("Arquivo 'bairros_unicos.json' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar a lista de bairros: {e}")
        return []

# --- Carregar os dados ---
modelo_pipeline = carregar_modelo()
bairros_unicos = carregar_bairros()

# ---------------------------------------------------------------------
# BARRA LATERAL (Sidebar)
# ---------------------------------------------------------------------

st.sidebar.title("🏙️ Estimador de Aluguel")
st.sidebar.markdown("Preencha os dados do imóvel para fazer uma previsão.")
st.sidebar.header("Características do Imóvel")

# Inputs (baseado no modelo de 4 variáveis)
metragem = st.sidebar.slider("Metragem (m²)", 20, 300, 65, 5)
quartos = st.sidebar.selectbox("Quartos", [0, 1, 2, 3, 4, 5, 6], index=2)
banheiros = st.sidebar.selectbox("Banheiros", [1, 2, 3, 4, 5], index=1)

bairro_default_index = 0
if 'aclimacao' in bairros_unicos:
    bairro_default_index = bairros_unicos.index('aclimacao')
bairro = st.sidebar.selectbox("Bairro", bairros_unicos, index=bairro_default_index)

prever = st.sidebar.button("Estimar Valor", type="primary")

# ---------------------------------------------------------------------
# LAYOUT PRINCIPAL COM ABAS
# ---------------------------------------------------------------------

st.title("Simulador e Análise de Mercado de Aluguéis")
tab1, tab2 = st.tabs(["🏠 Simulador", "📊 Análise de Mercado"])

# --- ABA 1: SIMULADOR ---
with tab1:
    st.header("Resultado da Simulação")
    
    if prever:
        try:
            # 1. Criar DataFrame de entrada
            input_data = pd.DataFrame({
                'Metragem': [metragem],
                'Quartos': [quartos],
                'Banheiros': [banheiros],
                'Bairro': [bairro]
            })
            
            # 2. Fazer a previsão
            previsao = modelo_pipeline.predict(input_data)[0]
            preco_formatado = f"R$ {previsao:,.2f}"
            
            st.success(f"## Valor Total Estimado: {preco_formatado}")
            
            st.markdown("---")
            st.subheader("Resumo dos Dados Informados:")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Metragem:** {metragem} m²")
                st.write(f"**Quartos:** {quartos}")
            with col2:
                st.write(f"**Banheiros:** {banheiros}")
                st.write(f"**Bairro:** {bairro.title()}")
        
        except Exception as e:
            st.error(f"Erro ao realizar a previsão: {e}")
    
    else:
        st.info("Preencha os dados na barra lateral à esquerda e clique em 'Estimar Valor'.")

    st.markdown("---")
    st.info(
        "**Sobre o Modelo:**\n"
        f"* **Modelo Utilizado:** Random Forest (R²: 0.878)\n"
        f"* **Base de Dados:** 11.283 imóveis (após limpeza)"
    )

# --- ABA 2: ANÁLISE DE MERCADO ---
with tab2:
    st.header("Análise Exploratória dos Dados")
    st.write("Estes gráficos são baseados nos 11.283 imóveis da base de dados.")

    try:
        # Tenta carregar as imagens
        img_caros = Image.open('graf_top_caros.png')
        img_baratos = Image.open('graf_top_baratos.png')
        img_metro = Image.open('graf_metragem_preco.png')
        img_quartos = Image.open('graf_quartos_preco.png')
        
        # Exibe os gráficos
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_caros, caption='Top 10 Bairros por Preço Médio Total')
            st.image(img_metro, caption='Metragem vs. Preço Total')
            
        with col2:
            st.image(img_baratos, caption='Top 10 Bairros Mais Baratos (Valor Total Médio)')
            st.image(img_quartos, caption='Preço por Número de Quartos')
            
    except FileNotFoundError:
        st.error("Arquivos de gráfico ('graf_*.png') não encontrados.")
        st.info("Certifique-se de que os arquivos .png estão no repositório do GitHub.")
    except Exception as e:
        st.error(f"Erro ao carregar gráficos: {e}")
