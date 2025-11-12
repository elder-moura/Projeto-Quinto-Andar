import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings

# Ignorar avisos para um app mais limpo
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# FUNÇÃO PARA CARREGAR OS ARQUIVOS (MODELO E BAIRROS)
# Usamos @st.cache_resource e @st.cache_data para carregar os arquivos apenas uma vez
# ---------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline de modelo treinado."""
    try:
        modelo = joblib.load('modelo_aluguel_rf.pkl')
        return modelo
    except FileNotFoundError:
        st.error("Arquivo do modelo 'modelo_aluguel_rf.pkl' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.stop()

@st.cache_data
def carregar_bairros():
    """Carrega a lista de bairros únicos."""
    try:
        with open('bairros_unicos.json', 'r', encoding='utf-8') as f:
            bairros = json.load(f)
        return bairros
    except FileNotFoundError:
        st.error("Arquivo de bairros 'bairros_unicos.json' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar a lista de bairros: {e}")
        return []

# Carregar os arquivos
modelo_pipeline = carregar_modelo()
bairros_unicos = carregar_bairros()

# ---------------------------------------------------------------------
# INTERFACE DO USUÁRIO (Inputs na Barra Lateral)
# ---------------------------------------------------------------------

st.title("🏙️ Estimador de Aluguel de Imóveis")
st.markdown("Use este app para estimar o valor total (Aluguel + Condomínio + IPTU) de um imóvel com base no modelo de Random Forest (R² de 0.901) treinado em seus dados.")

# Inputs na barra lateral
st.sidebar.header("Preencha os dados do imóvel:")

# Features usadas no seu modelo (X)
metragem = st.sidebar.number_input(
    "Metragem (m²)",
    min_value=10,
    max_value=1000,
    value=70,
    step=5
)

quartos = st.sidebar.selectbox(
    "Quartos",
    options=[0, 1, 2, 3, 4, 5, 6, 7, 8], # Incluindo 0
    index=2 # Padrão 2
)

banheiros = st.sidebar.selectbox(
    "Banheiros",
    options=[0, 1, 2, 3, 4, 5, 6], # Incluindo 0
    index=1 # Padrão 1
)

vagas = st.sidebar.selectbox(
    "Vagas de Garagem",
    options=[0, 1, 2, 3, 4, 5],
    index=1 # Padrão 1
)

andar = st.sidebar.number_input(
    "Andar (Use 0 para térreo)", # Baseado na sua função clean_andar
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

# Inputs baseados nas suas funções de limpeza (clean_mobilia, clean_pet)
mobilia = st.sidebar.selectbox(
    "Mobiliado?",
    options=[0, 1],
    format_func=lambda x: "Sim" if x == 1 else "Não",
    index=0 # Padrão "Não"
)

pet = st.sidebar.selectbox(
    "Aceita Pet?",
    options=[0, 1],
    format_func=lambda x: "Sim" if x == 1 else "Não",
    index=1 # Padrão "Sim"
)

# Input categórico
bairro_default_index = 0
if 'aclimacao' in bairros_unicos: # Um bom padrão
    bairro_default_index = bairros_unicos.index('aclimacao')

bairro = st.sidebar.selectbox(
    "Bairro",
    options=bairros_unicos,
    index=bairro_default_index
)

# ---------------------------------------------------------------------
# LÓGICA DE PREVISÃO E EXIBIÇÃO
# ---------------------------------------------------------------------

# Botão para prever
if st.sidebar.button("Estimar Valor", type="primary"):
    try:
        # 1. Criar DataFrame de entrada com os nomes exatos das colunas
        input_data = pd.DataFrame({
            'Metragem': [metragem],
            'Quartos': [quartos],
            'Banheiros': [banheiros],
            'Mobilia': [mobilia],
            'Pet': [pet],
            'Vagas': [vagas],
            'Andar': [andar],
            'Bairro': [bairro]
        })
        
        # 2. Fazer a previsão
        # O pipeline cuida do OneHotEncoding do 'Bairro' automaticamente
        previsao = modelo_pipeline.predict(input_data)[0]
        
        # 3. Exibir o resultado
        st.subheader("Resultado da Previsão:")
        
        # Formatação de moeda
        preco_formatado = f"R$ {previsao:,.2f}"
        
        st.success(f"## {preco_formatado}")
        
        st.markdown("---")
        st.subheader("Resumo dos Dados Informados:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Metragem:** {metragem} m²")
            st.write(f"**Quartos:** {quartos}")
            st.write(f"**Banheiros:** {banheiros}")
            st.write(f"**Vagas:** {vagas}")
        with col2:
            st.write(f"**Andar:** {andar if andar > 0 else 'Térreo'}")
            st.write(f"**Mobiliado:** {'Sim' if mobilia == 1 else 'Não'}")
            st.write(f"**Aceita Pet:** {'Sim' if pet == 1 else 'Não'}")
            st.write(f"**Bairro:** {bairro.title()}")

        # Informações do modelo
        st.markdown("---")
        st.info(
            "**Informações do Modelo:**\n"
            "* **Modelo Utilizado:** Random Forest Regressor\n"
            f"* **Precisão (R²):** 0.901 (nos dados de teste do seu notebook)"
        )

    except Exception as e:
        st.error(f"Erro ao realizar a previsão: {e}")

else:
    st.info("Preencha os dados ao lado e clique em 'Estimar Valor'.")