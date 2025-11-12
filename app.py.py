import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings

# Ignorar avisos
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# FUNÇÃO PARA CARREGAR OS ARQUIVOS
# ---------------------------------------------------------------------

@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline de modelo treinado."""
    try:
        # Certifique-se de que o nome do arquivo é o mesmo que você salvou!
        modelo = joblib.load('modelo_aluguel_4vars.pkl')
        return modelo
    except FileNotFoundError:
        st.error("Arquivo 'modelo_aluguel_4vars.pkl' não encontrado.")
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
        st.error("Arquivo 'bairros_unicos.json' não encontrado.")
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
st.markdown("Preencha os dados abaixo para estimar o valor total do aluguel (R² de **0.878**).")

st.sidebar.header("Preencha os dados do imóvel:")

# Features usadas no seu modelo (X)
metragem = st.sidebar.slider(
    "Metragem (m²)",
    min_value=20,
    max_value=300,
    value=65,
    step=5
)

quartos = st.sidebar.selectbox(
    "Quartos",
    options=[0, 1, 2, 3, 4, 5, 6],
    index=2 # Padrão 2
)

banheiros = st.sidebar.selectbox(
    "Banheiros",
    options=[1, 2, 3, 4, 5],
    index=1 # Padrão 1 (no seu notebook era 2, mas 1 é mais comum)
)

# Input categórico
bairro_default_index = 0
if 'aclimacao' in bairros_unicos:
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
        # 1. Criar DataFrame de entrada
        input_data = pd.DataFrame({
            'Metragem': [metragem],
            'Quartos': [quartos],
            'Banheiros': [banheiros],
            'Bairro': [bairro]
        })
        
        # 2. Fazer a previsão
        previsao = modelo_pipeline.predict(input_data)[0]
        
        # 3. Exibir o resultado
        st.subheader("Valor Total Estimado (Aluguel + Condomínio + IPTU):")
        preco_formatado = f"R$ {previsao:,.2f}"
        st.success(f"## {preco_formatado}")
        
        st.markdown("---")
        st.subheader("Resumo dos Dados Informados:")
        st.write(f"**Metragem:** {metragem} m²")
        st.write(f"**Quartos:** {quartos}")
        st.write(f"**Banheiros:** {banheiros}")
        st.write(f"**Bairro:** {bairro.title()}")

        st.info(f"**Modelo Utilizado:** Random Forest (R²: 0.878)")

    except Exception as e:
        st.error(f"Erro ao realizar a previsão: {e}")

else:
    st.info("Preencha os dados ao lado e clique em 'Estimar Valor'.")

