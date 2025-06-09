import streamlit as st
import tempfile
import os

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from document_loaders import *

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'Pdf', 'Csv', 'Txt'
]

CONFIG_MODELOS = {'OpenAI': {'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini', "gpt-4.1 nano", "gpt-4.1 mini", 'gpt-4.1'],
                             'chat': ChatOpenAI}}

MEMORIA = ConversationBufferMemory()

def carrega_arquivos(tipo_arquivo, arquivo):
    documento = None
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    elif tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
    elif tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    elif tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    elif tipo_arquivo == 'Txt':
        conteudo = arquivo.read().decode('utf-8')
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode="w", encoding="utf-8") as temp:
            temp.write(conteudo)
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    return documento

    print(documento)
def carrega_modelo(modelo, tipo_arquivo, arquivo):
    
    documento = carrega_arquivos(tipo_arquivo, arquivo)

    system_message = '''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_arquivo, documento)

    #print(system_message)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    # Cria uma instância do modelo OpenAI com o modelo escolhido
    chat = CONFIG_MODELOS["OpenAI"]['chat'](model=modelo, api_key=OPENAI_API_KEY)
    chain = template | chat

    st.session_state['chain'] = chain
    
def pagina_chat():
    st.header('🤖Bem-vindo ao Oráculo', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carrege o Oráculo')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o oráculo')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario, 
            'chat_history': memoria.buffer_as_messages
            }))
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    # Defina as variáveis fora do with
    tipo_arquivo = None
    arquivo = None
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do vídeo')
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type=['pdf'])
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo csv', type=['csv'])
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo txt', type=['txt'])
        
    with tabs[1]:
        st.markdown(f'Provedor: OpenAI')
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS['OpenAI']['modelos'])

    # Botão só habilitado se arquivo estiver correto
    if st.button('Inicializar Oráculo', use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("Chave de API não encontrada no ambiente! Verifique seu .env.")
        elif tipo_arquivo in ['Site', 'Youtube'] and not arquivo:
            st.error("Digite a URL!")
        elif tipo_arquivo in ['Pdf', 'Csv', 'Txt'] and arquivo is None:
            st.error("Faça upload de um arquivo.")
        else:
            carrega_modelo(modelo, tipo_arquivo, arquivo)
            st.success(f'Modelo {modelo} inicializado!')
    if st.button('Apagar Histórico de Conversas', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()
    

if __name__=='__main__':
    main()