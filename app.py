# app.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# .env dosyasını yükle
load_dotenv()

# Ayarlar
DB_PATH = "./chroma_db_banka_sss"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# LLM_MODEL'ı gemini-pro yerine, gemini-2.5-flash veya gemini-pro olarak kullanın.
LLM_MODEL = "gemini-2.5-flash" 

@st.cache_resource
def load_rag_components():
    """RAG bileşenlerini yükler"""
    
    # ⚠️ EK KONTROL: Veritabanı varlığını kontrol et
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Hata: Vektör veritabanı klasörü ({DB_PATH}) bulunamadı.")
        st.error("Lütfen önce 'python build_vector_db.py' komutunu çalıştırarak veritabanını oluşturun.")
        st.stop()
        
    st.info("🧠 RAG bileşenleri yükleniyor...")
    
    # Embedding modeli
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': 'cpu'}
    )
    
    # Vektör veritabanı
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # LLM Modeli Adı GÜNCELLENDİ (Hata veren ad yerine stabil ad kullanıldı)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    
    st.success("✅ RAG Bileşenleri Başarıyla Yüklendi.")
    return retriever, llm

# Prompt template (Aynı kaldı)
template = """Sen, bir bankanın müşteri hizmetleri temsilcisisin. Sadece aşağıda verilen 'Bağlam' içindeki bilgilere dayanarak soruyu yanıtla. Bağlamda cevabı bulunmayan bir soru sorulursa, "Bu konuda bir bilgim bulunmuyor, lütfen bankanızla doğrudan iletişime geçin." de.

Bağlam:
{context}

Kullanıcı Sorusu: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def create_rag_chain(retriever, llm):
    """RAG zinciri oluşturur"""
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Streamlit uygulaması
st.set_page_config(page_title="Akbank Sanal Asistan", layout="wide")
st.title("🏦 Akbank Sanal Asistan (RAG Chatbot)")
st.caption("GenAI Bootcamp Projesi | Finansal SSS yanıtlayıcısı")

# RAG bileşenlerini yükle
retriever, llm = load_rag_components()
rag_chain = create_rag_chain(retriever, llm)

# Sohbet geçmişi (Aynı kaldı)
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Merhaba! Ben Akbank Sanal Asistan. Size finans ve bankacılık konularında nasıl yardımcı olabilirim?"
    })

# Önceki mesajları göster (Aynı kaldı)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi (Aynı kaldı)
if prompt := st.chat_input("Lütfen sorunuzu buraya yazın..."):
    # Kullanıcı mesajını kaydet ve göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chatbot yanıtını üret
    with st.chat_message("assistant"):
        with st.spinner("Cevap aranıyor..."):
            try:
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                # Hata durumunda kullanıcıya bilgi ver
                error_msg = f"Üzgünüm, Gemini API'den yanıt alınırken bir hata oluştu. Hata detayları: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar (Aynı kaldı)
st.sidebar.title("Product Kılavuzu")
st.sidebar.markdown("""
**Test Senaryoları:**
- **Kredi kartı başvurusu nasıl yapılır?**
- **Döviz hesabı açmak için ne gerekiyor?**
- **Hesap işletim ücreti alıyor musunuz?**
- **Dünyanın en yüksek dağı nedir?** (bilgim yok testi)
""")