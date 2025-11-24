# app.py

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datasets import load_dataset # Veri oluşturma için gerekli
from langchain_core.documents import Document # Veri oluşturma için gerekli

# .env dosyasını yükle
load_dotenv()

# Ayarlar
DB_PATH = "./chroma_db_banka_sss"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "gemini-2.5-flash"

# --- KRİTİK DEĞİŞİKLİK: Vektör Veritabanı Oluşturma Mantığı ---
@st.cache_resource
def build_vector_database_on_demand(db_path, model_name):
    """
    Vektör veritabanı mevcut değilse, build_vector_db.py'deki mantığı kullanarak oluşturur.
    Streamlit Cloud'da veritabanının kaybolması durumunu ele almak için gereklidir.
    """
    # Veritabanı mevcutsa ve sağlam görünüyorsa tekrar oluşturma
    if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "chroma-collections.parquet")):
        # st.info("✅ Vektör veritabanı zaten mevcut. Tekrar oluşturulmuyor.")
        return

    st.info("🚨 Vektör veritabanı bulunamadı. Otomatik olarak oluşturuluyor...")
    
    # Veri Yükleme (build_vector_db.py'den kopyalanmıştır - sadece demo veri)
    dataset = [
        {'question': 'Kredi kartı başvurusu nasıl yapılır?', 'answer': 'Akbank müşterisiyseniz Akbank Mobil ve İnternet üzerinden, Akbank müşterisi değilseniz Akbank Mobil\'i indirerek görüntülü görüşme ile hızlıca başvuru yapabilirsiniz. Ayrıca 444 25 25 Müşteri İletişim Merkezi, Axess.com.tr ve tüm şubelerimizden de başvuru yapılabilir.'},
        {'question': 'Döviz hesabı açmak için ne gerekiyor?', 'answer': 'Şubelerimizden veya mobil bankacılık üzerinden, kimlik belgenizle kolayca döviz hesabı açabilirsiniz. Ek bir belgeye gerek yoktur.'},
        {'question': 'Hesap işletim ücreti alıyor musunuz?', 'answer': 'Belirli şartları sağlayan müşterilerimizden hesap işletim ücreti alınmamaktadır. Detaylı bilgi için sözleşmenizi inceleyin.'},
        {'question': 'Şifremi nasıl değiştirebilirim?', 'answer': 'Şifrenizi Akbank Mobil veya Akbank İnternet üzerinden "Şifre İşlemleri" menüsünü kullanarak anında değiştirebilirsiniz.'},
        {'question': 'Akbank mobil ile hangi işlemleri yapabilirim?', 'answer': 'Mobil uygulama ile para transferi, fatura ödemeleri, yatırım işlemleri ve yeni ürün başvuruları dahil birçok bankacılık işlemini şubeye gitmeden gerçekleştirebilirsiniz.'}
    ]
    
    documents = []
    for item in dataset:
        combined_content = f"Soru: {item['question']}\nCevap: {item['answer']}"
        doc = Document(
            page_content=combined_content,
            metadata={"source_question": item['question'], "source": "Demo Veri Seti"}
        )
        documents.append(doc)

    # Embedding modelini yükle
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

    # Chroma veritabanı oluşturma ve kalıcı hale getirme
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    vectorstore.persist()
    st.success("✅ Vektör veritabanı başarıyla oluşturuldu.")
    return

# --- load_rag_components fonksiyonu ---
@st.cache_resource
def load_rag_components():
    """RAG bileşenlerini yükler"""
    
    # ⚠️ 1. Kontrol: Veritabanını oluştur veya varlığını kontrol et
    # Bu fonksiyon, Streamlit Cloud'da veritabanının bulunamaması sorununu çözer.
    build_vector_database_on_demand(DB_PATH, EMBEDDING_MODEL_NAME)
    
    st.info("🧠 RAG bileşenleri yükleniyor...")
    
    # Embedding modeli
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # Vektör veritabanı
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # LLM Modeli
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
                error_msg = f"Üzgünüm, Gemini API'den yanıt alınırken bir hata oluştu. Lütfen GOOGLE_API_KEY'inizin doğru olduğundan emin olun. Hata: {e}"
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





