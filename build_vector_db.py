# build_vector_db.py
# Amaç: RAG sistemi için veri setini yükler, işler, vektörleştirir ve ChromaDB'ye kaydeder.

import os
from dotenv import load_dotenv
# HuggingFace veri setini yüklemek için
from datasets import load_dataset 
# LangChain Document yapısı
from langchain_core.documents import Document
# Vektör Veritabanı (VDB)
from langchain_community.vectorstores import Chroma
# Embedding Modeli
from langchain_community.embeddings import HuggingFaceEmbeddings

# .env dosyasını yükle (API anahtarını almak için)
load_dotenv()

# --- Vektör Veritabanı ve Embedding Parametreleri ---
DB_PATH = "./chroma_db_banka_sss"
# Türkçe için daha iyi eşleşme sağlayan çok dilli model seçildi
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" 
DATASET_NAME = "akbank/faq-tr (Demo Veri Seti)" 

def load_and_transform_data(dataset_name):
    """
    Hugging Face'ten veri setini yükler veya demo seti kullanır.
    Soruyu ve cevabı page_content içinde birleştirerek retrieval hassasiyetini artırır.
    """
    print(f"1. Veri seti yükleniyor (Kaynak: {dataset_name})...")
    
    # --- Veri Yükleme ve Hata Kontrolü ---
    try:
        # Gerçek Hugging Face veri setini yükleme denemesi
        # NOT: Eğer akbank/faq-tr erişim izni gerektiriyorsa bu kısım başarısız olur.
        dataset = load_dataset("akbank/faq-tr", split="train") 
        print(f"   Yükleme başarılı. Toplam {len(dataset)} kayıt bulundu.")
    except Exception as e:
        # Hata durumunda (erişim/internet sorunu) demo veri seti kullanılır
        print(f"   🔴 Hata: Gerçek veri seti yüklenemedi. Demo veri seti ile devam ediliyor. Detay: {e}")
        
        # Kesin eşleşme için kritik bilgileri içeren demo veri seti
        dataset = [
            {'question': 'Kredi kartı başvurusu nasıl yapılır?', 'answer': 'Akbank müşterisiyseniz Akbank Mobil ve İnternet üzerinden, Akbank müşterisi değilseniz Akbank Mobil\'i indirerek görüntülü görüşme ile hızlıca başvuru yapabilirsiniz. Ayrıca 444 25 25 Müşteri İletişim Merkezi, Axess.com.tr ve tüm şubelerimizden de başvuru yapılabilir.'},
            {'question': 'Döviz hesabı açmak için ne gerekiyor?', 'answer': 'Şubelerimizden veya mobil bankacılık üzerinden, kimlik belgenizle kolayca döviz hesabı açabilirsiniz. Ek bir belgeye gerek yoktur.'},
            {'question': 'Hesap işletim ücreti alıyor musunuz?', 'answer': 'Belirli şartları sağlayan müşterilerimizden hesap işletim ücreti alınmamaktadır. Detaylı bilgi için sözleşmenizi inceleyin.'},
            {'question': 'Şifremi nasıl değiştirebilirim?', 'answer': 'Şifrenizi Akbank Mobil veya Akbank İnternet üzerinden "Şifre İşlemleri" menüsünü kullanarak anında değiştirebilirsiniz.'},
            {'question': 'Akbank mobil ile hangi işlemleri yapabilirim?', 'answer': 'Mobil uygulama ile para transferi, fatura ödemeleri, yatırım işlemleri ve yeni ürün başvuruları dahil birçok bankacılık işlemini şubeye gitmeden gerçekleştirebilirsiniz.'}
        ]
        
    
    # --- LangChain Document Formatına Dönüştürme ---
    documents = []
    for item in dataset:
        # KRİTİK DÜZELTME: Soru ve Cevabı page_content içinde birleştirme (Eşleşme garantisi için)
        combined_content = f"Soru: {item['question']}\nCevap: {item['answer']}"

        doc = Document(
            page_content=combined_content, # Hem sorguyu hem bağlamı vektörle
            metadata={"source_question": item['question'], "source": dataset_name} 
        )
        documents.append(doc)

    print(f"2. {len(documents)} adet Document (Soru+Cevap birleştirilmiş) oluşturuldu.")
    return documents

def build_vector_database(documents: list[Document]):
    """
    Belgeleri vektörleştirir ve ChromaDB'ye kaydeder.
    """
    print(f"3. Embedding modeli ({EMBEDDING_MODEL_NAME}) yükleniyor...")
    
    # Embedding modelini yükle
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': 'cpu'} 
    )

    print(f"4. Belgeler vektörleştiriliyor ve ChromaDB'ye ({DB_PATH}) kaydediliyor...")
    
    # Chroma veritabanı oluşturma ve kalıcı hale getirme
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH 
    )
    vectorstore.persist()
    print("5. ✅ Vektör veritabanı başarıyla oluşturuldu ve diske kaydedildi.")

# --- Ana Akış ---
if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        # Not: Bu kontrol sadece API anahtarının varlığını kontrol eder, zorunlu değildir.
        # build_vector_db.py için sadece embedding'ler ve VDB önemlidir.
        print("🟡 UYARI: GOOGLE_API_KEY ortam değişkeni tanımlı değil. Web uygulaması (app.py) çalışmayabilir.")
    
    # Veritabanı oluşturma adımları
    docs = load_and_transform_data(DATASET_NAME) 
    build_vector_database(docs)