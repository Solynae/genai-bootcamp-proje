Akbank Sanal Asistan (RAG Chatbot)

Bu proje, GenAI Bootcamp için hazırlanmış bir RAG (Retrieval-Augmented Generation) chatbot uygulamasıdır. Akbank SSS verilerini kullanarak müşterilerin sorularını yanıtlar.



Kullanılan Teknolojiler

Streamlit



LangChain (LCEL)



Google Gemini Pro



ChromaDB (Vektör Veritabanı)



HuggingFace Embeddings



Nasıl Çalıştırılır?

Projeyi klonlayın:



```bash



git clone https://github.com/KULLANICI\_ADINIZ/REPO\_ADINIZ.git
```

(NOT: Yukarıdaki linki GitHub reponuzun linki ile değiştirin)



Gerekli kütüphaneleri yükleyin:



```bash



pip install -r requirements.txt

```

.env dosyası oluşturun ve Google API anahtarınızı ekleyin:



GOOGLE\_API\_KEY="AIzaSy...SizinAnahtarınız..."

Vektör veritabanını oluşturun (ilk çalıştırma için gereklidir):



```bash


python build\_vector\_db.py

```

Streamlit uygulamasını başlatın:



```bash



streamlit run app.py

```


