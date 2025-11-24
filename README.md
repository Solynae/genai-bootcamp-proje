# Akbank Sanal Asistan (RAG Chatbot)

Bu proje, **GenAI Bootcamp** için geliştirilmiş bir **RAG (Retrieval-Augmented Generation) chatbot** uygulamasıdır. Akbank SSS verilerini kullanarak kullanıcıların sorularına bağlamsal yanıtlar üretir.

---

## Özellikler

* LangChain ile RAG zinciri yönetimi
* ChromaDB vektör tabanlı arama
* Google Gemini Pro ile LLM entegrasyonu
* Streamlit web arayüzü
* Türkçe ↔ İngilizce bağlamsal cevap desteği

---

## Canlı Demo

[Streamlit Uygulamasını Deneyin](https://genai-bootcamp-proje-bpcjko7jafwyhydxcsvbzy.streamlit.app/)

---

## Kurulum

1. Python 3.10+ yüklü olduğundan emin olun
2. Reposu klonlayın:

```bash
git clone https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git
cd REPO_ADINIZ
```

3. Sanal ortam oluşturun ve aktifleştirin:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

4. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

5. `.env` dosyası oluşturun ve Google API anahtarınızı ekleyin:

```env
GOOGLE_API_KEY="AIzaSy...SizinAnahtarınız..."
```

6. Vektör veritabanını oluşturun:

```bash
python build_vector_db.py
```

7. Uygulamayı başlatın:

```bash
streamlit run app.py
```

---

## Kullanım

* Chatbot, kullanıcı sorularına bağlamsal cevaplar verir.
* Örnek sorular:

  * Kredi kartı başvurusu nasıl yapılır?
  * Döviz hesabı açmak için ne gerekiyor?
  * Hesap işletim ücreti alıyor musunuz?

---

## Deploy

Bu proje Streamlit Cloud üzerinde canlı olarak çalışmaktadır:
[Canlı Demo](https://genai-bootcamp-proje-bpcjko7jafwyhydxcsvbzy.streamlit.app/)

---

## Lisans

Bu proje [MIT Lisansı](https://opensource.org/licenses/MIT) ile lisanslanmıştır.
