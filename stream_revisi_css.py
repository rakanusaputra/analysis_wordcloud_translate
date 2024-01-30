# Import library Streamlit
import streamlit as st
from wordcloud import WordCloud  # Mengimport WordCloud
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from googletrans import Translator
from googletrans import LANGUAGES

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    # Menambahkan judul untuk aplikasi
    st.title("WOLF PROJECT")
    st.header("1. Analisis Sentimen + WordCloud")
    # Menambahkan input untuk mengunggah file teks
    file = st.file_uploader("Unggah file teks", type=["txt"])

    # Memproses file yang diunggah (jika ada)
    if file is not None:
        # Membaca isi file
        teks = file.read().decode("utf-8")
        # Menampilkan teks yang diunggah
        st.subheader("Teks yang Diunggah:")
        st.write(teks)

        # Load model dan tokenizer
        nama_model = "indobenchmark/indobert-base-p1"
        model = AutoModelForSequenceClassification.from_pretrained(nama_model)
        tokenizer = AutoTokenizer.from_pretrained(nama_model)

        # Analisis sentimen
        kelas_terprediksi, kepercayaan = analisis_sentimen(teks, model, tokenizer)

        # Tampilkan hasil analisis sentimen
        label_sentimen = "Positif" if kelas_terprediksi == 1 else "Negatif"
        st.subheader("Hasil Analisis Sentimen:")
        st.write(f"Sentimen: {label_sentimen} dengan tingkat kepercayaan: {kepercayaan}")

        # Membuat dan menampilkan WordCloud
        tag_cloud(teks)  # Memanggil fungsi tag_cloud
        
    st.header("2. Translator")

    #pilih bahasa
    source_lang = st.selectbox("Pilih bahasa sumber", list(LANGUAGES.values()))
    target_lang = st.selectbox("Pilih bahasa tujuan", list(LANGUAGES.values()))
    
    # Menambahkan input teks dan pilihan bahasa sumber
    source_text = st.text_area("Masukkan Teks", "")


    # Memproses teks yang dimasukkan (jika ada)
    if source_text:
        # Terjemahkan teks ke bahasa tujuan yang dipilih
        translated_text = translate_text(source_text, src_lang=source_lang, dest_lang=target_lang)

        # Tampilkan hasil terjemahan
        st.header("Hasil Terjemahan:")
        st.write(translated_text)

# Fungsi untuk melakukan analisis sentimen
def analisis_sentimen(teks, model, tokenizer):
    # Tokenisasi teks
    tokens = tokenizer(teks, return_tensors='pt', truncation=True, padding=True)

    # Prediksi sentimen menggunakan model
    with torch.no_grad():
        hasil = model(**tokens)

    # Hitung probabilitas distribusi softmax
    probabilitas = softmax(hasil.logits, dim=1)

    # Ambil label dengan probabilitas tertinggi
    kelas_terprediksi = torch.argmax(probabilitas, dim=1).item()

    return kelas_terprediksi, probabilitas[0][kelas_terprediksi].item()

# Fungsi untuk membuat dan menampilkan WordCloud
def tag_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the generated word cloud using matplotlib
    st.subheader("WordCloud")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Fungsi untuk melakukan terjemahan teks
def translate_text(text, src_lang='id', dest_lang='en'):
    translator = Translator()
    translated = translator.translate(text, src=src_lang, dest=dest_lang)
    return translated.text

if __name__ == "__main__":
    main()
