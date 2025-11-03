import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
# Tambahkan import lain yang Anda gunakan (misalnya skimage, scipy, dll.)
# from skimage.metrics import structural_similarity as ssim
# import scipy.linalg

# --- 1. KONFIGURASI DAN DATA STATIS ---

st.set_page_config(
    page_title="Virtual Try-On Sepatu",
    layout="wide"
)

# Inisialisasi state untuk melacak sepatu yang dipilih
if "selected_shoe" not in st.session_state:
    st.session_state.selected_shoe = None

# Detail 4 Sepatu (DIPERBAIKI: Pastikan path gambar statis menggunakan huruf kecil 'static/')
SHOES = {
    "shoe_1": {"name": "DUNK LOW SEAN CLIVER", "price": "Rp1.150.000", "image_path": "static/shoe_1.jpg"},
    "shoe_2": {"name": "DUNK LOW BLEACHED AQUA", "price": "Rp7.200.000", "image_path": "static/shoe_2.jpg"},
    "shoe_3": {"name": "AIR MAX 1 PATTA WAVES", "price": "Rp3.600.000", "image_path": "static/shoe_3.jpg"},
    "shoe_4": {"name": "DUNK LOW GREY FOG", "price": "Rp1.600.000", "image_path": "static/shoe_4.jpg"},
}

# Daftar Gambar Kaki Sampel (DIPERBAIKI: Pastikan path menggunakan huruf kecil 'static/')
FOOT_SAMPLES = {
    "sample_1": "static/foot_sample_1.jpg",
    "sample_2": "static/foot_sample_2.jpg",
}

# Ukuran input yang diharapkan oleh model pix2pix
IMG_SIZE = 256
MODEL_PATH = 'models/pix2pix_tryon_G.h5'

# --- 2. FUNGSI PEMUATAN MODEL DENGAN CACHE (PENTING) ---

# Menggunakan @st.cache_resource agar model HANYA dimuat sekali saat startup
@st.cache_resource
def load_generator_model(path):
    st.info(f"Memuat model generator dari: {path}. Tunggu sebentar...")
    try:
        # Pemuatan model Keras/TensorFlow
        model = tf.keras.models.load_model(path)
        st.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop() # Hentikan eksekusi jika gagal memuat model
        
# Panggil fungsi ini untuk mendapatkan model yang sudah di-cache
generator = load_generator_model(MODEL_PATH)

# --- 3. FUNGSI PRAPROSES DAN TRY-ON (Anda harus memastikan fungsi ini benar) ---

def preprocess_image(image_path, size):
    # Pastikan file gambar ada sebelum dibuka
    if not os.path.exists(image_path):
        st.error(f"File gambar tidak ditemukan di path: {image_path}")
        st.stop()
        
    img = Image.open(image_path).convert('RGB').resize((size, size))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)

def perform_try_on(shoe_path, foot_path, generator_model):
    shoe_input = preprocess_image(shoe_path, IMG_SIZE)
    foot_input = preprocess_image(foot_path, IMG_SIZE)

    # Gabungkan input (sesuaikan ini dengan input model Anda: 2 input atau 1 input gabungan)
    # Ini adalah contoh untuk 1 input gabungan (channel terakhir)
    combined_input = np.concatenate([shoe_input, foot_input], axis=-1) 

    # Prediksi
    prediction = generator_model.predict(combined_input)
    
    # Denormalisasi output ke range [0, 255]
    result_array = (prediction[0] + 1.0) * 127.5
    result_image = Image.fromarray(result_array.astype(np.uint8))
    
    return result_image

# --- 4. TAMPILAN HALAMAN UTAMA/MODEL ---

def display_catalog_page():
    st.header("👟 SNEAKERSKU: Virtual Try-On")
    st.subheader("New Arrival")

    cols = st.columns(len(SHOES))

    for i, (shoe_id, data) in enumerate(SHOES.items()):
        with cols[i]:
            # BARIS KRITIS (sekarang harusnya bekerja jika path benar)
            st.image(data['image_path'], use_column_width="auto")
            st.subheader(data['name'])
            
            # Tampilkan harga
            st.markdown(f"**Harga:** <del>Rp9.999.000</del> | **{data['price']}**", unsafe_allow_html=True)
            
            # Tombol pilih sepatu
            if st.button("Pilih Sepatu", key=shoe_id):
                st.session_state.selected_shoe = data
                st.rerun() 

# --- 5. TAMPILAN HALAMAN TRY-ON ---

def display_try_on_page():
    st.header(f"✨ Virtual Try-On: {st.session_state.selected_shoe['name']}")
    
    # Pilihan gambar kaki sampel
    foot_sample_choice = st.radio(
        "Pilih Gambar Kaki Sampel:",
        list(FOOT_SAMPLES.keys()),
        format_func=lambda x: f"Kaki Sampel {x.split('_')[-1]}",
        horizontal=True
    )
    
    selected_foot_path = FOOT_SAMPLES[foot_sample_choice]
    
    # Tampilkan input dan hasil
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Sepatu")
        st.image(st.session_state.selected_shoe['image_path'], caption=st.session_state.selected_shoe['name'], use_column_width=True)
    
    with col2:
        st.subheader("2. Kaki Sampel")
        st.image(selected_foot_path, caption=foot_sample_choice, use_column_width=True)

    st.markdown("---")
    
    st.subheader("Hasil Virtual Try-On")
    
    # Lakukan Try-On dan tampilkan hasil
    if st.button("Lakukan Try-On", type="primary"):
        with st.spinner("Melakukan prediksi model..."):
            try:
                result_img = perform_try_on(
                    shoe_path=st.session_state.selected_shoe['image_path'],
                    foot_path=selected_foot_path,
                    generator_model=generator
                )
                
                # Tampilkan gambar hasil
                st.image(result_img, caption="Hasil Try-On", use_column_width=True)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan Try-On. Cek log Streamlit Cloud. Error: {e}")

    st.markdown("---")
    if st.button("Kembali ke Katalog"):
        st.session_state.selected_shoe = None
        st.rerun()

# --- 6. NAVIGASI UTAMA ---

if st.session_state.selected_shoe is None:
    display_catalog_page()
else:
    display_try_on_page()