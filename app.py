
import streamlit as st
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kelulusan",
    page_icon="🎓"
)

# Judul aplikasi
st.title("Sistem Prediksi Kelulusan Mahasiswa")
st.write(
    "Aplikasi ini menggunakan model Machine Learning "
    "untuk memprediksi kelulusan."
)

# Memuat model
@st.cache_resource
def load_model():
    return joblib.load("model_kelulusan.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "File model tidak ditemukan! "
        "Jalankan 1_latih_model.py terlebih dahulu."
    )
    st.stop()

# Form input data mahasiswa
with st.form("form_prediksi"):
    st.subheader("Masukkan Data Mahasiswa")

    ipk = st.number_input(
        "IPK (0.0 - 4.0)",
        min_value=0.0,
        max_value=4.0,
        value=3.0,
        step=0.01
    )

    sks = st.number_input(
        "Total SKS",
        min_value=0,
        max_value=200,
        value=144,
        step=1
    )

    tombol = st.form_submit_button("Prediksi Kelulusan")

# Proses prediksi
if tombol:
    hasil = model.predict([[ipk, sks]])

    st.divider()
    st.subheader("Hasil Prediksi")

    if hasil[0] == 1:
        st.success(
            f"Mahasiswa dengan IPK {ipk} dan SKS {sks} "
            "diprediksi LULUS TEPAT WAKTU."
        )
    else:
        st.error(
            f"Mahasiswa dengan IPK {ipk} dan SKS {sks} "
            "diprediksi TIDAK LULUS."
        )
