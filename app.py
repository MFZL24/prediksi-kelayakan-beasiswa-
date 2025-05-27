import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Judul Aplikasi
st.title("🎓 Prediksi Kelayakan Beasiswa (Fuzzy Mamdani)")

# Inisialisasi session state untuk menyimpan data
if 'data_siswa' not in st.session_state:
    st.session_state.data_siswa = []

# Input Data Manual
with st.form("form_input"):
    nama = st.text_input("Nama Siswa")
    ipk = st.number_input("IPK (0.0 - 4.0)", min_value=0.0, max_value=4.0, step=0.01, value=3.5)
    aktif = st.selectbox("Aktif Organisasi?", ["Ya", "Tidak"])
    ekonomi = st.selectbox("Kondisi Ekonomi", ["Lemah", "Cukup"])
    submit = st.form_submit_button("🔍 Prediksi")

if submit and nama != "":
    # Konversi input non-numerik ke angka
    aktif_val = 1 if aktif == "Ya" else 0
    ekonomi_val = 0 if ekonomi == "Lemah" else 1

    # Definisi variabel fuzzy
    ipk_fz = ctrl.Antecedent(np.arange(0, 4.1, 0.01), 'ipk')
    aktif_fz = ctrl.Antecedent(np.arange(0, 2, 1), 'aktif')
    ekonomi_fz = ctrl.Antecedent(np.arange(0, 2, 1), 'ekonomi')
    status = ctrl.Consequent(np.arange(0, 101, 1), 'status')

    # Fungsi keanggotaan
    ipk_fz['rendah'] = fuzz.trapmf(ipk_fz.universe, [0, 0, 2.5, 3.0])
    ipk_fz['sedang'] = fuzz.trimf(ipk_fz.universe, [2.5, 3.0, 3.5])
    ipk_fz['tinggi'] = fuzz.trapmf(ipk_fz.universe, [3.0, 3.5, 4.0, 4.0])
    aktif_fz['tidak'] = fuzz.trimf(aktif_fz.universe, [0, 0, 1])
    aktif_fz['ya'] = fuzz.trimf(aktif_fz.universe, [0, 1, 1])
    ekonomi_fz['lemah'] = fuzz.trimf(ekonomi_fz.universe, [0, 0, 1])
    ekonomi_fz['cukup'] = fuzz.trimf(ekonomi_fz.universe, [0, 1, 1])
    status['tidak_layak'] = fuzz.trimf(status.universe, [0, 0, 50])
    status['dipertimbangkan'] = fuzz.trimf(status.universe, [25, 50, 75])
    status['layak'] = fuzz.trimf(status.universe, [50, 100, 100])

    # Aturan fuzzy
    rules = [
        ctrl.Rule(ipk_fz['tinggi'] & aktif_fz['ya'], status['layak']),
        ctrl.Rule(ipk_fz['tinggi'] & aktif_fz['tidak'], status['dipertimbangkan']),
        ctrl.Rule(ipk_fz['sedang'] & ekonomi_fz['lemah'], status['dipertimbangkan']),
        ctrl.Rule(ipk_fz['rendah'], status['tidak_layak']),
        ctrl.Rule(ekonomi_fz['lemah'] & aktif_fz['ya'], status['dipertimbangkan']),
    ]

    # Sistem kontrol fuzzy
    status_ctrl = ctrl.ControlSystem(rules)
    simulasi = ctrl.ControlSystemSimulation(status_ctrl)

    # Masukkan nilai input
    simulasi.input['ipk'] = ipk
    simulasi.input['aktif'] = aktif_val
    simulasi.input['ekonomi'] = ekonomi_val

    # Hitung
    simulasi.compute()
    nilai_output = simulasi.output['status']

    # Interpretasi Output
    if nilai_output >= 70:
        hasil = "Layak"
    elif 40 <= nilai_output < 70:
        hasil = "Dipertimbangkan"
    else:
        hasil = "Tidak Layak"

    # Simpan data ke session_state
    st.session_state.data_siswa.append({
        "Nama Siswa": nama,
        "IPK": ipk,
        "Aktif Organisasi": aktif,
        "Ekonomi": ekonomi,
        "Nilai Fuzzy": round(nilai_output, 2),
        "Status": hasil
    })

# Tampilkan seluruh data yang sudah disimpan
if st.session_state.data_siswa:
    st.subheader("📋 Hasil Prediksi Seluruh Kandidat")
    df = pd.DataFrame(st.session_state.data_siswa)
    st.table(df)
