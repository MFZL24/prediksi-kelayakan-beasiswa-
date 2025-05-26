import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Judul Aplikasi
st.title("🎓 Prediksi Kelayakan Beasiswa (Fuzzy Mamdani)")

# Input Data Manual
with st.form("form_input"):
    nama = st.text_input("Nama Siswa")
    ipk = st.number_input("IPK (0.0 - 4.0)", min_value=0.0, max_value=4.0, step=0.01, value=3.5)
    aktif = st.selectbox("Aktif Organisasi?", ["Ya", "Tidak"])
    ekonomi = st.selectbox("Kondisi Ekonomi", ["Lemah", "Cukup"])
    submit = st.form_submit_button("🔍 Prediksi")

if submit:
    # Konversi input non-numerik ke angka
    aktif_val = 1 if aktif == "Ya" else 0
    ekonomi_val = 0 if ekonomi == "Lemah" else 1

    # Definisi variabel fuzzy
    ipk_fz = ctrl.Antecedent(np.arange(0, 4.1, 0.01), 'ipk')
    aktif_fz = ctrl.Antecedent(np.arange(0, 2, 1), 'aktif')
    ekonomi_fz = ctrl.Antecedent(np.arange(0, 2, 1), 'ekonomi')
    status = ctrl.Consequent(np.arange(0, 101, 1), 'status')

    # Fungsi keanggotaan IPK
    ipk_fz['rendah'] = fuzz.trapmf(ipk_fz.universe, [0, 0, 2.5, 3.0])
    ipk_fz['sedang'] = fuzz.trimf(ipk_fz.universe, [2.5, 3.0, 3.5])
    ipk_fz['tinggi'] = fuzz.trapmf(ipk_fz.universe, [3.0, 3.5, 4.0, 4.0])

    # Fungsi keanggotaan Aktif Organisasi
    aktif_fz['tidak'] = fuzz.trimf(aktif_fz.universe, [0, 0, 1])
    aktif_fz['ya'] = fuzz.trimf(aktif_fz.universe, [0, 1, 1])

    # Fungsi keanggotaan Ekonomi
    ekonomi_fz['lemah'] = fuzz.trimf(ekonomi_fz.universe, [0, 0, 1])
    ekonomi_fz['cukup'] = fuzz.trimf(ekonomi_fz.universe, [0, 1, 1])

    # Output status
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

    # Sistem kontrol
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

    # Tampilkan dalam tabel
    df = pd.DataFrame([{
        "Nama Siswa": nama,
        "IPK": ipk,
        "Aktif Organisasi": aktif,
        "Ekonomi": ekonomi,
        "Nilai Fuzzy": round(nilai_output, 2),
        "Status": hasil
    }])

    st.subheader("📋 Hasil Prediksi")
    st.table(df)
