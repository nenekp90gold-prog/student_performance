import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi CGPA Mahasiswa", page_icon="🎓", layout="wide")
st.title("🎓 Prediksi CGPA Mahasiswa")
st.write("Masukkan data mahasiswa di bawah ini untuk memprediksi nilai CGPA menggunakan Machine Learning.")

# 2. Load Model yang sudah dilatih
@st.cache_resource
def load_model():
    # Pastikan file final_cgpa_model.pkl ada di folder yang sama
    return joblib.load('final_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}. Pastikan Anda sudah menjalankan script training dan file 'final_cgpa_model.pkl' tersedia.")
    st.stop()

# 3. Form Input Data
# Membagi layar menjadi 3 kolom agar rapi
col1, col2, col3 = st.columns(3)

with st.form("prediction_form"):
    st.subheader("Data Akademik & Profil")
    
    with col1:
        adm_year = st.number_input("University Admission year", min_value=2010, max_value=2024, value=2021)
        age = st.number_input("Age", min_value=16, max_value=50, value=20)
        hsc_year = st.number_input("H.S.C passing year", min_value=2010, max_value=2024, value=2020)
        program = st.selectbox("Program", ["BCSE", "BSE", "BBA", "Other"]) # Sesuaikan dengan data asli
        semester = st.number_input("Current Semester", min_value=1, max_value=14, value=4)
        credits_completed = st.number_input("How many Credit did you have completed?", min_value=0, max_value=150, value=36)
        prev_sgpa = st.number_input("What was your previous SGPA?", min_value=0.0, max_value=4.0, value=3.0, step=0.01)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        scholarship = st.selectbox("Do you have meritorious scholarship ?", ["Yes", "No"])
        transport = st.selectbox("Do you use University transportation?", ["Yes", "No"])
        probation = st.selectbox("Did you ever fall in probation?", ["Yes", "No"])
        suspension = st.selectbox("Did you ever got suspension?", ["Yes", "No"])
        consultancy = st.selectbox("Do you attend in teacher consultancy for academical problems?", ["Yes", "No"])
        relationship = st.selectbox("What is your relationship status?", ["Single", "Relationship"])
        living_with = st.selectbox("With whom you are living with?", ["Family", "Bachelor", "Hostel"])

    with col3:
        study_hours = st.number_input("How many hour do you study daily?", min_value=0.0, max_value=24.0, value=3.0)
        study_freq = st.number_input("How many times do you seat for study in a day?", min_value=0, max_value=10, value=2)
        learning_mode = st.selectbox("What is your preferable learning mode?", ["Offline", "Online", "Blended"])
        english_prof = st.selectbox("Status of your English language proficiency", ["Basic", "Intermediate", "Advance"])
        attendance = st.number_input("Average attendance on class (%)", min_value=0, max_value=100, value=90)
        family_income = st.number_input("What is your monthly family income?", min_value=0, value=25000)
    
    st.subheader("Kebiasaan & Keahlian")
    col4, col5 = st.columns(2)
    
    with col4:
        smartphone = st.selectbox("Do you use smart phone?", ["Yes", "No"])
        pc = st.selectbox("Do you have personal Computer?", ["Yes", "No"])
        social_media = st.number_input("How many hour do you spent daily in social media?", min_value=0.0, max_value=24.0, value=3.0)
        co_curriculum = st.selectbox("Are you engaged with any co-curriculum activities?", ["Yes", "No"])
        health_issues = st.selectbox("Do you have any health issues?", ["Yes", "No"])
        disabilities = st.selectbox("Do you have any physical disabilities?", ["Yes", "No"])

    with col5:
        skills = st.selectbox("What are the skills do you have ?", ["Programming", "Software Development", "Networking", "None"])
        skill_hours = st.number_input("How many hour do you spent daily on your skill development?", min_value=0.0, max_value=24.0, value=2.0)
        interest = st.selectbox("What is you interested area?", ["Data Science", "Software Engineering", "Web development", "Other"])

    # Tombol submit
    submit_button = st.form_submit_button("Prediksi CGPA")

# 4. Logika Prediksi
if submit_button:
    # Mengumpulkan input menjadi format dictionary (NAMA KEYS HARUS SAMA PERSIS DENGAN KOLOM CSV)
    input_data = {
        "University Admission year": adm_year,
        "Gender": gender,
        "Age": age,
        "H.S.C passing year": hsc_year,
        "Program": program,
        "Current Semester": semester,
        "Do you have meritorious scholarship ?": scholarship,
        "Do you use University transportation?": transport,
        "How many hour do you study daily?": study_hours,
        "How many times do you seat for study in a day?": study_freq,
        "What is your preferable learning mode?": learning_mode,
        "Do you use smart phone?": smartphone,
        "Do you have personal Computer?": pc,
        "How many hour do you spent daily in social media?": social_media,
        "Status of your English language proficiency": english_prof,
        "Average attendance on class": attendance,
        "Did you ever fall in probation?": probation,
        "Did you ever got suspension?": suspension,
        "Do you attend in teacher consultancy for any kind of academical problems?": consultancy,
        "What are the skills do you have ?": skills,
        "How many hour do you spent daily on your skill development?": skill_hours,
        "What is you interested area?": interest,
        "What is your relationship status?": relationship,
        "Are you engaged with any co-curriculum activities?": co_curriculum,
        "With whom you are living with?": living_with,
        "Do you have any health issues?": health_issues,
        "What was your previous SGPA?": prev_sgpa,
        "Do you have any physical disabilities?": disabilities,
        "How many Credit did you have completed?": credits_completed,
        "What is your monthly family income?": family_income
    }

    # Ubah ke DataFrame agar bisa dibaca oleh Pipeline Scikit-Learn
    input_df = pd.DataFrame([input_data])

    try:
        # Melakukan prediksi
        prediction = model.predict(input_df)
        
        # Menampilkan hasil
        st.success(f"📈 **Prediksi Current CGPA Anda adalah: {prediction[0]:.2f}**")
        st.balloons()
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
        st.info("Pastikan semua opsi (seperti nama Program atau Skills) sama persis dengan ejaan di dalam dataset asli Anda.")