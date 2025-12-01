# Face Recognition System - Tubes Deep Learning

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

Aplikasi Web Pengenalan Wajah (Face Recognition) yang dibangun menggunakan pendekatan *Hybrid Deep Learning* (CNN + Machine Learning) dan di-deploy menggunakan Google Cloud Platform.

## 👥 Anggota Tim

| Nama | NIM |
| :--- | :--- |
| **Muhammad Riveldo Hermawan Putra** | 122140037 |
| **Joshia Fernandes Sectio Purba** | 122140170 |
| **Randy Hendriyawan** | 122140171 |

---

## 📖 Tentang Proyek

Proyek ini bertujuan untuk mengklasifikasikan wajah ke dalam **70 kelas (orang)** yang berbeda. Tantangan utama dalam proyek ini adalah keterbatasan jumlah data per kelas (hanya sekitar 4 gambar per orang), yang menyebabkan model *Deep Learning* standar (seperti ResNet atau ViT yang dilatih dari awal/finetuning) mengalami *overfitting* yang parah.

### Solusi: Transfer Learning & Embedding
Untuk mengatasi masalah data yang sedikit (*Few-Shot Learning*), kami menggunakan pendekatan:

1.  **Ekstraksi Fitur (Feature Extraction):** Menggunakan **InceptionResnetV1 (FaceNet)** yang telah dilatih sebelumnya (*pre-trained*) pada dataset VGGFace2. Model ini bertugas mengubah gambar wajah menjadi vektor angka (embedding 512-dimensi).
2.  **Klasifikasi (Classification):** Menggunakan **Support Vector Machine (SVM)**. SVM dipilih karena kemampuannya yang sangat baik dalam menemukan *hyperplane* pemisah yang optimal pada dataset kecil dengan dimensi tinggi.

Hasilnya, sistem ini mampu mencapai akurasi dan tingkat keyakinan (*confidence*) yang tinggi meskipun dengan data latih yang minimal.

---

## 🛠️ Tech Stack

* **Bahasa Pemrograman:** Python 3.9
* **Framework ML/DL:** PyTorch, Torchvision, Facenet-PyTorch, Scikit-Learn
* **Image Processing:** OpenCV, Pillow
* **Web Framework:** Streamlit
* **Deployment:** Docker, Google Cloud Run (Serverless)

---

## 📂 Struktur File

TubesDL/ ├── app.py # Main aplikasi Streamlit ├── requirements.txt # Daftar dependensi library ├── Dockerfile # Konfigurasi container untuk deployment ├── svm_face_model.pkl # (Opsional) Model SVM terpisah ├── facenet_weights.pth # (Opsional) Bobot FaceNet terpisah ├── face_recognition_complete.pth # Model gabungan (FaceNet + SVM + Labels) └── README.md # Dokumentasi proyek


---

## 🚀 Cara Menjalankan (Local)

### Prasyarat
Pastikan Anda sudah menginstall Python dan Git. Karena file model berukuran besar (>100MB), pastikan Anda telah menginstall **Git LFS**.

### Langkah-langkah

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/Randyh-25/TubesDL.git](https://github.com/Randyh-25/TubesDL.git)
    cd TubesDL
    ```

2.  **Pull File Model (LFS)**
    ```bash
    git lfs pull
    ```

3.  **Install Dependensi**
    Disarankan menggunakan virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```
    Aplikasi akan berjalan di `http://localhost:8501`.

---

## ☁️ Deployment (Google Cloud Platform)

Aplikasi ini dirancang untuk dapat berjalan di lingkungan *serverless* seperti Google Cloud Run.

### Langkah Deployment via Cloud Shell

1.  **Build Docker Image**
    ```bash
    gcloud builds submit --tag gcr.io/[PROJECT_ID]/face-app
    ```

2.  **Deploy ke Cloud Run**
    ```bash
    gcloud run deploy face-app \
      --image gcr.io/[PROJECT_ID]/face-app \
      --platform managed \
      --region asia-southeast2 \
      --allow-unauthenticated \
      --memory 2Gi
    ```

---

## 📸 Preview Aplikasi

*(Anda bisa menambahkan screenshot aplikasi yang berjalan di sini)*

1.  **Upload Foto:** Pengguna mengunggah foto wajah.
2.  **Deteksi:** Sistem mendeteksi lokasi wajah menggunakan Haar Cascade.
3.  **Prediksi:** Sistem menampilkan nama pemilik wajah beserta tingkat akurasinya.

---

**Tugas Besar Deep Learning**
*Institut Teknologi Sumatera (ITERA)*
