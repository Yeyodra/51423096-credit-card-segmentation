# 💳 Segmentasi Nasabah Kartu Kredit — K-Means Clustering

**Tugas Mandiri Pertemuan 4 — Deployment Model**
**51423096 · Nazril Bintang Pratama · Kelas F**

Segmentasi nasabah kartu kredit menggunakan **K-Means Clustering** (metodologi CRISP-DM),
di-deploy sebagai aplikasi **Streamlit** interaktif.

## 🔗 Link

| | |
|---|---|
| **Aplikasi (Live)** | **https://tugas.novela.biz.id** |
| **Source Code** | https://github.com/Yeyodra/51423096-credit-card-segmentation |

## 📁 Isi Folder

| Berkas | Keterangan |
|---|---|
| `51423096_Nazril Bintang Pratama_Kelas F.ipynb` | Notebook CRISP-DM lengkap (Business Understanding → Evaluation), dapat dijalankan tanpa error |
| `CC_GENERAL.csv` | Dataset — Credit Card Dataset for Clustering (Kaggle: `arjunbhasin2013/ccdata`) |
| `credit_card_clustered.csv` | Dataset hasil clustering (dengan kolom `Cluster`) |
| `app.py` | Source code aplikasi Streamlit |
| `requirements.txt` | Dependensi Python |
| `Dockerfile` | Konfigurasi container (opsional, untuk deployment) |
| `model/kmeans_model.pkl` | Model K-Means terlatih |
| `model/scaler.pkl` | StandardScaler (wajib dipakai untuk input baru) |
| `model/metadata.json` | Metadata model (fitur, jumlah cluster, metrik evaluasi) |

## 📊 Dataset

- **Sumber:** Kaggle — Credit Card Dataset for Clustering
  (<https://www.kaggle.com/datasets/arjunbhasin2013/ccdata>)
- **Ukuran:** 8.950 baris × 18 kolom (17 fitur numerik)
- **Fitur training:** `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`

## 🚀 Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`.

## 📈 Hasil Model

| Metrik | Nilai |
|---|---|
| Jumlah cluster (k) | 2 |
| Silhouette Score | 0.549 |
| Davies-Bouldin Index | 1.0596 |

## 🧭 Metodologi (CRISP-DM)

1. **Business Understanding** — segmentasi nasabah untuk strategi pemasaran
2. **Data Understanding** — eksplorasi struktur, kualitas, dan karakteristik data
3. **Data Preparation** — penanganan *missing value*, pemilihan fitur, standarisasi
4. **Modeling** — K-Means, Elbow Method, Silhouette Score, visualisasi PCA
5. **Evaluation** — Silhouette Score, Davies-Bouldin Index, interpretasi profil cluster
6. **Deployment** — aplikasi Streamlit (lihat `app.py`)
