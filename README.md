# 💳 Segmentasi Nasabah Kartu Kredit — K-Means Clustering

**Tugas Mandiri Pertemuan 4 — Deployment Model**
**51423096 · Nazril Bintang Pratama · Kelas F**

Segmentasi nasabah kartu kredit menggunakan **K-Means Clustering** (metodologi CRISP-DM),
di-deploy sebagai aplikasi **Streamlit** interaktif.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📁 Isi Repo

| Berkas | Keterangan |
|---|---|
| `51423096_Nazril Bintang Pratama_Kelas F.ipynb` | Notebook CRISP-DM lengkap (Business Understanding → Evaluation), dapat dijalankan tanpa error |
| `CC_GENERAL.csv` | Dataset — Credit Card Dataset for Clustering (Kaggle: `arjunbhasin2013/ccdata`) |
| `credit_card_clustered.csv` | Dataset hasil clustering (dengan kolom `Cluster`) |
| `app.py` | Source code aplikasi Streamlit |
| `requirements.txt` | Dependensi Python |
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

## ☁️ Deploy ke Streamlit Community Cloud

1. Buka <https://share.streamlit.io> dan login dengan GitHub.
2. Klik **New app** → pilih repo ini, branch `main`, file `app.py`.
3. Klik **Deploy**. Selesai — dapatkan link publik aplikasi.

## 📈 Hasil Model

| Metrik | Nilai |
|---|---|
| Jumlah cluster (k) | 2 |
| Silhouette Score | 0.549 |
| Davies-Bouldin Index | 1.0596 |
