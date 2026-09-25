"""
Aplikasi Deployment — Segmentasi Nasabah Kartu Kredit (K-Means Clustering)

Tugas Mandiri Pertemuan 4 - Deployment Model
NPM         : 51423096
Nama        : Nazril Bintang Pratama
Kelas       : Kelas F

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py

Dataset:
    Credit Card Dataset for Clustering (Kaggle - arjunbhasin2013/ccdata)
    https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Segmentasi Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ============================================================
# PEMUATAN MODEL (di-cache agar tidak dimuat ulang setiap interaksi)
# ============================================================


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    """Memuat model K-Means, scaler, dan metadata dari folder model/."""
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)

    return kmeans, scaler, metadata


@st.cache_data(show_spinner=False)
def muat_dataset():
    """Memuat dataset hasil clustering untuk keperluan eksplorasi."""
    jalur = os.path.join(BASE_DIR, "credit_card_clustered.csv")
    if os.path.exists(jalur):
        return pd.read_csv(jalur)
    return None


try:
    kmeans, scaler, metadata = muat_model()
except Exception as exc:  # pragma: no cover
    st.error(
        "Model gagal dimuat. Pastikan folder `model/` berisi "
        "`kmeans_model.pkl`, `scaler.pkl`, dan `metadata.json`.\n\n"
        f"Detail error: `{exc}`"
    )
    st.stop()

FITUR = metadata["fitur"]
JUMLAH_CLUSTER = metadata["jumlah_cluster"]
PROFIL = metadata["profil_cluster"]
CENTROID_ASLI = metadata["centroid_asli"]


def fmt_angka(nilai):
    """Format bilangan bulat dengan pemisah ribuan titik (gaya Indonesia)."""
    return f"{int(nilai):,}".replace(",", ".")

# ============================================================
# INTERPRETASI BISNIS SETIAP CLUSTER
# ============================================================
# Ditentukan dari hasil analisis profil cluster pada notebook.
# Cluster diurutkan berdasarkan nilai PURCHASES (tertinggi = paling bernilai).

_urut = sorted(range(JUMLAH_CLUSTER), key=lambda i: PROFIL[str(i)]["PURCHASES"], reverse=True)
LABEL_SEGMEN = {}
_detail_segmen = {}

_presets = [
    {
        "nama": "Nasabah Bernilai Tinggi (High Value)",
        "ikon": "💎",
        "deskripsi": (
            "Nasabah dengan saldo, total pembelian, dan limit kredit yang jauh di atas "
            "rata-rata. Kelompok ini merupakan kontributor pendapatan utama perusahaan."
        ),
        "rekomendasi": [
            "Pertahankan dengan program loyalitas eksklusif (*rewards* / *cashback*).",
            "Tawarkan kenaikan limit kredit sebagai bentuk apresiasi.",
            "Prioritaskan layanan pelanggan premium dan penawaran produk baru lebih awal.",
            "Kirim penawaran *cross-selling* (asuransi, cicilan 0%, kartu tambahan).",
        ],
    },
    {
        "nama": "Nasabah Berkembang (Developing)",
        "ikon": "📈",
        "deskripsi": (
            "Nasabah dengan saldo, total pembelian, dan limit kredit di bawah rata-rata. "
            "Kelompok ini memiliki potensi untuk ditingkatkan nilai transaksinya."
        ),
        "rekomendasi": [
            "Jalankan kampanye aktivasi untuk mendorong frekuensi transaksi.",
            "Berikan promo diskon atau *cashback* pada kategori belanja tertentu.",
            "Tawarkan kenaikan limit bertahap untuk mendorong pembelian lebih besar.",
            "Edukasi manfaat kartu dan lakukan *reminder* berkala agar kartu tetap aktif.",
        ],
    },
]

for pos, idx in enumerate(_urut):
    preset = _presets[pos] if pos < len(_presets) else _presets[-1]
    LABEL_SEGMEN[idx] = f"{preset['ikon']} {preset['nama']}"
    _detail_segmen[idx] = preset

# ============================================================
# SIDEBAR — NAVIGASI
# ============================================================

st.sidebar.title("💳 Segmentasi Nasabah")
st.sidebar.markdown("**K-Means Clustering** — Kartu Kredit")
st.sidebar.markdown("---")

halaman = st.sidebar.radio(
    "Navigasi",
    ["🔍 Prediksi Segmen", "📊 Eksplorasi Data", "ℹ️ Tentang Model"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
**Informasi Model**
- Algoritma: K-Means
- Jumlah cluster: **{JUMLAH_CLUSTER}**
- Fitur: {", ".join(FITUR)}
- Silhouette Score: **{metadata["silhouette_score"]}**
- Davies-Bouldin: **{metadata["davies_bouldin_index"]}**
- Total data: {fmt_angka(metadata["jumlah_data"])} nasabah
"""
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**51423096** · Nazril Bintang Pratama · Kelas F\n\n"
    "Tugas Mandiri Pertemuan 4 — Deployment Model"
)

# ============================================================
# HALAMAN 1 — PREDIKSI SEGMEN
# ============================================================

if halaman == "🔍 Prediksi Segmen":
    st.title("🔍 Prediksi Segmen Nasabah")
    st.markdown(
        "Masukkan data nasabah di bawah ini untuk mengetahui **segmen** nasabah tersebut "
        "beserta **rekomendasi strategi pemasaran** yang sesuai."
    )

    st.markdown("---")

    # Batas slider diambil dari dataset agar input tetap realistis
    dataset = muat_dataset()

    if dataset is not None:
        batas = {f: (0.0, float(np.percentile(dataset[f], 99.5))) for f in FITUR}
        nilai_default = {
            f: float(np.median(dataset[f])) for f in FITUR
        }
    else:
        batas = {f: (0.0, 20000.0) for f in FITUR}
        nilai_default = {f: 1000.0 for f in FITUR}

    kolom_kiri, kolom_kanan = st.columns([1, 1.2])

    with kolom_kiri:
        st.subheader("Data Nasabah")

        nilai_input = {}
        for fitur in FITUR:
            batas_atas = batas[fitur][1]
            nilai_input[fitur] = st.slider(
                label=fitur,
                min_value=0.0,
                max_value=float(round(batas_atas, 2)),
                value=float(round(min(nilai_default[fitur], batas_atas), 2)),
                step=float(round(batas_atas / 200, 2)) or 1.0,
                help=f"Nilai {fitur} nasabah",
            )

        tombol = st.button("🚀 Prediksi Segmen", type="primary", use_container_width=True)

    with kolom_kanan:
        st.subheader("Hasil Prediksi")

        if tombol:
            # DataFrame (bukan np.array) agar nama fitur cocok dengan saat fit.
            vektor = pd.DataFrame([[nilai_input[f] for f in FITUR]], columns=FITUR)

            # Standarisasi WAJIB memakai scaler yang sama dengan saat training
            vektor_scaled = scaler.transform(vektor)

            cluster = int(kmeans.predict(vektor_scaled)[0])

            # Jarak ke setiap centroid (dalam ruang terstandarisasi)
            jarak = np.linalg.norm(
                vektor_scaled - kmeans.cluster_centers_, axis=1
            )

            detail = _detail_segmen.get(cluster, _presets[-1])

            st.success(f"### {LABEL_SEGMEN.get(cluster, f'Cluster {cluster}')}")
            st.markdown(detail["deskripsi"])

            st.markdown("**Rekomendasi Strategi Pemasaran:**")
            for poin in detail["rekomendasi"]:
                st.markdown(f"- {poin}")

            st.markdown("---")

            st.markdown("**Perbandingan dengan rata-rata segmen:**")

            baris = []
            for fitur in FITUR:
                nilai_nasabah = nilai_input[fitur]
                rata_segmen = PROFIL[str(cluster)][fitur]
                selisih = nilai_nasabah - rata_segmen

                baris.append({
                    "Fitur": fitur,
                    "Nilai Nasabah": round(nilai_nasabah, 2),
                    "Rata-rata Segmen": round(rata_segmen, 2),
                    "Selisih": round(selisih, 2),
                    "Posisi": "Di atas rata-rata" if selisih > 0 else "Di bawah rata-rata",
                })

            st.dataframe(
                pd.DataFrame(baris),
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("**Tingkat keyakinan model:**")
            st.caption(
                "Semakin kecil jarak, semakin yakin model terhadap segmen yang dipilih."
            )

            for i in range(JUMLAH_CLUSTER):
                st.progress(
                    float(max(0.0, 1.0 - jarak[i] / (jarak.max() + 1e-9))),
                    text=(
                        f"Jarak ke {LABEL_SEGMEN.get(i, f'Cluster {i}')}: "
                        f"**{jarak[i]:.3f}**"
                    ),
                )

            if len(np.unique(kmeans.labels_)) > 1:
                st.caption(
                    "Catatan: prediksi ini bersifat indikatif. Untuk keputusan bisnis "
                    "yang berskala besar, validasi dengan data historis tetap diperlukan."
                )
        else:
            st.info(
                "👈 Atur nilai nasabah pada panel sebelah kiri, "
                "lalu klik **Prediksi Segmen**."
            )

            st.markdown("**Fitur yang digunakan untuk prediksi:**")
            tabel_fitur = pd.DataFrame({
                "Fitur": FITUR,
                "Deskripsi": [
                    "Saldo yang tersisa di rekening untuk melakukan pembelian",
                    "Total jumlah pembelian yang dilakukan melalui rekening",
                    "Batas kredit yang diberikan kepada nasabah",
                ][: len(FITUR)],
            })
            st.dataframe(tabel_fitur, hide_index=True, use_container_width=True)

# ============================================================
# HALAMAN 2 — EKSPLORASI DATA
# ============================================================

elif halaman == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data & Hasil Clustering")

    dataset = muat_dataset()

    if dataset is None:
        st.warning(
            "Berkas `credit_card_clustered.csv` tidak ditemukan. "
            "Jalankan notebook terlebih dahulu untuk menghasilkannya."
        )
        st.stop()

    # --- Ringkasan ---
    kol1, kol2, kol3, kol4 = st.columns(4)
    kol1.metric("Total Nasabah", fmt_angka(len(dataset)))
    kol2.metric("Jumlah Cluster", JUMLAH_CLUSTER)
    kol3.metric("Silhouette Score", metadata["silhouette_score"])
    kol4.metric("Davies-Bouldin", metadata["davies_bouldin_index"])

    st.markdown("---")

    # --- Distribusi cluster ---
    st.subheader("Distribusi Jumlah Nasabah per Cluster")

    distribusi = dataset["Cluster"].value_counts().sort_index()
    tabel_dist = pd.DataFrame({
        "Cluster": [LABEL_SEGMEN.get(int(i), f"Cluster {i}") for i in distribusi.index],
        "Jumlah Nasabah": distribusi.values,
        "Persentase (%)": (distribusi.values / len(dataset) * 100).round(2),
    })

    kol_a, kol_b = st.columns([1, 1])
    with kol_a:
        st.dataframe(tabel_dist, hide_index=True, use_container_width=True)
    with kol_b:
        st.bar_chart(
            pd.DataFrame(
                {"Jumlah Nasabah": distribusi.values},
                index=[LABEL_SEGMEN.get(int(i), f"Cluster {i}") for i in distribusi.index],
            )
        )

    st.markdown("---")

    # --- Profil cluster ---
    st.subheader("Profil Rata-rata Setiap Cluster")

    kolom_profil = [c for c in PROFIL["0"].keys() if c != "Jumlah Nasabah"]
    profil_tabel = pd.DataFrame(
        {LABEL_SEGMEN.get(int(i), f"Cluster {i}"): PROFIL[str(i)] for i in range(JUMLAH_CLUSTER)}
    ).T

    st.dataframe(profil_tabel, use_container_width=True)

    st.caption(
        "Tabel di atas menunjukkan nilai rata-rata setiap variabel pada masing-masing "
        "cluster. Perbedaan nilai inilah yang menjadi dasar interpretasi bisnis."
    )

    st.markdown("---")

    # --- Pilih fitur untuk scatter ---
    st.subheader("Sebaran Data Berdasarkan Fitur")

    kolom_numerik = [c for c in FITUR if c in dataset.columns]

    kol_x, kol_y = st.columns(2)
    with kol_x:
        sumbu_x = st.selectbox("Sumbu X", kolom_numerik, index=0)
    with kol_y:
        sumbu_y = st.selectbox(
            "Sumbu Y",
            kolom_numerik,
            index=min(1, len(kolom_numerik) - 1),
        )

    data_plot = dataset[[sumbu_x, sumbu_y, "Cluster"]].copy()
    data_plot["Segmen"] = data_plot["Cluster"].map(
        lambda c: LABEL_SEGMEN.get(int(c), f"Cluster {c}")
    )

    st.scatter_chart(
        data_plot,
        x=sumbu_x,
        y=sumbu_y,
        color="Segmen",
    )

    st.markdown("---")

    # --- Centroid ---
    st.subheader("Titik Pusat (Centroid) Setiap Cluster")

    centroid_df = pd.DataFrame(
        CENTROID_ASLI,
        columns=FITUR,
        index=[LABEL_SEGMEN.get(i, f"Cluster {i}") for i in range(JUMLAH_CLUSTER)],
    )
    st.dataframe(centroid_df.round(2), use_container_width=True)

    st.caption(
        "Centroid adalah titik pusat cluster. Setiap nasabah dimasukkan ke cluster "
        "dengan centroid terdekat."
    )

# ============================================================
# HALAMAN 3 — TENTANG MODEL
# ============================================================

else:
    st.title("ℹ️ Tentang Model & Dataset")

    st.subheader("Dataset")
    st.markdown(
        f"""
**Nama:** {metadata["dataset"]}

**Sumber:** {metadata["sumber"]}

**Deskripsi:** Dataset ini merangkum perilaku penggunaan sekitar 9.000 nasabah
kartu kredit aktif selama 6 bulan terakhir. Data berada pada level nasabah
dengan 18 variabel perilaku.

**Karakteristik:**
- Bentuk data: tabular (CSV)
- Jumlah baris: {fmt_angka(metadata["jumlah_data"])} nasabah (setelah pembersihan)
- Jumlah kolom: 18 kolom
- Fitur numerik: 17 kolom
"""
    )

    st.markdown("---")

    st.subheader("Metodologi — CRISP-DM")
    st.markdown(
        """
| Fase | Kegiatan |
|---|---|
| 1. Business Understanding | Merumuskan kebutuhan segmentasi nasabah untuk strategi pemasaran |
| 2. Data Understanding | Eksplorasi struktur, kualitas, dan karakteristik data |
| 3. Data Preparation | Penanganan *missing value*, pemilihan fitur, standarisasi |
| 4. Modeling | K-Means, Elbow Method, Silhouette Score, visualisasi PCA |
| 5. Evaluation | Silhouette Score, Davies-Bouldin Index, interpretasi profil cluster |
| 6. Deployment | Aplikasi Streamlit ini |
"""
    )

    st.markdown("---")

    st.subheader("Algoritma: K-Means Clustering")

    st.markdown(
        f"""
**K-Means** mengelompokkan data ke dalam **{JUMLAH_CLUSTER} cluster** dengan cara:
1. Menempatkan *centroid* awal secara acak.
2. Menugaskan setiap data ke *centroid* terdekat.
3. Memperbarui posisi *centroid* berdasarkan rata-rata anggotanya.
4. Mengulangi langkah 2–3 hingga posisi *centroid* stabil.

**Mengapa standarisasi diperlukan?**
K-Means memakai jarak Euclidean. Tanpa standarisasi, fitur dengan rentang nilai besar
akan mendominasi perhitungan jarak. `StandardScaler` menyetarakan skala semua fitur
(rata-rata 0, simpangan baku 1).
"""
    )

    st.markdown("---")

    st.subheader("Hasil Evaluasi")

    kol1, kol2 = st.columns(2)
    with kol1:
        st.metric(
            "Silhouette Score",
            metadata["silhouette_score"],
            help="Rentang −1 hingga 1. Semakin tinggi semakin baik.",
        )
    with kol2:
        st.metric(
            "Davies-Bouldin Index",
            metadata["davies_bouldin_index"],
            help="Nilai ≥ 0. Semakin rendah semakin baik.",
        )

    st.markdown(
        f"""
**Interpretasi:**

- **Silhouette Score {metadata["silhouette_score"]}** menunjukkan kualitas pemisahan
  cluster yang tergolong **baik**. Nilai ini jauh di atas ambang 0,2 dan mendekati 1,
  artinya data dalam satu cluster cukup kompak dan terpisah jelas dari cluster lain.
- **Davies-Bouldin Index {metadata["davies_bouldin_index"]}** tergolong **rendah**,
  menandakan cluster rapat di dalam dan berjauhan satu sama lain.

Kedua metrik ini **saling mendukung** dan menunjukkan bahwa struktur cluster yang
terbentuk konsisten.
"""
    )

    st.markdown("---")

    st.subheader("Keterbatasan")

    st.markdown(
        """
- Hanya **3 fitur** yang digunakan sesuai ketentuan tugas. Penambahan fitur lain
  (misalnya `CASH_ADVANCE`, `PURCHASES_TRX`) berpotensi menghasilkan segmentasi
  yang lebih kaya.
- Sebaran data sangat *skewed*. Transformasi logaritmik sebelum standarisasi dapat
  meningkatkan kualitas pemisahan cluster.
- K-Means mengasumsikan cluster berbentuk bulat dengan ukuran serupa. Algoritma
  **DBSCAN** atau **Gaussian Mixture Model** dapat dijadikan pembanding.
"""
    )

st.markdown("---")
st.caption(
    "**51423096** · Nazril Bintang Pratama · Kelas F · "
    "Tugas Mandiri Pertemuan 4 — Deployment Model"
)
