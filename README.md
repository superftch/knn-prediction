# 🎓 Jurusan Predictor

Aplikasi prediksi jurusan kuliah menggunakan algoritma K-Nearest Neighbors (KNN) dengan integrasi AI Agent (OpenAI GPT & Google Gemini) untuk memberikan rekomendasi jurusan yang tepat berdasarkan nilai rapor dan hasil psikotes siswa.

## 📋 Fitur Utama

- **KNN Prediction**: Prediksi jurusan menggunakan algoritma K-Nearest Neighbors
- **AI Integration**: Integrasi dengan OpenAI GPT dan Google Gemini untuk analisis mendalam
- **Confusion Matrix**: Evaluasi performa dengan true positive, true negative, false positive, false negative
- **Interactive Mode**: Mode interaktif untuk input manual data siswa
- **Data Visualization**: Grafik distribusi skor dan confusion matrix
- **HTML Report**: Laporan evaluasi dalam format HTML
- **Sample Data Generator**: Generator data sampel untuk testing

## 📁 Struktur Project

```
jurusan-predictor/
├── config.py              # Konfigurasi aplikasi
├── main.py                 # Aplikasi utama
├── knn_predictor.py        # Model KNN
├── ai_agent.py             # Integrasi AI Agent
├── evaluation.py           # Sistem evaluasi
├── data_generator.py       # Generator data sampel
├── utils.py                # Fungsi utility
├── requirements.txt        # Dependencies
├── README.md              # Dokumentasi
├── data/                  # Folder data
│   ├── training_data.json
│   └── test_data.json
└── results/               # Folder hasil evaluasi
    ├── evaluation_results.json
    └── evaluation_report.html
```

## 🚀 Instalasi

1. **Clone repository** (atau copy semua file):
```bash
git clone <repository-url>
cd jurusan-predictor
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup API Keys** (edit `config.py`):
```python
# config.py
OPENAI_API_KEY = "your_openai_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
```

## 🎯 Cara Penggunaan

### 1. Menjalankan Aplikasi Utama
```bash
python main.py
```

### 2. Menu Aplikasi
- **Option 1**: Initialize Data & Train Model
- **Option 2**: Test Individual Prediction  
- **Option 3**: Run Full Evaluation (with Confusion Matrix)
- **Option 4**: Interactive Prediction
- **Option 5**: Show Configuration
- **Option 6**: Exit

### 3. Generate Data Sampel
```bash
python data_generator.py
```

### 4. Test Utility Functions
```bash
python utils.py
```

## 📊 Format Data

### Data Siswa (JSON)
```json
{
  "nama": "Ahmad Rizki",
  "matematika": 88,
  "fisika": 85,
  "kimia": 80,
  "biologi": 75,
  "bahasa_indonesia": 78,
  "bahasa_inggris": 82,
  "skor_logika": 87,
  "skor_kreativitas": 73,
  "skor_kepemimpinan": 78,
  "skor_komunikasi": 80,
  "jurusan_actual": "Teknik Informatika",
  "jurusan_diinginkan": "Teknik Informatika"
}
```

### Fitur yang Digunakan
- **Nilai Rapor**: Matematika, Fisika, Kimia, Biologi, Bahasa Indonesia, Bahasa Inggris (0-100)
- **Hasil Psikotes**: Skor Logika, Kreativitas, Kepemimpinan, Komunikasi (0-100)

## 🤖 AI Integration

### OpenAI GPT
```python
ai_agent.set_openai_key("your_api_key")
prediction = ai_agent.predict_with_openai(student_data, desired_major)
```

### Google Gemini
```python
ai_agent.set_gemini_key("your_api_key")
prediction = ai_agent.predict_with_gemini(student_data, desired_major)
```

## 📈 Evaluasi Model

### Confusion Matrix
- **True Positive (TP)**: AI dan KNN sama-sama benar
- **True Negative (TN)**: AI dan KNN sama-sama salah
- **False Positive (FP)**: AI benar, KNN salah
- **False Negative (FN)**: AI salah, KNN benar

### Metrik Evaluasi
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

## 🎨 Visualisasi

### Grafik yang Tersedia
- Distribusi skor siswa
- Confusion matrix heatmap
- Distribusi jurusan
- Correlation matrix fitur

### Contoh Pembuatan Grafik
```python
from utils import Utils

# Score distribution
Utils.create_score_distribution_plot(data, "results/score_distribution.png")

# Confusion matrix heatmap
Utils.create_confusion_matrix_heatmap(tp, tn, fp, fn, "results/confusion_matrix.png")
```

## 📋 Jurusan yang Tersedia

1. Teknik Informatika
2. Kedokteran
3. Teknik Elektro
4. Psikologi
5. Teknik Mesin
6. Ekonomi
7. Hukum

## 🔧 Konfigurasi

### File `config.py`
```python
# Model Configuration
KNN_NEIGHBORS = 3
RANDOM_STATE = 42

# File Paths
TRAINING_DATA_PATH = "data/training_data.json"
TEST_DATA_PATH = "data/test_data.json"

# AI Settings
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 200
```

## 📊 Contoh Output

### Prediksi Individual
```
🤖 KNN Prediction: Teknik Informatika
KNN Probabilities:
  Teknik Informatika: 0.667
  Teknik Elektro: 0.333

🧠 OpenAI Analysis:
  COCOK - Nilai matematika dan logika tinggi mendukung jurusan Teknik Informatika

🤖 Gemini Analysis:  
  COCOK - Profil akademik sesuai dengan requirements Teknik Informatika
```

### Confusion Matrix
```
=== CONFUSION MATRIX ===
True Positive (TP): 8
True Negative (TN): 2
False Positive (FP): 3
False Negative (FN): 2

Accuracy: 0.6667
Precision: 0.7273
Recall: 0.8000
F1-Score: 0.7619
```

## 🛠️ Development

### Struktur Class
- **KNNPredictor**: Model machine learning
- **AIAgent**: Integrasi AI services
- **EvaluationSystem**: Sistem evaluasi dan metrics
- **DataGenerator**: Generator data untuk testing
- **Utils**: Fungsi utility dan visualisasi

### Extending Functionality
```python
# Menambah jurusan baru
config.AVAILABLE_MAJORS.append("Teknik Sipil")

# Menambah fitur baru
new_features = ['nilai_olahraga', 'nilai_seni']
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Pastikan API key sudah diset di `config.py`
   - Periksa kuota API

2. **File Not Found**
   - Jalankan data generator terlebih dahulu
   - Periksa path file di config

3. **Import Error**
   - Install semua dependencies: `pip install -r requirements.txt`

4. **Model Not Trained**
   - Jalankan training terlebih dahulu (Menu 1)

## 📝 TODO / Future Improvements

- [ ] Support untuk format data Excel/CSV
- [ ] Web interface dengan Flask/Django
- [ ] Database integration
- [ ] Real-time prediction API
- [ ] Advanced feature engineering
- [ ] Cross-validation untuk model evaluation
- [ ] Support untuk multiple AI providers
- [ ] Automated hyperparameter tuning

## 🤝 Kontribusi

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - silakan gunakan untuk tujuan pembelajaran dan komersial.
<!-- 
## 👥 Tim Pengembang

- **Developer**: Ze
- **Email**: [your.email@example.com]
- **GitHub**: [your-github-username]

## 📞 Support

Jika ada pertanyaan atau masalah:
1. Buka issue di GitHub
2. Email ke: [support.email@example.com]
3. Dokumentasi lengkap: [wiki-link] -->

<!-- --- -->

<!-- **Happy Coding! 🚀** -->