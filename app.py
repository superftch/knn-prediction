import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import openai
import google.generativeai as genai
import requests
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class JurusanPredictor:
    def __init__(self):
        self.knn_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.trained = False
        
    def load_data_from_json(self, json_file_path: str) -> pd.DataFrame:
        """Load data from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for training"""
        # Feature columns (excluding target)
        self.feature_columns = [col for col in df.columns if col not in ['nama', 'jurusan_actual']]
        
        # Prepare features
        X = df[self.feature_columns].values
        
        # Prepare target
        y = self.label_encoder.fit_transform(df['jurusan_actual'])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, json_file_path: str, k: int = 5):
        """Train the KNN model"""
        print("Loading training data...")
        df = self.load_data_from_json(json_file_path)
        
        if df is None:
            return False
        
        print("Preprocessing data...")
        X, y = self.preprocess_data(df)
        
        print(f"Training KNN model with k={k}...")
        self.knn_model = KNeighborsClassifier(n_neighbors=k)
        self.knn_model.fit(X, y)
        
        self.trained = True
        print("Model trained successfully!")
        
        # Show training accuracy
        y_pred = self.knn_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Training accuracy: {accuracy:.4f}")
        
        return True
    
    def predict_with_knn(self, student_data: Dict) -> str:
        """Predict using KNN model"""
        if not self.trained:
            return "Model belum dilatih!"
        
        # Extract features in the same order as training
        features = []
        for col in self.feature_columns:
            if col in student_data:
                features.append(student_data[col])
            else:
                features.append(0)  # Default value if missing
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.knn_model.predict(features_scaled)
        predicted_jurusan = self.label_encoder.inverse_transform(prediction)[0]
        
        return predicted_jurusan

class AIAgent:
    def __init__(self):
        self.openai_api_key = None
        self.gemini_api_key = None
        
    def set_openai_key(self, api_key: str):
        """Set OpenAI API key"""
        self.openai_api_key = api_key
        openai.api_key = api_key
    
    def set_gemini_key(self, api_key: str):
        """Set Gemini API key"""
        self.gemini_api_key = api_key
        genai.configure(api_key=api_key)
    
    def predict_with_openai(self, student_data: Dict, desired_major: str) -> str:
        """Get prediction from OpenAI GPT"""
        try:
            prompt = self._create_prompt(student_data, desired_major)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Anda adalah konselor pendidikan yang ahli dalam memberikan rekomendasi jurusan berdasarkan nilai rapor dan hasil psikotes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error with OpenAI: {e}"
    
    def predict_with_gemini(self, student_data: Dict, desired_major: str) -> str:
        """Get prediction from Gemini"""
        try:
            model = genai.GenerativeModel('gemini-pro')
            prompt = self._create_prompt(student_data, desired_major)
            
            response = model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            return f"Error with Gemini: {e}"
    
    def _create_prompt(self, student_data: Dict, desired_major: str) -> str:
        """Create prompt for AI prediction"""
        prompt = f"""
        Berdasarkan data siswa berikut:
        - Nilai Matematika: {student_data.get('matematika', 0)}
        - Nilai Fisika: {student_data.get('fisika', 0)}
        - Nilai Kimia: {student_data.get('kimia', 0)}
        - Nilai Biologi: {student_data.get('biologi', 0)}
        - Nilai Bahasa Indonesia: {student_data.get('bahasa_indonesia', 0)}
        - Nilai Bahasa Inggris: {student_data.get('bahasa_inggris', 0)}
        - Skor Logika: {student_data.get('skor_logika', 0)}
        - Skor Kreativitas: {student_data.get('skor_kreativitas', 0)}
        - Skor Kepemimpinan: {student_data.get('skor_kepemimpinan', 0)}
        - Skor Komunikasi: {student_data.get('skor_komunikasi', 0)}
        
        Siswa ini ingin masuk jurusan: {desired_major}
        
        Berdasarkan nilai rapor dan hasil psikotes di atas, apakah jurusan {desired_major} cocok untuk siswa ini?
        Jawab dengan format: "COCOK" atau "TIDAK COCOK" diikuti penjelasan singkat.
        """
        return prompt

class EvaluationSystem:
    def __init__(self, predictor: JurusanPredictor, ai_agent: AIAgent):
        self.predictor = predictor
        self.ai_agent = ai_agent
    
    def evaluate_predictions(self, test_data_path: str, use_ai: str = "openai"):
        """Evaluate predictions and calculate confusion matrix"""
        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)
        
        results = []
        
        for student in test_data:
            # Get actual jurusan
            actual_jurusan = student['jurusan_actual']
            desired_jurusan = student['jurusan_diinginkan']
            
            # Get KNN prediction
            knn_prediction = self.predictor.predict_with_knn(student)
            
            # Get AI prediction
            if use_ai == "openai":
                ai_response = self.ai_agent.predict_with_openai(student, desired_jurusan)
            else:
                ai_response = self.ai_agent.predict_with_gemini(student, desired_jurusan)
            
            # Determine if AI recommends the desired major
            ai_recommends = "COCOK" in ai_response.upper()
            
            # Compare predictions
            knn_correct = (knn_prediction == actual_jurusan)
            ai_correct = (ai_recommends and desired_jurusan == actual_jurusan) or \
                        (not ai_recommends and desired_jurusan != actual_jurusan)
            
            results.append({
                'nama': student['nama'],
                'actual': actual_jurusan,
                'desired': desired_jurusan,
                'knn_prediction': knn_prediction,
                'ai_response': ai_response,
                'knn_correct': knn_correct,
                'ai_correct': ai_correct,
                'knn_vs_actual': knn_prediction == actual_jurusan,
                'ai_vs_desired': ai_recommends
            })
        
        return results
    
    def calculate_confusion_matrix(self, results: List[Dict]):
        """Calculate confusion matrix for AI vs KNN comparison"""
        # Compare AI recommendation with KNN prediction accuracy
        tp = sum(1 for r in results if r['ai_correct'] and r['knn_correct'])
        tn = sum(1 for r in results if not r['ai_correct'] and not r['knn_correct'])
        fp = sum(1 for r in results if r['ai_correct'] and not r['knn_correct'])
        fn = sum(1 for r in results if not r['ai_correct'] and r['knn_correct'])
        
        print("\n=== CONFUSION MATRIX ===")
        print(f"True Positive (TP): {tp}")
        print(f"True Negative (TN): {tn}")
        print(f"False Positive (FP): {fp}")
        print(f"False Negative (FN): {fn}")
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 
                'accuracy': accuracy, 'precision': precision, 
                'recall': recall, 'f1_score': f1_score}

def create_sample_data():
    """Create sample training and test data"""
    
    # Sample training data
    training_data = [
        {
            "nama": "Andi",
            "matematika": 85,
            "fisika": 80,
            "kimia": 78,
            "biologi": 70,
            "bahasa_indonesia": 75,
            "bahasa_inggris": 80,
            "skor_logika": 85,
            "skor_kreativitas": 70,
            "skor_kepemimpinan": 75,
            "skor_komunikasi": 80,
            "jurusan_actual": "Teknik Informatika"
        },
        {
            "nama": "Budi",
            "matematika": 70,
            "fisika": 75,
            "kimia": 85,
            "biologi": 90,
            "bahasa_indonesia": 80,
            "bahasa_inggris": 75,
            "skor_logika": 75,
            "skor_kreativitas": 80,
            "skor_kepemimpinan": 70,
            "skor_komunikasi": 85,
            "jurusan_actual": "Kedokteran"
        },
        {
            "nama": "Citra",
            "matematika": 90,
            "fisika": 85,
            "kimia": 75,
            "biologi": 65,
            "bahasa_indonesia": 70,
            "bahasa_inggris": 85,
            "skor_logika": 90,
            "skor_kreativitas": 75,
            "skor_kepemimpinan": 80,
            "skor_komunikasi": 70,
            "jurusan_actual": "Teknik Elektro"
        },
        {
            "nama": "Dina",
            "matematika": 75,
            "fisika": 70,
            "kimia": 70,
            "biologi": 85,
            "bahasa_indonesia": 90,
            "bahasa_inggris": 85,
            "skor_logika": 70,
            "skor_kreativitas": 90,
            "skor_kepemimpinan": 85,
            "skor_komunikasi": 95,
            "jurusan_actual": "Psikologi"
        },
        {
            "nama": "Eko",
            "matematika": 95,
            "fisika": 90,
            "kimia": 80,
            "biologi": 70,
            "bahasa_indonesia": 75,
            "bahasa_inggris": 80,
            "skor_logika": 95,
            "skor_kreativitas": 70,
            "skor_kepemimpinan": 75,
            "skor_komunikasi": 70,
            "jurusan_actual": "Teknik Informatika"
        }
    ]
    
    # Sample test data
    test_data = [
        {
            "nama": "Farah",
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
        },
        {
            "nama": "Gilang",
            "matematika": 75,
            "fisika": 78,
            "kimia": 88,
            "biologi": 92,
            "bahasa_indonesia": 82,
            "bahasa_inggris": 78,
            "skor_logika": 78,
            "skor_kreativitas": 82,
            "skor_kepemimpinan": 75,
            "skor_komunikasi": 88,
            "jurusan_actual": "Kedokteran",
            "jurusan_diinginkan": "Kedokteran"
        }
    ]
    
    # Save to JSON files
    with open('training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    with open('test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print("Sample data files created: training_data.json, test_data.json")

def main():
    """Main application function"""
    print("=== APLIKASI PREDIKSI JURUSAN ===")
    
    # Create sample data
    create_sample_data()
    
    # Initialize components
    predictor = JurusanPredictor()
    ai_agent = AIAgent()
    
    # Set API keys (replace with your actual keys)
    # ai_agent.set_openai_key("your_openai_api_key_here")
    # ai_agent.set_gemini_key("your_gemini_api_key_here")
    
    # Train KNN model
    if predictor.train_model('training_data.json', k=3):
        print("\n=== TESTING INDIVIDUAL PREDICTION ===")
        
        # Test individual prediction
        test_student = {
            "matematika": 88,
            "fisika": 85,
            "kimia": 80,
            "biologi": 75,
            "bahasa_indonesia": 78,
            "bahasa_inggris": 82,
            "skor_logika": 87,
            "skor_kreativitas": 73,
            "skor_kepemimpinan": 78,
            "skor_komunikasi": 80
        }
        
        knn_result = predictor.predict_with_knn(test_student)
        print(f"KNN Prediction: {knn_result}")
        
        # Uncomment these lines when you have API keys
        # ai_result_openai = ai_agent.predict_with_openai(test_student, "Teknik Informatika")
        # print(f"OpenAI Prediction: {ai_result_openai}")
        
        # ai_result_gemini = ai_agent.predict_with_gemini(test_student, "Teknik Informatika")
        # print(f"Gemini Prediction: {ai_result_gemini}")
        
        print("\n=== EVALUATION ===")
        print("Note: Untuk evaluasi lengkap dengan confusion matrix,")
        print("silakan set API key untuk OpenAI atau Gemini terlebih dahulu.")
        
        # Initialize evaluation system
        eval_system = EvaluationSystem(predictor, ai_agent)
        
        # Uncomment when you have API keys
        # results = eval_system.evaluate_predictions('test_data.json', use_ai="openai")
        # confusion_metrics = eval_system.calculate_confusion_matrix(results)
        
        print("\n=== DEMO COMPLETED ===")
        print("Untuk menggunakan fully:")
        print("1. Set API key OpenAI atau Gemini")
        print("2. Uncomment bagian kode yang menggunakan AI agent")
        print("3. Tambahkan lebih banyak data training untuk akurasi yang lebih baik")

if __name__ == "__main__":
    main()