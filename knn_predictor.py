# knn_predictor.py
"""
K-Nearest Neighbors Predictor for Major Recommendation
"""

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple
import config


class KNNPredictor:
    def __init__(self, k: int = config.KNN_NEIGHBORS):
        self.k = k
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
        except FileNotFoundError:
            print(f"File {json_file_path} tidak ditemukan!")
            return None
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for training"""
        # Feature columns (excluding target and metadata)
        self.feature_columns = [col for col in df.columns 
                               if col not in config.EXCLUDE_COLUMNS]
        
        # Prepare features
        X = df[self.feature_columns].values
        
        # Prepare target
        y = self.label_encoder.fit_transform(df['jurusan_actual'])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, json_file_path: str = config.TRAINING_DATA_PATH):
        """Train the KNN model"""
        print("Loading training data...")
        df = self.load_data_from_json(json_file_path)
        
        if df is None:
            return False
        
        print("Preprocessing data...")
        X, y = self.preprocess_data(df)
        
        print(f"Training KNN model with k={self.k}...")
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k)
        self.knn_model.fit(X, y)
        
        self.trained = True
        print("Model trained successfully!")
        
        # Show training accuracy
        y_pred = self.knn_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Training accuracy: {accuracy:.4f}")
        
        return True
    
    def predict(self, student_data: Dict) -> str:
        """Predict major using KNN model"""
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
        predicted_major = self.label_encoder.inverse_transform(prediction)[0]
        
        return predicted_major
    
    def predict_proba(self, student_data: Dict) -> Dict:
        """Get prediction probabilities"""
        if not self.trained:
            return {}
        
        # Extract features
        features = []
        for col in self.feature_columns:
            if col in student_data:
                features.append(student_data[col])
            else:
                features.append(0)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get probabilities
        probabilities = self.knn_model.predict_proba(features_scaled)[0]
        
        # Map to class names
        class_names = self.label_encoder.inverse_transform(
            range(len(probabilities)))
        
        return dict(zip(class_names, probabilities))
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'trained': self.trained,
            'k_neighbors': self.k,
            'feature_count': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'classes': self.label_encoder.classes_.tolist() if self.trained else []
        }