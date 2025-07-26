# data_generator.py
"""
Sample Data Generator for Jurusan Predictor
Creates training and test data in JSON format
"""

import json
import os
import random
from typing import List, Dict
import config


class DataGenerator:
    def __init__(self):
        self.majors = config.AVAILABLE_MAJORS
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs('data', exist_ok=True)
        print("Data directory created/verified")
    
    def generate_student_scores(self, major_bias: str = None) -> Dict:
        """Generate realistic student scores with optional major bias"""
        base_scores = {
            'matematika': random.randint(60, 100),
            'fisika': random.randint(60, 100),
            'kimia': random.randint(60, 100),
            'biologi': random.randint(60, 100),
            'bahasa_indonesia': random.randint(60, 100),
            'bahasa_inggris': random.randint(60, 100),
            'skor_logika': random.randint(60, 100),
            'skor_kreativitas': random.randint(60, 100),
            'skor_kepemimpinan': random.randint(60, 100),
            'skor_komunikasi': random.randint(60, 100)
        }
        
        # Apply bias based on major
        if major_bias:
            base_scores = self._apply_major_bias(base_scores, major_bias)
        
        return base_scores
    
    def _apply_major_bias(self, scores: Dict, major: str) -> Dict:
        """Apply realistic bias to scores based on major"""
        bias_patterns = {
            'Teknik Informatika': {
                'matematika': 10, 'fisika': 8, 'skor_logika': 12
            },
            'Kedokteran': {
                'biologi': 12, 'kimia': 10, 'skor_komunikasi': 8
            },
            'Teknik Elektro': {
                'matematika': 8, 'fisika': 12, 'skor_logika': 10
            },
            'Psikologi': {
                'bahasa_indonesia': 10, 'skor_komunikasi': 12, 'skor_kepemimpinan': 8
            },
            'Teknik Mesin': {
                'matematika': 8, 'fisika': 10, 'skor_logika': 8
            },
            'Ekonomi': {
                'matematika': 6, 'bahasa_inggris': 8, 'skor_kepemimpinan': 10
            },
            'Hukum': {
                'bahasa_indonesia': 12, 'skor_komunikasi': 10, 'skor_kepemimpinan': 8
            }
        }
        
        if major in bias_patterns:
            for subject, bonus in bias_patterns[major].items():
                if subject in scores:
                    scores[subject] = min(100, scores[subject] + bonus)
        
        return scores
    
    def generate_training_data(self, num_samples: int = 50) -> List[Dict]:
        """Generate training data"""
        training_data = []
        names = self._generate_names(num_samples)
        
        for i, name in enumerate(names):
            # Randomly select a major
            major = random.choice(self.majors)
            
            # Generate scores with bias toward the selected major
            scores = self.generate_student_scores(major)
            
            student = {
                'nama': name,
                **scores,
                'jurusan_actual': major
            }
            
            training_data.append(student)
        
        return training_data
    
    def generate_test_data(self, num_samples: int = 20) -> List[Dict]:
        """Generate test data with desired majors"""
        test_data = []
        names = self._generate_names(num_samples, start_index=100)
        
        for i, name in enumerate(names):
            # Randomly select actual major
            actual_major = random.choice(self.majors)
            
            # 70% chance desired matches actual, 30% chance it doesn't
            if random.random() < 0.7:
                desired_major = actual_major
            else:
                desired_major = random.choice([m for m in self.majors if m != actual_major])
            
            # Generate scores with bias toward actual major
            scores = self.generate_student_scores(actual_major)
            
            student = {
                'nama': name,
                **scores,
                'jurusan_actual': actual_major,
                'jurusan_diinginkan': desired_major
            }
            
            test_data.append(student)
        
        return test_data
    
    def _generate_names(self, count: int, start_index: int = 0) -> List[str]:
        """Generate realistic Indonesian names"""
        first_names = [
            'Andi', 'Budi', 'Citra', 'Dina', 'Eko', 'Farah', 'Gilang', 'Hana',
            'Indra', 'Joko', 'Kiki', 'Lina', 'Maya', 'Nina', 'Oscar', 'Putri',
            'Qori', 'Rina', 'Sari', 'Toni', 'Umar', 'Vina', 'Wati', 'Yani', 'Zaki',
            'Adit', 'Bella', 'Chandra', 'Dewi', 'Erik', 'Fitri', 'Galih', 'Hesti',
            'Ivan', 'Jihan', 'Kevin', 'Layla', 'Miko', 'Nanda', 'Oki', 'Prita'
        ]
        
        last_names = [
            'Pratama', 'Sari', 'Putra', 'Wati', 'Santoso', 'Lestari', 'Wijaya',
            'Safitri', 'Kurniawan', 'Maharani', 'Setiawan', 'Andriani', 'Susanto',
            'Permata', 'Hakim', 'Anggraini', 'Rahman', 'Kusuma', 'Firmansyah',
            'Pertiwi', 'Hidayat', 'Salsabila', 'Nugroho', 'Ramadhani'
        ]
        
        names = []
        for i in range(count):
            first = random.choice(first_names)
            last = random.choice(last_names)
            name = f"{first} {last}"
            
            # Ensure unique names
            counter = 1
            original_name = name
            while name in names:
                name = f"{original_name} {counter}"
                counter += 1
            
            names.append(name)
        
        return names
    
    def create_sample_data(self, training_samples: int = 50, test_samples: int = 20):
        """Create and save sample training and test data"""
        self.create_directories()
        
        print(f"Generating {training_samples} training samples...")
        training_data = self.generate_training_data(training_samples)
        
        print(f"Generating {test_samples} test samples...")
        test_data = self.generate_test_data(test_samples)
        
        # Save training data
        training_path = config.TRAINING_DATA_PATH
        with open(training_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to {training_path}")
        
        # Save test data
        test_path = config.TEST_DATA_PATH
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"Test data saved to {test_path}")
        
        return training_data, test_data
    
    def create_custom_sample(self) -> Dict:
        """Create a custom sample for individual testing"""
        sample_data = [
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
            },
            {
                "nama": "Siti Nurhaliza",
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
            },
            {
                "nama": "Bayu Pratama",
                "matematika": 70,
                "fisika": 72,
                "kimia": 68,
                "biologi": 65,
                "bahasa_indonesia": 85,
                "bahasa_inggris": 80,
                "skor_logika": 70,
                "skor_kreativitas": 88,
                "skor_kepemimpinan": 90,
                "skor_komunikasi": 92,
                "jurusan_actual": "Psikologi",
                "jurusan_diinginkan": "Teknik Informatika"  # Mismatch case
            }
        ]
        
        return sample_data
    
    def print_data_summary(self, data: List[Dict], data_type: str = "Data"):
        """Print summary of generated data"""
        if not data:
            print(f"No {data_type.lower()} available")
            return
        
        print(f"\n{data_type.upper()} SUMMARY:")
        print(f"Total samples: {len(data)}")
        
        # Major distribution
        if 'jurusan_actual' in data[0]:
            major_counts = {}
            for item in data:
                major = item['jurusan_actual']
                major_counts[major] = major_counts.get(major, 0) + 1
            
            print("Major distribution:")
            for major, count in sorted(major_counts.items()):
                print(f"  {major}: {count}")
        
        # Score statistics
        score_fields = ['matematika', 'fisika', 'kimia', 'biologi', 
                       'bahasa_indonesia', 'bahasa_inggris',
                       'skor_logika', 'skor_kreativitas', 
                       'skor_kepemimpinan', 'skor_komunikasi']
        
        print("\nScore ranges:")
        for field in score_fields:
            if field in data[0]:
                scores = [item[field] for item in data]
                print(f"  {field}: {min(scores)}-{max(scores)} (avg: {sum(scores)/len(scores):.1f})")


def main():
    """Demo function for data generator"""
    generator = DataGenerator()
    
    print("=== DATA GENERATOR DEMO ===")
    
    # Create sample data
    training_data, test_data = generator.create_sample_data(
        training_samples=30, 
        test_samples=10
    )
    
    # Print summaries
    generator.print_data_summary(training_data, "Training Data")
    generator.print_data_summary(test_data, "Test Data")
    
    print("\n=== CUSTOM SAMPLES ===")
    custom_samples = generator.create_custom_sample()
    generator.print_data_summary(custom_samples, "Custom Samples")
    
    print("\nData generation completed successfully!")


if __name__ == "__main__":
    main()