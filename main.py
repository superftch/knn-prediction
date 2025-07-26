# main.py
"""
Main Application for Jurusan Predictor
Integrates KNN, AI Agent, and Evaluation System
"""

import sys
import json
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from knn_predictor import KNNPredictor
from ai_agent import AIAgent
from evaluation import EvaluationSystem
from data_generator import DataGenerator
import config


class JurusanPredictorApp:
    def __init__(self):
        self.knn_predictor = KNNPredictor()
        self.ai_agent = AIAgent()
        self.evaluation_system = None
        self.data_generator = DataGenerator()
        
    def setup_ai_agents(self):
        """Setup AI agents with API keys"""
        print("Setting up AI agents...")
        
        # Try to set API keys (will show warnings if keys are placeholder)
        if config.OPENAI_API_KEY != "your_openai_api_key_here":
            self.ai_agent.set_openai_key(config.OPENAI_API_KEY)
        else:
            print("⚠️  OpenAI API key not configured. Set in config.py")
            
        if config.GEMINI_API_KEY != "your_gemini_api_key_here":
            self.ai_agent.set_gemini_key(config.GEMINI_API_KEY)
        else:
            print("⚠️  Gemini API key not configured. Set in config.py")
        
        # Show AI status
        status = self.ai_agent.get_ai_status()
        print(f"AI Status - OpenAI: {'✓' if status['openai_ready'] else '✗'}, "
              f"Gemini: {'✓' if status['gemini_ready'] else '✗'}")
    
    def initialize_data(self):
        """Initialize sample data if not exists"""
        print("Initializing sample data...")
        self.data_generator.create_sample_data(
            training_samples=40, 
            test_samples=15
        )
    
    def train_model(self):
        """Train the KNN model"""
        print("\n" + "="*50)
        print("TRAINING KNN MODEL")
        print("="*50)
        
        success = self.knn_predictor.train_model()
        if success:
            model_info = self.knn_predictor.get_model_info()
            print(f"Model Info:")
            print(f"  - Features: {model_info['feature_count']}")
            print(f"  - Classes: {len(model_info['classes'])}")
            print(f"  - Available majors: {', '.join(model_info['classes'])}")
            return True
        else:
            print("❌ Failed to train model")
            return False
    
    def test_individual_prediction(self):
        """Test individual student prediction"""
        print("\n" + "="*50)
        print("INDIVIDUAL PREDICTION TEST")
        print("="*50)
        
        # Use custom sample data
        custom_samples = self.data_generator.create_custom_sample()
        test_student = custom_samples[0]  # Use first sample
        
        print(f"Testing student: {test_student['nama']}")
        print(f"Actual major: {test_student['jurusan_actual']}")
        print(f"Desired major: {test_student['jurusan_diinginkan']}")
        
        # KNN Prediction
        knn_result = self.knn_predictor.predict(test_student)
        knn_probabilities = self.knn_predictor.predict_proba(test_student)
        
        print(f"\n🤖 KNN Prediction: {knn_result}")
        print("KNN Probabilities:")
        for major, prob in sorted(knn_probabilities.items(), 
                                key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {major}: {prob:.3f}")
        
        # AI Predictions (if available)
        ai_status = self.ai_agent.get_ai_status()
        
        if ai_status['openai_ready']:
            print(f"\n🧠 OpenAI Prediction:")
            openai_result = self.ai_agent.predict_with_openai(
                test_student, test_student['jurusan_diinginkan']
            )
            print(f"  {openai_result}")
        
        if ai_status['gemini_ready']:
            print(f"\n🤖 Gemini Prediction:")
            gemini_result = self.ai_agent.predict_with_gemini(
                test_student, test_student['jurusan_diinginkan']
            )
            print(f"  {gemini_result}")
        
        if not ai_status['openai_ready'] and not ai_status['gemini_ready']:
            print("\n⚠️  No AI agents configured for comparison")
    
    def run_full_evaluation(self):
        """Run full evaluation with confusion matrix"""
        print("\n" + "="*50)
        print("FULL EVALUATION")
        print("="*50)
        
        # Initialize evaluation system
        self.evaluation_system = EvaluationSystem(
            self.knn_predictor, self.ai_agent
        )
        
        # Check if AI is available
        ai_status = self.ai_agent.get_ai_status()
        
        if ai_status['openai_ready']:
            ai_model = "openai"
            print("Using OpenAI for evaluation...")
        elif ai_status['gemini_ready']:
            ai_model = "gemini"
            print("Using Gemini for evaluation...")
        else:
            print("❌ No AI agents available for evaluation")
            print("Please configure API keys in config.py")
            return
        
        # Run evaluation
        results = self.evaluation_system.evaluate_predictions(use_ai=ai_model)
        
        if results:
            # Calculate and show confusion matrix
            metrics = self.evaluation_system.calculate_confusion_matrix()
            self.evaluation_system.print_confusion_matrix(metrics)
            
            # Show detailed results
            self.evaluation_system.print_detailed_results(show_all=False)
            
            # Show summary statistics
            summary = self.evaluation_system.get_summary_stats()
            self.print_summary_stats(summary)
            
            # Save results
            self.evaluation_system.save_results_to_json("results/evaluation_results.json")
        else:
            print("❌ No evaluation results generated")
    
    def print_summary_stats(self, summary: Dict):
        """Print summary statistics"""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total samples evaluated: {summary['total_samples']}")
        print(f"KNN accuracy: {summary['knn_accuracy']:.3f}")
        print(f"AI accuracy: {summary['ai_accuracy']:.3f}")
        print(f"Both models correct: {summary['both_correct_rate']:.3f}")
        print(f"Student choice accuracy: {summary['student_choice_accuracy']:.3f}")
        
        print(f"\nDetailed counts:")
        print(f"KNN correct: {summary['knn_correct_count']}/{summary['total_samples']}")
        print(f"AI correct: {summary['ai_correct_count']}/{summary['total_samples']}")
        print(f"Both correct: {summary['both_correct_count']}/{summary['total_samples']}")
        print(f"Student choice matches actual: {summary['desired_matches_actual_count']}/{summary['total_samples']}")
    
    def interactive_prediction(self):
        """Interactive prediction for user input"""
        print("\n" + "="*50)
        print("INTERACTIVE PREDICTION")
        print("="*50)
        
        if not self.knn_predictor.trained:
            print("❌ Model not trained yet!")
            return
        
        print("Enter student data (or press Enter for default values):")
        
        try:
            student_data = {}
            
            # Get input for each field
            fields = [
                ('matematika', 'Mathematics'),
                ('fisika', 'Physics'),
                ('kimia', 'Chemistry'),
                ('biologi', 'Biology'),
                ('bahasa_indonesia', 'Indonesian'),
                ('bahasa_inggris', 'English'),
                ('skor_logika', 'Logic Score'),
                ('skor_kreativitas', 'Creativity Score'),
                ('skor_kepemimpinan', 'Leadership Score'),
                ('skor_komunikasi', 'Communication Score')
            ]
            
            for field_key, field_name in fields:
                while True:
                    try:
                        value = input(f"{field_name} (0-100): ").strip()
                        if value == "":
                            student_data[field_key] = 75  # Default value
                            break
                        else:
                            score = int(value)
                            if 0 <= score <= 100:
                                student_data[field_key] = score
                                break
                            else:
                                print("Score must be between 0-100")
                    except ValueError:
                        print("Please enter a valid number")
            
            # Get desired major
            print(f"\nAvailable majors: {', '.join(config.AVAILABLE_MAJORS)}")
            desired_major = input("Desired major: ").strip()
            if not desired_major:
                desired_major = "Teknik Informatika"  # Default
            
            # Make predictions
            print(f"\n{'='*30}")
            print("PREDICTION RESULTS")
            print(f"{'='*30}")
            
            # KNN prediction
            knn_prediction = self.knn_predictor.predict(student_data)
            print(f"🤖 KNN Recommendation: {knn_prediction}")
            
            # AI predictions
            ai_status = self.ai_agent.get_ai_status()
            
            if ai_status['openai_ready']:
                openai_result = self.ai_agent.predict_with_openai(student_data, desired_major)
                print(f"\n🧠 OpenAI Analysis:")
                print(f"  {openai_result}")
            
            if ai_status['gemini_ready']:
                gemini_result = self.ai_agent.predict_with_gemini(student_data, desired_major)
                print(f"\n🤖 Gemini Analysis:")
                print(f"  {gemini_result}")
            
        except KeyboardInterrupt:
            print("\nInteractive session cancelled")
    
    def show_menu(self):
        """Show main menu"""
        print("\n" + "="*60)
        print("JURUSAN PREDICTOR - MAIN MENU")
        print("="*60)
        print("1. Initialize Data & Train Model")
        print("2. Test Individual Prediction")
        print("3. Run Full Evaluation (with Confusion Matrix)")
        print("4. Interactive Prediction")
        print("5. Show Configuration")
        print("6. Exit")
        print("="*60)
    
    def show_configuration(self):
        """Show current configuration"""
        print("\n" + "="*50)
        print("CURRENT CONFIGURATION")
        print("="*50)
        
        print(f"KNN Neighbors: {config.KNN_NEIGHBORS}")
        print(f"Training Data: {config.TRAINING_DATA_PATH}")
        print(f"Test Data: {config.TEST_DATA_PATH}")
        print(f"Available Majors: {len(config.AVAILABLE_MAJORS)}")
        
        for i, major in enumerate(config.AVAILABLE_MAJORS, 1):
            print(f"  {i}. {major}")
        
        # AI Status
        ai_status = self.ai_agent.get_ai_status()
        print(f"\nAI Configuration:")
        print(f"  OpenAI: {'✓ Ready' if ai_status['openai_ready'] else '✗ Not configured'}")
        print(f"  Gemini: {'✓ Ready' if ai_status['gemini_ready'] else '✗ Not configured'}")
        
        # Model Status
        model_info = self.knn_predictor.get_model_info()
        print(f"\nModel Status:")
        print(f"  Trained: {'✓ Yes' if model_info['trained'] else '✗ No'}")
        if model_info['trained']:
            print(f"  Features: {model_info['feature_count']}")
            print(f"  Classes: {len(model_info['classes'])}")
    
    def run(self):
        """Run the main application"""
        print("🎓 JURUSAN PREDICTOR APPLICATION")
        print("Prediksi Jurusan dengan KNN + AI Integration")
        
        # Setup
        self.setup_ai_agents()
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\nPilih menu (1-6): ").strip()
                
                if choice == "1":
                    self.initialize_data()
                    self.train_model()
                    
                elif choice == "2":
                    if not self.knn_predictor.trained:
                        print("❌ Please train the model first (option 1)")
                    else:
                        self.test_individual_prediction()
                        
                elif choice == "3":
                    if not self.knn_predictor.trained:
                        print("❌ Please train the model first (option 1)")
                    else:
                        self.run_full_evaluation()
                        
                elif choice == "4":
                    self.interactive_prediction()
                    
                elif choice == "5":
                    self.show_configuration()
                    
                elif choice == "6":
                    print("👋 Terima kasih telah menggunakan Jurusan Predictor!")
                    break
                    
                else:
                    print("❌ Pilihan tidak valid")
                    
            except KeyboardInterrupt:
                print("\n👋 Program dihentikan oleh user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run application
    app = JurusanPredictorApp()
    app.run()


if __name__ == "__main__":
    main()