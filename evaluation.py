# evaluation.py
"""
Evaluation System for comparing AI and KNN predictions
"""

import json
from typing import Dict, List, Tuple
import pandas as pd
from knn_predictor import KNNPredictor
from ai_agent import AIAgent
import config


class EvaluationSystem:
    def __init__(self, knn_predictor: KNNPredictor, ai_agent: AIAgent):
        self.knn_predictor = knn_predictor
        self.ai_agent = ai_agent
        self.results = []
        
    def load_test_data(self, test_data_path: str = config.TEST_DATA_PATH) -> List[Dict]:
        """Load test data from JSON file"""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Test data file {test_data_path} tidak ditemukan!")
            return []
        except Exception as e:
            print(f"Error loading test data: {e}")
            return []
    
    def evaluate_predictions(self, test_data_path: str = config.TEST_DATA_PATH, 
                           use_ai: str = "openai") -> List[Dict]:
        """Evaluate predictions and store results"""
        print(f"Loading test data from {test_data_path}...")
        test_data = self.load_test_data(test_data_path)
        
        if not test_data:
            return []
        
        print(f"Evaluating {len(test_data)} students...")
        self.results = []
        
        for i, student in enumerate(test_data, 1):
            print(f"Processing student {i}/{len(test_data)}: {student.get('nama', 'Unknown')}")
            
            # Get actual and desired majors
            actual_major = student['jurusan_actual']
            desired_major = student['jurusan_diinginkan']
            
            # Get KNN prediction
            knn_prediction = self.knn_predictor.predict(student)
            
            # Get AI prediction
            if use_ai == "openai":
                ai_response = self.ai_agent.predict_with_openai(student, desired_major)
            elif use_ai == "gemini":
                ai_response = self.ai_agent.predict_with_gemini(student, desired_major)
            else:
                ai_response = "Invalid AI model specified"
            
            # Parse AI response
            ai_recommends = self._parse_ai_response(ai_response)
            
            # Evaluate predictions
            result = self._evaluate_single_prediction(
                student, actual_major, desired_major, 
                knn_prediction, ai_response, ai_recommends
            )
            
            self.results.append(result)
        
        print("Evaluation completed!")
        return self.results
    
    def _parse_ai_response(self, ai_response: str) -> bool:
        """Parse AI response to determine if it recommends the major"""
        response_upper = ai_response.upper()
        return "COCOK" in response_upper and "TIDAK COCOK" not in response_upper
    
    def _evaluate_single_prediction(self, student: Dict, actual_major: str, 
                                  desired_major: str, knn_prediction: str,
                                  ai_response: str, ai_recommends: bool) -> Dict:
        """Evaluate a single student's predictions"""
        # Check if predictions are correct
        knn_correct = (knn_prediction == actual_major)
        
        # AI is correct if:
        # - It recommends the desired major AND the desired major matches actual
        # - It doesn't recommend the desired major AND the desired major doesn't match actual
        ai_correct = (ai_recommends and desired_major == actual_major) or \
                    (not ai_recommends and desired_major != actual_major)
        
        # Check if desired major matches actual
        desired_matches_actual = (desired_major == actual_major)
        
        return {
            'nama': student.get('nama', 'Unknown'),
            'actual_major': actual_major,
            'desired_major': desired_major,
            'knn_prediction': knn_prediction,
            'ai_response': ai_response,
            'ai_recommends': ai_recommends,
            'knn_correct': knn_correct,
            'ai_correct': ai_correct,
            'desired_matches_actual': desired_matches_actual,
            'knn_vs_actual': knn_prediction == actual_major,
            'student_data': {k: v for k, v in student.items() 
                           if k not in ['nama', 'jurusan_actual', 'jurusan_diinginkan']}
        }
    
    def calculate_confusion_matrix(self, results: List[Dict] = None) -> Dict:
        """Calculate confusion matrix for AI vs KNN comparison"""
        if results is None:
            results = self.results
        
        if not results:
            print("No results to evaluate. Run evaluate_predictions first.")
            return {}
        
        # Compare AI recommendation accuracy with KNN prediction accuracy
        tp = sum(1 for r in results if r['ai_correct'] and r['knn_correct'])
        tn = sum(1 for r in results if not r['ai_correct'] and not r['knn_correct'])
        fp = sum(1 for r in results if r['ai_correct'] and not r['knn_correct'])
        fn = sum(1 for r in results if not r['ai_correct'] and r['knn_correct'])
        
        total = len(results)
        
        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'total': total,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        return metrics
    
    def print_confusion_matrix(self, metrics: Dict = None):
        """Print confusion matrix and metrics"""
        if metrics is None:
            metrics = self.calculate_confusion_matrix()
        
        if not metrics:
            return
        
        print("\n" + "="*50)
        print("CONFUSION MATRIX - AI vs KNN Accuracy Comparison")
        print("="*50)
        print(f"True Positive (TP): {metrics['tp']} - Both AI and KNN correct")
        print(f"True Negative (TN): {metrics['tn']} - Both AI and KNN incorrect")
        print(f"False Positive (FP): {metrics['fp']} - AI correct, KNN incorrect")
        print(f"False Negative (FN): {metrics['fn']} - AI incorrect, KNN correct")
        print(f"Total Samples: {metrics['total']}")
        
        print("\n" + "-"*30)
        print("PERFORMANCE METRICS")
        print("-"*30)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    def print_detailed_results(self, show_all: bool = False):
        """Print detailed results for each student"""
        if not self.results:
            print("No results available. Run evaluate_predictions first.")
            return
        
        print("\n" + "="*80)
        print("DETAILED EVALUATION RESULTS")
        print("="*80)
        
        for i, result in enumerate(self.results, 1):
            if not show_all and i > 5:  # Show only first 5 by default
                print(f"... and {len(self.results) - 5} more results")
                break
                
            print(f"\nStudent {i}: {result['nama']}")
            print(f"Actual Major: {result['actual_major']}")
            print(f"Desired Major: {result['desired_major']}")
            print(f"KNN Prediction: {result['knn_prediction']} ({'✓' if result['knn_correct'] else '✗'})")
            print(f"AI Recommendation: {'COCOK' if result['ai_recommends'] else 'TIDAK COCOK'} ({'✓' if result['ai_correct'] else '✗'})")
            print(f"AI Response: {result['ai_response'][:100]}...")
            print("-" * 60)
    
    def save_results_to_json(self, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON file"""
        if not self.results:
            print("No results to save.")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if not self.results:
            return {}
        
        total = len(self.results)
        knn_correct_count = sum(1 for r in self.results if r['knn_correct'])
        ai_correct_count = sum(1 for r in self.results if r['ai_correct'])
        both_correct = sum(1 for r in self.results if r['knn_correct'] and r['ai_correct'])
        desired_matches_actual = sum(1 for r in self.results if r['desired_matches_actual'])
        
        return {
            'total_samples': total,
            'knn_accuracy': knn_correct_count / total,
            'ai_accuracy': ai_correct_count / total,
            'both_correct_rate': both_correct / total,
            'student_choice_accuracy': desired_matches_actual / total,
            'knn_correct_count': knn_correct_count,
            'ai_correct_count': ai_correct_count,
            'both_correct_count': both_correct,
            'desired_matches_actual_count': desired_matches_actual
        }