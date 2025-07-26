# utils.py
"""
Utility functions for Jurusan Predictor Application
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class Utils:
    @staticmethod
    def load_json(file_path: str) -> Any:
        """Load data from JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File {file_path} not found")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
        """Save data to JSON file with error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=indent, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")
            return False
    
    @staticmethod
    def validate_student_data(student_data: Dict) -> Dict:
        """Validate and clean student data"""
        required_fields = [
            'matematika', 'fisika', 'kimia', 'biologi',
            'bahasa_indonesia', 'bahasa_inggris',
            'skor_logika', 'skor_kreativitas', 
            'skor_kepemimpinan', 'skor_komunikasi'
        ]
        
        cleaned_data = {}
        errors = []
        
        for field in required_fields:
            if field not in student_data:
                cleaned_data[field] = 0
                errors.append(f"Missing field: {field}, set to 0")
            else:
                try:
                    value = float(student_data[field])
                    if 0 <= value <= 100:
                        cleaned_data[field] = value
                    else:
                        cleaned_data[field] = max(0, min(100, value))
                        errors.append(f"Field {field} out of range, clamped to {cleaned_data[field]}")
                except ValueError:
                    cleaned_data[field] = 0
                    errors.append(f"Invalid value for {field}, set to 0")
        
        return {
            'data': cleaned_data,
            'errors': errors,
            'is_valid': len(errors) == 0
        }
    
    @staticmethod
    def calculate_score_statistics(data: List[Dict]) -> Dict:
        """Calculate statistics for student scores"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        score_columns = [col for col in df.columns 
                        if col not in ['nama', 'jurusan_actual', 'jurusan_diinginkan']]
        
        stats = {}
        for column in score_columns:
            if column in df.columns:
                stats[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'median': df[column].median()
                }
        
        return stats
    
    @staticmethod
    def create_score_distribution_plot(data: List[Dict], save_path: str = None):
        """Create score distribution visualization"""
        try:
            df = pd.DataFrame(data)
            score_columns = [col for col in df.columns 
                           if col not in ['nama', 'jurusan_actual', 'jurusan_diinginkan']]
            
            # Set up the plot
            fig, axes = plt.subplots(2, 5, figsize=(20, 10))
            fig.suptitle('Score Distribution Analysis', fontsize=16)
            
            axes = axes.flatten()
            
            for i, column in enumerate(score_columns[:10]):  # Max 10 plots
                if column in df.columns:
                    axes[i].hist(df[column], bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{column.replace("_", " ").title()}')
                    axes[i].set_xlabel('Score')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(score_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating distribution plot: {e}")
    
    @staticmethod
    def create_confusion_matrix_heatmap(tp: int, tn: int, fp: int, fn: int, 
                                      save_path: str = None):
        """Create confusion matrix heatmap"""
        try:
            # Create confusion matrix
            cm = [[tp, fp], [fn, tn]]
            labels = [['True Positive', 'False Positive'], 
                     ['False Negative', 'True Negative']]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted Positive', 'Predicted Negative'],
                       yticklabels=['Actual Positive', 'Actual Negative'],
                       ax=ax)
            
            # Add labels inside cells
            for i in range(2):
                for j in range(2):
                    ax.text(j+0.5, i+0.7, labels[i][j], 
                           ha='center', va='center', fontsize=10, weight='bold')
            
            ax.set_title('Confusion Matrix: AI vs KNN Accuracy Comparison', 
                        fontsize=14, weight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Confusion matrix saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating confusion matrix heatmap: {e}")
    
    @staticmethod
    def generate_report(results: List[Dict], metrics: Dict, 
                       output_path: str = "results/evaluation_report.html"):
        """Generate HTML evaluation report"""
        try:
            # Calculate additional statistics
            total_students = len(results)
            ai_correct = sum(1 for r in results if r['ai_correct'])
            knn_correct = sum(1 for r in results if r['knn_correct'])
            
            # Create HTML content
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Jurusan Predictor - Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-card {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .correct {{ color: #27ae60; font-weight: bold; }}
        .incorrect {{ color: #e74c3c; font-weight: bold; }}
        .timestamp {{ text-align: right; color: #95a5a6; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Jurusan Predictor</h1>
            <h2>Evaluation Report</h2>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>📊 Performance Metrics</h3>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('accuracy', 0):.3f}</div>
                    <div class="metric-label">Overall Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('precision', 0):.3f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('recall', 0):.3f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('f1_score', 0):.3f}</div>
                    <div class="metric-label">F1-Score</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h3>🔄 Confusion Matrix</h3>
            <table style="width: 60%; margin: 0 auto;">
                <tr>
                    <th></th>
                    <th>Predicted Positive</th>
                    <th>Predicted Negative</th>
                </tr>
                <tr>
                    <th>Actual Positive</th>
                    <td style="text-align: center; background-color: #d5f4e6;">TP: {metrics.get('tp', 0)}</td>
                    <td style="text-align: center; background-color: #ffeaa7;">FP: {metrics.get('fp', 0)}</td>
                </tr>
                <tr>
                    <th>Actual Negative</th>
                    <td style="text-align: center; background-color: #ffeaa7;">FN: {metrics.get('fn', 0)}</td>
                    <td style="text-align: center; background-color: #d5f4e6;">TN: {metrics.get('tn', 0)}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h3>📈 Summary Statistics</h3>
            <ul>
                <li><strong>Total Students Evaluated:</strong> {total_students}</li>
                <li><strong>AI Correct Predictions:</strong> {ai_correct}/{total_students} ({ai_correct/total_students*100:.1f}%)</li>
                <li><strong>KNN Correct Predictions:</strong> {knn_correct}/{total_students} ({knn_correct/total_students*100:.1f}%)</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>📋 Detailed Results</h3>
            <table>
                <tr>
                    <th>Student</th>
                    <th>Actual Major</th>
                    <th>Desired Major</th>
                    <th>KNN Prediction</th>
                    <th>AI Recommendation</th>
                    <th>KNN Status</th>
                    <th>AI Status</th>
                </tr>"""
            
            # Add student results
            for result in results:
                knn_status = "✓ Correct" if result['knn_correct'] else "✗ Incorrect"
                ai_status = "✓ Correct" if result['ai_correct'] else "✗ Incorrect"
                knn_class = "correct" if result['knn_correct'] else "incorrect"
                ai_class = "correct" if result['ai_correct'] else "incorrect"
                
                html_content += f"""
                <tr>
                    <td>{result['nama']}</td>
                    <td>{result['actual_major']}</td>
                    <td>{result['desired_major']}</td>
                    <td>{result['knn_prediction']}</td>
                    <td>{'COCOK' if result['ai_recommends'] else 'TIDAK COCOK'}</td>
                    <td class="{knn_class}">{knn_status}</td>
                    <td class="{ai_class}">{ai_status}</td>
                </tr>"""
            
            html_content += """
            </table>
        </div>
        
        <div class="section">
            <h3>💡 Insights & Recommendations</h3>
            <ul>
                <li><strong>Model Performance:</strong> The evaluation shows how well AI and KNN models perform in predicting suitable majors.</li>
                <li><strong>True Positives:</strong> Cases where both models agree and are correct.</li>
                <li><strong>False Positives:</strong> Cases where AI is correct but KNN is wrong.</li>
                <li><strong>False Negatives:</strong> Cases where KNN is correct but AI is wrong.</li>
                <li><strong>True Negatives:</strong> Cases where both models are incorrect.</li>
            </ul>
        </div>
        
        <div class="section">
            <h3>⚙️ Technical Details</h3>
            <p><strong>Evaluation Method:</strong> K-Nearest Neighbors (KNN) vs AI Agent comparison</p>
            <p><strong>AI Models:</strong> OpenAI GPT / Google Gemini</p>
            <p><strong>Features Used:</strong> Academic scores (6 subjects) + Psychometric scores (4 dimensions)</p>
            <p><strong>Evaluation Criteria:</strong> Accuracy in recommending suitable majors based on student profiles</p>
        </div>
    </div>
</body>
</html>"""
            
            # Save HTML report
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML report generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return False
    
    @staticmethod
    def export_to_csv(data: List[Dict], file_path: str):
        """Export data to CSV file"""
        try:
            df = pd.DataFrame(data)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False, encoding='utf-8')
            print(f"Data exported to CSV: {file_path}")
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def create_major_distribution_chart(data: List[Dict], save_path: str = None):
        """Create major distribution pie chart"""
        try:
            df = pd.DataFrame(data)
            if 'jurusan_actual' not in df.columns:
                print("No major data found for distribution chart")
                return
            
            major_counts = df['jurusan_actual'].value_counts()
            
            plt.figure(figsize=(10, 8))
            colors = plt.cm.Set3(range(len(major_counts)))
            
            wedges, texts, autotexts = plt.pie(major_counts.values, 
                                             labels=major_counts.index,
                                             autopct='%1.1f%%',
                                             colors=colors,
                                             startangle=90)
            
            # Beautify the chart
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            plt.title('Distribution of Majors in Dataset', fontsize=16, weight='bold')
            plt.axis('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Major distribution chart saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating major distribution chart: {e}")
    
    @staticmethod
    def validate_api_keys():
        """Validate API key configuration"""
        import config
        
        validation_results = {
            'openai': {
                'configured': config.OPENAI_API_KEY != "your_openai_api_key_here",
                'key_preview': config.OPENAI_API_KEY[:10] + "..." if len(config.OPENAI_API_KEY) > 10 else config.OPENAI_API_KEY
            },
            'gemini': {
                'configured': config.GEMINI_API_KEY != "your_gemini_api_key_here",
                'key_preview': config.GEMINI_API_KEY[:10] + "..." if len(config.GEMINI_API_KEY) > 10 else config.GEMINI_API_KEY
            }
        }
        
        return validation_results
    
    @staticmethod
    def print_system_info():
        """Print system information and dependencies"""
        import sys
        import platform
        
        print("="*50)
        print("SYSTEM INFORMATION")
        print("="*50)
        print(f"Python Version: {sys.version}")
        print(f"Platform: {platform.platform()}")
        print(f"Architecture: {platform.architecture()[0]}")
        
        # Check required packages
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 
            'openai', 'google-generativeai', 'matplotlib', 'seaborn'
        ]
        
        print("\nPackage Status:")
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package} (not installed)")
        
        print("="*50)


class DataProcessor:
    """Advanced data processing utilities"""
    
    @staticmethod
    def normalize_scores(data: List[Dict]) -> List[Dict]:
        """Normalize scores to 0-1 range"""
        df = pd.DataFrame(data)
        score_columns = [col for col in df.columns 
                        if col not in ['nama', 'jurusan_actual', 'jurusan_diinginkan']]
        
        normalized_data = data.copy()
        
        for column in score_columns:
            if column in df.columns:
                min_val = df[column].min()
                max_val = df[column].max()
                
                if max_val > min_val:  # Avoid division by zero
                    for i, item in enumerate(normalized_data):
                        if column in item:
                            normalized_data[i][column] = (item[column] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    @staticmethod
    def detect_outliers(data: List[Dict], method='iqr') -> Dict:
        """Detect outliers in student data"""
        df = pd.DataFrame(data)
        score_columns = [col for col in df.columns 
                        if col not in ['nama', 'jurusan_actual', 'jurusan_diinginkan']]
        
        outliers = {}
        
        for column in score_columns:
            if column in df.columns:
                if method == 'iqr':
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_indices = df[(df[column] < lower_bound) | 
                                       (df[column] > upper_bound)].index.tolist()
                    
                    outliers[column] = {
                        'indices': outlier_indices,
                        'count': len(outlier_indices),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        
        return outliers
    
    @staticmethod
    def create_feature_correlation_matrix(data: List[Dict], save_path: str = None):
        """Create correlation matrix heatmap for features"""
        try:
            df = pd.DataFrame(data)
            score_columns = [col for col in df.columns 
                           if col not in ['nama', 'jurusan_actual', 'jurusan_diinginkan']]
            
            correlation_matrix = df[score_columns].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       square=True,
                       fmt='.2f')
            
            plt.title('Feature Correlation Matrix', fontsize=16, weight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Correlation matrix saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating correlation matrix: {e}")


# Example usage and testing
def main():
    """Demo of utility functions"""
    print("🔧 UTILS MODULE DEMO")
    
    # System info
    Utils.print_system_info()
    
    # Validate API keys
    api_validation = Utils.validate_api_keys()
    print(f"\nAPI Key Validation:")
    print(f"OpenAI: {'✓' if api_validation['openai']['configured'] else '✗'}")
    print(f"Gemini: {'✓' if api_validation['gemini']['configured'] else '✗'}")
    
    # Sample data validation
    sample_student = {
        'matematika': 85,
        'fisika': 'invalid',  # Invalid value to test validation
        'kimia': 120,  # Out of range
        # Missing some fields
    }
    
    validation_result = Utils.validate_student_data(sample_student)
    print(f"\nData Validation Result:")
    print(f"Valid: {validation_result['is_valid']}")
    print(f"Errors: {validation_result['errors']}")
    
    print("\nUtils demo completed!")


if __name__ == "__main__":
    main()