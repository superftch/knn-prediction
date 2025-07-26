# config.py
"""
Configuration file
"""

# API Configuration
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""

# Model Configuration
KNN_NEIGHBORS = 3
RANDOM_STATE = 42

# File Paths
TRAINING_DATA_PATH = "data/training_data.json"
TEST_DATA_PATH = "data/test_data.json"

# Feature columns (exclude these from features)
EXCLUDE_COLUMNS = ['nama', 'jurusan_actual', 'jurusan_diinginkan']

# Available majors
AVAILABLE_MAJORS = [
    'Teknik Informatika',
    'Kedokteran', 
    'Teknik Elektro',
    'Psikologi',
    'Teknik Mesin',
    'Ekonomi',
    'Hukum'
]

# AI Model Settings
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 200