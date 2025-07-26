"""
AI Agent Integration for Major Recommendation
Supports OpenAI GPT and Google Gemini
"""

import openai
import google.generativeai as genai
from typing import Dict
import config


class AIAgent:
    def __init__(self):
        self.openai_api_key = None
        self.gemini_api_key = None
        self.openai_client = None  # This will be an instance of openai.OpenAI
        self.gemini_model = None
        
    def set_openai_key(self, api_key: str = config.OPENAI_API_KEY):
        """Set OpenAI API key"""
        self.openai_api_key = api_key
        # Initialize the OpenAI client directly
        self.openai_client = openai.OpenAI(api_key=api_key) 
        print("OpenAI API key set successfully")
    
    def set_gemini_key(self, api_key: str = config.GEMINI_API_KEY):
        """Set Gemini API key"""
        self.gemini_api_key = api_key
        genai.configure(api_key=api_key)
        # Consider using a more recent Gemini model if available and suitable
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
        print("Gemini API key set successfully")
    
    def predict_with_openai(self, student_data: Dict, desired_major: str) -> str:
        """Get prediction from OpenAI GPT"""
        if not self.openai_client: # Check if the client instance exists
            return "OpenAI API key belum diset atau client belum diinisialisasi."
        
        try:
            prompt = self._create_prompt(student_data, desired_major)
            
            # Updated API call for openai>=1.0.0
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=config.AI_MAX_TOKENS,
                temperature=config.AI_TEMPERATURE
            )
            
            # Accessing content from the new response object
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"
    
    def predict_with_gemini(self, student_data: Dict, desired_major: str) -> str:
        """Get prediction from Gemini"""
        if not self.gemini_model:
            return "Gemini API key belum diset"
        
        try:
            prompt = self._create_prompt(student_data, desired_major)
            system_prompt = self._get_system_prompt()
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.gemini_model.generate_content(full_prompt)
            return response.text.strip()
        
        except Exception as e:
            return f"Error with Gemini: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for AI models"""
        return """Anda adalah konselor pendidikan yang ahli dalam memberikan rekomendasi jurusan berdasarkan nilai rapor dan hasil psikotes. 
        
Tugas Anda:
1. Analisis nilai rapor siswa (skala 0-100)
2. Analisis hasil psikotes (skor logika, kreativitas, kepemimpinan, komunikasi)
3. Evaluasi kesesuaian jurusan yang diinginkan
4. Berikan rekomendasi yang objektif

Format jawaban: "COCOK" atau "TIDAK COCOK" diikuti penjelasan singkat dan saran."""
    
    def _create_prompt(self, student_data: Dict, desired_major: str) -> str:
        """Create prompt for AI prediction"""
        prompt = f"""
Berdasarkan data siswa berikut:

NILAI RAPOR:
- Matematika: {student_data.get('matematika', 0)}/100
- Fisika: {student_data.get('fisika', 0)}/100  
- Kimia: {student_data.get('kimia', 0)}/100
- Biologi: {student_data.get('biologi', 0)}/100
- Bahasa Indonesia: {student_data.get('bahasa_indonesia', 0)}/100
- Bahasa Inggris: {student_data.get('bahasa_inggris', 0)}/100

HASIL PSIKOTES:
- Skor Logika: {student_data.get('skor_logika', 0)}/100
- Skor Kreativitas: {student_data.get('skor_kreativitas', 0)}/100  
- Skor Kepemimpinan: {student_data.get('skor_kepemimpinan', 0)}/100
- Skor Komunikasi: {student_data.get('skor_komunikasi', 0)}/100

JURUSAN YANG DIINGINKAN: {desired_major}

Pertanyaan: Apakah jurusan {desired_major} cocok untuk siswa ini berdasarkan nilai rapor dan hasil psikotes di atas?

Jawab dengan format:
[COCOK/TIDAK COCOK] - [Penjelasan singkat mengapa cocok/tidak cocok dan saran alternatif jika diperlukan]
        """
        return prompt.strip()
    
    def get_ai_status(self) -> Dict:
        """Get status of AI integrations"""
        return {
            'openai_ready': self.openai_client is not None, # Check the client instance
            'gemini_ready': self.gemini_model is not None,
            'available_models': ['openai', 'gemini']
        }