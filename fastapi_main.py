# fastapi_main.py
"""
FastAPI Web Service for Jurusan Predictor
Provides REST API endpoints for major prediction using KNN and AI models
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
import uvicorn
import json
import os
import asyncio
from datetime import datetime
import logging

# Import custom modules
from knn_predictor import KNNPredictor
from ai_agent import AIAgent
from evaluation import EvaluationSystem
from data_generator import DataGenerator
from utils import Utils
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Jurusan Predictor API",
    description="AI-powered major recommendation system using KNN and LLM models",
    version="1.0.0",
    contact={
        "name": "Jurusan Predictor Team",
        "email": "support@jurusanpredictor.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
knn_predictor = KNNPredictor()
ai_agent = AIAgent()
data_generator = DataGenerator()

# Application state
app_state = {
    "model_trained": False,
    "ai_configured": False,
    "last_training": None,
    "prediction_count": 0,
    "evaluation_results": None
}

# Pydantic models for request/response
class StudentData(BaseModel):
    """Student data model for predictions"""
    matematika: float = Field(..., ge=0, le=100, description="Mathematics score (0-100)")
    fisika: float = Field(..., ge=0, le=100, description="Physics score (0-100)")
    kimia: float = Field(..., ge=0, le=100, description="Chemistry score (0-100)")
    biologi: float = Field(..., ge=0, le=100, description="Biology score (0-100)")
    bahasa_indonesia: float = Field(..., ge=0, le=100, description="Indonesian language score (0-100)")
    bahasa_inggris: float = Field(..., ge=0, le=100, description="English language score (0-100)")
    skor_logika: float = Field(..., ge=0, le=100, description="Logic score (0-100)")
    skor_kreativitas: float = Field(..., ge=0, le=100, description="Creativity score (0-100)")
    skor_kepemimpinan: float = Field(..., ge=0, le=100, description="Leadership score (0-100)")
    skor_komunikasi: float = Field(..., ge=0, le=100, description="Communication score (0-100)")
    
    @validator('*', pre=True)
    def validate_scores(cls, v):
        """Validate that all scores are numeric and within range"""
        if not isinstance(v, (int, float)):
            raise ValueError("Score must be a number")
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return float(v)

class StudentDataWithName(StudentData):
    """Extended student data with optional name"""
    nama: Optional[str] = Field(None, description="Student name (optional)")

class PredictionRequest(BaseModel):
    """Request model for major prediction"""
    student_data: StudentData
    desired_major: Optional[str] = Field(None, description="Desired major for AI evaluation")
    include_probabilities: bool = Field(False, description="Include prediction probabilities")
    ai_model: Optional[str] = Field("openai", description="AI model to use (openai/gemini)")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    students: List[StudentDataWithName]
    include_probabilities: bool = Field(False, description="Include prediction probabilities")
    ai_model: Optional[str] = Field("openai", description="AI model to use")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    success: bool
    knn_prediction: Optional[str] = None
    knn_probabilities: Optional[Dict[str, float]] = None
    ai_recommendation: Optional[str] = None
    ai_response: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    message: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    success: bool
    results: List[Dict]
    summary: Dict
    processing_time: float
    message: Optional[str] = None

class TrainingRequest(BaseModel):
    """Request model for model training"""
    k_neighbors: Optional[int] = Field(3, ge=1, le=20, description="Number of neighbors for KNN")
    generate_sample_data: bool = Field(True, description="Generate sample training data")
    training_samples: int = Field(50, ge=10, le=1000, description="Number of training samples")

class APIKeyRequest(BaseModel):
    """Request model for setting API keys"""
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None

class EvaluationRequest(BaseModel):
    """Request model for model evaluation"""
    test_samples: int = Field(20, ge=5, le=100, description="Number of test samples")
    ai_model: str = Field("openai", description="AI model for evaluation")

# Dependency functions
async def get_trained_model():
    """Dependency to ensure model is trained"""
    if not app_state["model_trained"]:
        raise HTTPException(
            status_code=400, 
            detail="Model not trained. Please train the model first using /train endpoint."
        )
    return knn_predictor

async def get_ai_agent():
    """Dependency to get configured AI agent"""
    if not app_state["ai_configured"]:
        raise HTTPException(
            status_code=400,
            detail="AI agent not configured. Please set API keys using /config/api-keys endpoint."
        )
    return ai_agent

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jurusan Predictor API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; }
            .endpoint { background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #27ae60; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎓 Jurusan Predictor API</h1>
                <p>AI-powered major recommendation system</p>
            </div>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/docs</strong> - Interactive API documentation
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong> - Health check
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/train</strong> - Train the KNN model
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/predict</strong> - Single student prediction
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/predict/batch</strong> - Batch predictions
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/evaluate</strong> - Model evaluation
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/status</strong> - System status
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/config/api-keys</strong> - Configure AI API keys
            </div>
            
            <p><a href="/docs" target="_blank">👉 Go to Interactive Documentation</a></p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "model_trained": app_state["model_trained"],
        "ai_configured": app_state["ai_configured"]
    }

@app.get("/status")
async def get_status():
    """Get system status and statistics"""
    ai_status = ai_agent.get_ai_status()
    model_info = knn_predictor.get_model_info() if app_state["model_trained"] else {}
    
    return {
        "system": {
            "model_trained": app_state["model_trained"],
            "last_training": app_state["last_training"],
            "prediction_count": app_state["prediction_count"],
            "ai_configured": app_state["ai_configured"]
        },
        "ai_agents": ai_status,
        "model": model_info,
        "available_majors": config.AVAILABLE_MAJORS,
        "configuration": {
            "knn_neighbors": config.KNN_NEIGHBORS,
            "max_tokens": config.AI_MAX_TOKENS,
            "temperature": config.AI_TEMPERATURE
        }
    }

@app.post("/config/api-keys")
async def set_api_keys(request: APIKeyRequest):
    """Configure AI API keys"""
    try:
        configured = []
        
        if request.openai_key:
            ai_agent.set_openai_key(request.openai_key)
            configured.append("OpenAI")
        
        if request.gemini_key:
            ai_agent.set_gemini_key(request.gemini_key)
            configured.append("Gemini")
        
        if configured:
            app_state["ai_configured"] = True
            return {
                "success": True,
                "message": f"API keys configured for: {', '.join(configured)}",
                "configured_agents": configured
            }
        else:
            return {
                "success": False,
                "message": "No API keys provided"
            }
    
    except Exception as e:
        logger.error(f"Error setting API keys: {e}")
        raise HTTPException(status_code=500, detail=f"Error configuring API keys: {str(e)}")

@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the KNN model"""
    try:
        # Generate sample data if requested
        if request.generate_sample_data:
            logger.info("Generating sample training data...")
            data_generator.create_sample_data(
                training_samples=request.training_samples,
                test_samples=max(10, request.training_samples // 5)
            )
        
        # Update KNN neighbors if specified
        if request.k_neighbors != config.KNN_NEIGHBORS:
            knn_predictor.k = request.k_neighbors
        
        # Train model
        logger.info("Starting model training...")
        success = knn_predictor.train_model()
        
        if success:
            app_state["model_trained"] = True
            app_state["last_training"] = datetime.now().isoformat()
            
            model_info = knn_predictor.get_model_info()
            
            return {
                "success": True,
                "message": "Model trained successfully",
                "model_info": model_info,
                "training_time": app_state["last_training"],
                "k_neighbors": request.k_neighbors
            }
        else:
            raise HTTPException(status_code=500, detail="Model training failed")
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_major(
    request: PredictionRequest,
    trained_model: KNNPredictor = Depends(get_trained_model)
):
    """Predict major for a single student"""
    start_time = datetime.now()
    
    try:
        student_dict = request.student_data.dict()
        
        # KNN Prediction
        knn_prediction = trained_model.predict(student_dict)
        knn_probabilities = None
        
        if request.include_probabilities:
            knn_probabilities = trained_model.predict_proba(student_dict)
        
        # AI Prediction (if configured and desired major provided)
        ai_recommendation = None
        ai_response = None
        
        if request.desired_major and app_state["ai_configured"]:
            try:
                if request.ai_model == "openai":
                    ai_response = ai_agent.predict_with_openai(student_dict, request.desired_major)
                elif request.ai_model == "gemini":
                    ai_response = ai_agent.predict_with_gemini(student_dict, request.desired_major)
                
                ai_recommendation = "COCOK" if "COCOK" in ai_response.upper() else "TIDAK COCOK"
            except Exception as e:
                logger.warning(f"AI prediction failed: {e}")
                ai_response = f"AI prediction unavailable: {str(e)}"
        
        # Calculate confidence score (based on top probability)
        confidence_score = None
        if knn_probabilities:
            confidence_score = max(knn_probabilities.values())
        
        # Update prediction count
        app_state["prediction_count"] += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            success=True,
            knn_prediction=knn_prediction,
            knn_probabilities=knn_probabilities,
            ai_recommendation=ai_recommendation,
            ai_response=ai_response,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        return PredictionResponse(
            success=False,
            message=f"Prediction failed: {str(e)}",
            processing_time=processing_time
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    trained_model: KNNPredictor = Depends(get_trained_model)
):
    """Batch prediction for multiple students"""
    start_time = datetime.now()
    
    try:
        results = []
        successful_predictions = 0
        
        for i, student in enumerate(request.students):
            try:
                student_dict = student.dict()
                student_name = student_dict.pop('nama', f'Student_{i+1}')
                
                # KNN Prediction
                knn_prediction = trained_model.predict(student_dict)
                knn_probabilities = None
                
                if request.include_probabilities:
                    knn_probabilities = trained_model.predict_proba(student_dict)
                
                result = {
                    "name": student_name,
                    "knn_prediction": knn_prediction,
                    "success": True
                }
                
                if knn_probabilities:
                    result["knn_probabilities"] = knn_probabilities
                    result["confidence_score"] = max(knn_probabilities.values())
                
                results.append(result)
                successful_predictions += 1
                
            except Exception as e:
                results.append({
                    "name": f"Student_{i+1}",
                    "success": False,
                    "error": str(e)
                })
        
        # Update prediction count
        app_state["prediction_count"] += successful_predictions
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        summary = {
            "total_students": len(request.students),
            "successful_predictions": successful_predictions,
            "failed_predictions": len(request.students) - successful_predictions,
            "success_rate": successful_predictions / len(request.students) if request.students else 0
        }
        
        return BatchPredictionResponse(
            success=True,
            results=results,
            summary=summary,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        return BatchPredictionResponse(
            success=False,
            results=[],
            summary={},
            processing_time=processing_time,
            message=f"Batch prediction failed: {str(e)}"
        )

@app.post("/evaluate")
async def evaluate_model(
    request: EvaluationRequest,
    trained_model: KNNPredictor = Depends(get_trained_model),
    configured_ai: AIAgent = Depends(get_ai_agent)
):
    """Evaluate model performance"""
    try:
        # Generate test data if needed
        logger.info("Generating test data for evaluation...")
        _, test_data = data_generator.create_sample_data(
            training_samples=10,  # Don't need new training data
            test_samples=request.test_samples
        )
        
        # Initialize evaluation system
        eval_system = EvaluationSystem(trained_model, configured_ai)
        
        # Run evaluation
        logger.info("Running evaluation...")
        results = eval_system.evaluate_predictions(use_ai=request.ai_model)
        
        if not results:
            raise HTTPException(status_code=500, detail="Evaluation failed to generate results")
        
        # Calculate metrics
        metrics = eval_system.calculate_confusion_matrix(results)
        summary_stats = eval_system.get_summary_stats()
        
        # Store results in app state
        app_state["evaluation_results"] = {
            "results": results,
            "metrics": metrics,
            "summary": summary_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "metrics": metrics,
            "summary": summary_stats,
            "sample_results": results[:5],  # Return first 5 results as sample
            "total_evaluated": len(results),
            "message": "Evaluation completed successfully"
        }
    
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/evaluate/results")
async def get_evaluation_results():
    """Get latest evaluation results"""
    if app_state["evaluation_results"] is None:
        raise HTTPException(
            status_code=404, 
            detail="No evaluation results available. Run evaluation first using /evaluate endpoint."
        )
    
    return app_state["evaluation_results"]

@app.get("/majors")
async def get_available_majors():
    """Get list of available majors"""
    return {
        "majors": config.AVAILABLE_MAJORS,
        "total_count": len(config.AVAILABLE_MAJORS)
    }

@app.post("/generate-data")
async def generate_sample_data(
    training_samples: int = 50,
    test_samples: int = 20
):
    """Generate sample training and test data"""
    try:
        training_data, test_data = data_generator.create_sample_data(
            training_samples=training_samples,
            test_samples=test_samples
        )
        
        return {
            "success": True,
            "message": "Sample data generated successfully",
            "training_samples": len(training_data),
            "test_samples": len(test_data),
            "files_created": [
                config.TRAINING_DATA_PATH,
                config.TEST_DATA_PATH
            ]
        }
    
    except Exception as e:
        logger.error(f"Data generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Jurusan Predictor API...")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Try to set API keys from config if available
    try:
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
            ai_agent.set_openai_key()
            app_state["ai_configured"] = True
            logger.info("OpenAI API key configured from config")
    except Exception as e:
        logger.warning(f"Could not configure OpenAI from config: {e}")
    
    try:
        if config.GEMINI_API_KEY and config.GEMINI_API_KEY != "your_gemini_api_key_here":
            ai_agent.set_gemini_key()
            app_state["ai_configured"] = True
            logger.info("Gemini API key configured from config")
    except Exception as e:
        logger.warning(f"Could not configure Gemini from config: {e}")
    
    # Try to load existing trained model or train with sample data
    if os.path.exists(config.TRAINING_DATA_PATH):
        try:
            success = knn_predictor.train_model()
            if success:
                app_state["model_trained"] = True
                app_state["last_training"] = datetime.now().isoformat()
                logger.info("Model trained successfully from existing data")
        except Exception as e:
            logger.warning(f"Could not train model from existing data: {e}")
    
    logger.info("Jurusan Predictor API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Jurusan Predictor API...")

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )