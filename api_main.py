from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import json
import logging
from datetime import datetime
import os
from pathlib import Path

# Import avec gestion d'erreur
try:
    from ml_pipeline import OptimizedFashionChatbot
except ImportError as e:
    print(f"❌ Erreur import ML pipeline: {e}")
    raise

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic simplifiés
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    success: bool
    intent: Optional[str] = None
    confidence: Optional[float] = None
    entities: Optional[Dict] = None
    response: str
    recommended_products: Optional[List[Dict]] = None
    timestamp: str

# Fonction de résolution de chemins simplifiée
def find_data_file(filename: str) -> str:
    """Trouve un fichier de données dans différents emplacements"""
    possible_paths = [
        filename,
        f"data/{filename}",
        f"../data/{filename}"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Fichier trouvé: {path}")
            return path
    
    raise FileNotFoundError(f"❌ Fichier '{filename}' introuvable")

# Initialisation FastAPI
app = FastAPI(
    title="Chanel Maison Heritage - Chatbot API",
    description="API intelligente pour conseillers de vente",
    version="2.0.1"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
chatbot_ml = None
products_data = None

# Startup event corrigé
@app.on_event("startup")
async def startup_event():
    """Initialisation corrigée au démarrage"""
    global chatbot_ml, products_data
    
    logger.info("🚀 Démarrage API Chanel Maison Heritage...")
    
    try:
        # Chargement des données produits
        products_path = find_data_file("products.json")
        with open(products_path, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        logger.info(f"✅ Produits chargés: {len(products_data.get('products', []))}")
        
        # Initialisation du modèle ML
        chatbot_ml = OptimizedFashionChatbot()
        
        # Tentative de chargement d'un modèle pré-entraîné
        model_path = "fashion_chatbot_optimized.pkl"
        if os.path.exists(model_path):
            try:
                chatbot_ml.load_model(model_path)
                logger.info("✅ Modèle pré-entraîné chargé")
            except Exception as e:
                logger.warning(f"⚠️  Échec chargement modèle: {e}")
                logger.info("🔄 Entraînement du nouveau modèle...")
                await train_new_model()
        else:
            logger.info("🔄 Entraînement du modèle...")
            await train_new_model()
        
        logger.info("✅ API prête !")
        
    except Exception as e:
        logger.error(f"❌ Erreur au démarrage: {e}")
        # Ne pas lever l'exception pour permettre à l'API de démarrer

async def train_new_model():
    """Entraînement du modèle avec gestion d'erreur"""
    try:
        training_path = find_data_file("training_data.json")
        products_path = find_data_file("products.json")
        
        results = chatbot_ml.train_full_pipeline(training_path, products_path)
        
        # Gestion flexible des clés de résultat
        if results.get('status') == 'optimized_success':
            intent_results = results.get('intent_classification', {})
            
            # Essai de différentes clés possibles
            accuracy = (intent_results.get('cv_accuracy') or 
                       intent_results.get('accuracy') or 
                       intent_results.get('best_score', 0))
            
            logger.info(f"✅ Modèle entraîné - Précision: {accuracy:.3f}")
            
            # Sauvegarde
            try:
                chatbot_ml.save_model("fashion_chatbot_optimized.pkl")
                logger.info("💾 Modèle sauvegardé")
            except Exception as e:
                logger.warning(f"⚠️  Sauvegarde échouée: {e}")
        else:
            logger.error(f"❌ Échec entraînement: {results.get('error', 'Erreur inconnue')}")
            
    except Exception as e:
        logger.error(f"❌ Erreur entraînement: {e}")

# Routes principales
@app.get("/", response_class=HTMLResponse)
async def root():
    """Interface web améliorée pour conseillers"""
    try:
        with open('app.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Erreur: Interface non trouvée</h1>"

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Endpoint principal pour le chat"""
    try:
        if not chatbot_ml:
            return ChatResponse(
                success=False,
                response="Service en cours d'initialisation, veuillez patienter...",
                timestamp=datetime.now().isoformat()
            )
        
        # Traitement de la requête
        result = chatbot_ml.process_query(message.message)
        
        return ChatResponse(
            success=result.get('success', True),
            intent=result.get('intent'),
            confidence=result.get('confidence'),
            entities=result.get('entities'),
            response=result.get('response', result.get('fallback_response', 'Désolé, je n\'ai pas compris.')),
            recommended_products=result.get('recommended_products', []),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur dans chat_endpoint: {e}")
        return ChatResponse(
            success=False,
            response="Une erreur technique est survenue.",
            timestamp=datetime.now().isoformat()
        )

@app.get("/health")
async def health_check():
    """Vérification de santé"""
    return {
        "status": "healthy" if chatbot_ml else "initializing",
        "timestamp": datetime.now().isoformat(),
        "ml_model": "loaded" if chatbot_ml and hasattr(chatbot_ml, 'intent_classifier') else "loading",
        "products": "loaded" if products_data else "not_loaded"
    }

@app.get("/api/stats")
async def get_stats():
    """Statistiques de l'API"""
    return {
        "status": "operational",
        "model_loaded": chatbot_ml is not None,
        "products_count": len(products_data.get('products', [])) if products_data else 0,
        "timestamp": datetime.now().isoformat()
    }

# Point d'entrée
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )