import json
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from difflib import get_close_matches
from collections import Counter

# ML Libraries 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# NLP 
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class PredictionResult:
    """Structure pour les résultats de prédiction"""
    intent: str
    confidence: float
    entities: Dict[str, str]
    response_template: str

class OptimizedTextPreprocessor:
    """Preprocessing ultra-optimisé pour le français"""
    
    def __init__(self):
        self.french_stopwords = set(stopwords.words('french'))
        
        # Stopwords à ne pas filtrer pour la mode
        self.preserve_words = {
            'sac', 'sur', 'en', 'de', 'du', 'le', 'la', 'un', 'une',
            'pour', 'avec', 'sans', 'très', 'bien', 'bon', 'belle'
        }
        
        # Filtrage intelligent des stopwords
        self.filtered_stopwords = self.french_stopwords - self.preserve_words
        
        # Mots fashion cruciaux à préserver
        self.fashion_keywords = {
            # Produits
            'tailleur', 'robe', 'sac', 'sacoche', 'maroquinerie',
            'chaussure', 'chaussures', 'escarpin', 'escarpins', 'ballerine', 'ballerines',
            'manteau', 'veste', 'bijou', 'bijoux', 'collier', 'bracelet',
            
            # Matières
            'tweed', 'cuir', 'agneau', 'soie', 'satin', 'matelasse', 'boucle',
            'laine', 'coton', 'perle', 'perles', 'metal',
            
            # Couleurs
            'noir', 'blanc', 'rouge', 'beige', 'marine', 'camel', 'gris', 'rose',
            'vert', 'bleu', 'dore', 'argente', 'multicolore', 'nude',
            
            # Occasions
            'business', 'bureau', 'travail', 'professionnel',
            'soiree', 'cocktail', 'gala', 'mariage', 'ceremonie',
            'quotidien', 'decontracte', 'casual', 'ville',
            
            # Actions/Verbes fashion
            'cherche', 'chercher', 'voudrais', 'veux', 'besoin',
            'recommande', 'conseille', 'suggere',
            'disponible', 'stock', 'inventaire',
            'prix', 'cout', 'tarif', 'budget',
            'entretien', 'nettoyer', 'soin', 'maintenance',
            
            # Marque/Style
            'chanel', 'heritage', 'maison', 'collection',
            'elegant', 'moderne', 'classique', 'iconique',
            'luxe', 'haut', 'gamme', 'prestige'
        }
        
        # Patterns d'entités améliorés
        self.entity_patterns = {
            'size': r'\b(3[4-8]|4[0-6]|small|medium|large|s|m|l|xl|petite?|grande?|taille)\b',
            'color': r'\b(noir|blanc|rouge|beige|marine|camel|rose|gris|bleu|vert|dore|argente|multicolore|nude)\w*\b',
            'price': r'\b(\d+[€$]|\d+\s*euros?|prix|cout|tarif|budget|montant)\b',
            'product_id': r'\b(CMH\d{3})\b',
            'occasion': r'\b(bureau|soiree|cocktail|mariage|business|quotidien|gala|decontracte|travail|ceremonie)\b',
            'material': r'\b(cuir|tweed|soie|satin|laine|agneau|perles?|metal|coton|matelasse)\b',
            'brand': r'\b(chanel|heritage|maison)\b',
            'location': r'\b(paris|lyon|cannes|france|magasin|boutique|entrepot)\b',
            'product_type': r'\b(sac|chaussure|robe|tailleur|manteau|veste|bijou|collier|escarpin|ballerine)\w*\b'
        }

        self.vocabulary = []  
        self.vocab_built = False  

    def build_vocabulary_from_data(self, training_data: List[Dict]):
        """Construit automatiquement le vocabulaire depuis les données"""
        if self.vocab_built:
            return
            
        all_words = []
        
        # Extraire tous les mots de vos données d'entraînement
        for example in training_data:
            text = example.get('text', '')
            # Preprocessing léger pour extraire mots propres
            clean_text = self.clean_and_normalize(text)
            words = clean_text.split()
            all_words.extend(words)
        
        # Ajouter mots de vos patterns d'entités existants
        for pattern in self.entity_patterns.values():
            # Extraire mots des regex (simplification)
            import re
            words_in_pattern = re.findall(r'\b[a-zA-Z]{3,}\b', pattern)
            all_words.extend(words_in_pattern)
        
        # Ajouter mots de vos fashion_keywords existants
        all_words.extend(self.fashion_keywords)
        
        # Garder seulement les mots fréquents (filtre automatique)
        word_counts = Counter(all_words)
        self.vocabulary = [word for word, count in word_counts.items() if count >= 2]
        
        self.vocab_built = True
        print(f"✅ Vocabulaire automatique construit: {len(self.vocabulary)} mots")
    
    def auto_correct_typos(self, text: str) -> str:
        """Correction automatique sans dictionnaire prédéfini"""
        if not self.vocab_built or not self.vocabulary:
            return text
        
        words = text.lower().split()
        corrected_words = []
        
        for word in words:
            # Seulement corriger si le mot fait 4+ caractères
            # (évite de corriger "sac" → "sec")
            if len(word) >= 4:
                # Chercher correspondance proche dans VOTRE vocabulaire
                matches = get_close_matches(
                    word, 
                    self.vocabulary, 
                    n=1,           
                    cutoff=0.8     # 80% de similarité minimum
                )
                
                if matches and matches[0] != word:
                    # Seulement si différent ET plus proche
                    corrected_words.append(matches[0])
                else:
                    corrected_words.append(word)
            else:
                # Mots courts : pas de correction
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def clean_and_normalize(self, text: str) -> str:
        """Nettoyage optimisé"""
        if not text:
            return ""
        
        text = text.lower()
        
        #  Normalisation française simplifiée
        replacements = {
            'à': 'a', 'â': 'a', 'ä': 'a', 'á': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'î': 'i', 'ï': 'i', 'í': 'i',
            'ô': 'o', 'ö': 'o', 'ó': 'o',
            'ù': 'u', 'û': 'u', 'ü': 'u', 'ú': 'u',
            'ç': 'c', 'ñ': 'n'
        }
        
        for accented, clean in replacements.items():
            text = text.replace(accented, clean)
        
        # Nettoyage doux
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extraction d'entités robuste"""
        entities = {}
        text_clean = self.clean_and_normalize(text)
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            if matches:
                # Déduplication et nettoyage
                unique_matches = list(set(match.strip() for match in matches if match.strip()))
                if unique_matches:
                    entities[entity_type] = unique_matches
        
        return entities
    
    def intelligent_tokenize(self, text: str) -> List[str]:
        """Tokenisation intelligente """
        clean_text = self.clean_and_normalize(text)
        tokens = clean_text.split()
        
        filtered_tokens = []
        for token in tokens:
            if token in self.fashion_keywords:
                # Garde ABSOLUMENT les mots fashion
                filtered_tokens.append(token)
            elif token in self.preserve_words:
                # Garde les mots courts importants
                filtered_tokens.append(token)
            elif token not in self.filtered_stopwords and len(token) >= 2:
                #  Seuil réduit à 2 caractères
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def preprocess_for_ml(self, text: str) -> str:
        """Preprocessing final pour ML"""

        text = self.auto_correct_typos(text)
        tokens = self.intelligent_tokenize(text)
        return ' '.join(tokens)

class OptimizedIntentClassifier:
    """Classificateur"""
    
    def __init__(self):
        self.preprocessor = OptimizedTextPreprocessor()
        
        # TF-IDF optimisé pour petit dataset
        self.vectorizer = TfidfVectorizer(
            max_features=1500,      
            ngram_range=(1, 2),     
            min_df=1,               
            max_df=0.85,            
            analyzer='word',
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            sublinear_tf=True
        )
        
        # Modèles mieux configurés
        self.models = {
            'logistic': LogisticRegression(
                C=0.5,                   
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear'        
            ),
            'svm_linear': SVC(
                kernel='linear',
                C=0.8,                    
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=150,        
                max_depth=8,              
                min_samples_split=3,     
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            
            'xgboost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'  
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1  
            )
        }
        
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
   
    
    def prepare_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Préparation robuste des données"""
        texts = []
        labels = []
        
        for example in training_data:
            if 'text' in example and 'intent' in example:
                processed_text = self.preprocessor.preprocess_for_ml(example['text'])
                if processed_text.strip(): 
                    texts.append(processed_text)
                    labels.append(example['intent'])
        
        return np.array(texts), np.array(labels)
    
    def train(self, training_data: List[Dict]) -> Dict[str, float]:
        """Entraînement robuste avec validation croisée"""
        print("🤖 Entraînement optimisé du classificateur...")
        
        # Préparation des données
        X_text, y = self.prepare_data(training_data)
        
        print(f"📊 Données: {len(X_text)} exemples, {len(set(y))} classes")
        print(f"📊 Classes: {list(set(y))}")
        
        #Vérification des données
        if len(X_text) == 0:
            raise ValueError("Aucune donnée valide après preprocessing!")
        
        # Encodage des labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Vectorisation
        X_vectorized = self.vectorizer.fit_transform(X_text)
        print(f"📊 Features TF-IDF: {X_vectorized.shape}")
        
        # Validation croisée pour sélection robuste
        model_scores = {}
        model_cv_scores = {}
        trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"   🔬 Test {model_name} avec validation croisée...")
            
            # Validation croisée stratifiée
            try:
                cv_scores = cross_val_score(
                    model, X_vectorized, y_encoded, 
                    cv=3, scoring='accuracy', n_jobs=-1
                )
                mean_cv_score = cv_scores.mean()
                model_cv_scores[model_name] = mean_cv_score
                
                # Entraînement sur tout le dataset
                model.fit(X_vectorized, y_encoded)
                trained_models[model_name] = model
                
                print(f"      ✅ {model_name}: CV={mean_cv_score:.3f} (±{cv_scores.std():.3f})")
                
            except Exception as e:
                print(f"      ❌ {model_name}: Erreur - {e}")
                model_cv_scores[model_name] = 0.0
        
        # Sélection basée sur validation croisée
        if model_cv_scores:
            self.best_model_name = max(model_cv_scores, key=model_cv_scores.get)
            self.best_model = trained_models[self.best_model_name]
            best_score = model_cv_scores[self.best_model_name]
        else:
            raise ValueError("Aucun modèle n'a pu être entraîné!")
        
        print(f"🏆 Meilleur: {self.best_model_name} (CV: {best_score:.3f})")

        # 🔥 AJOUTEZ ICI LE CODE DE TEST
        print("🧪 Test train/test split pour comparaison...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Créer une nouvelle instance du meilleur modèle pour le test
        test_model = trained_models[self.best_model_name].__class__(**trained_models[self.best_model_name].get_params())
        test_model.fit(X_train, y_train)
        simple_accuracy = test_model.score(X_test, y_test)
        print(f"🧪 Test simple (train/test): {simple_accuracy:.3f}")
        print(f"📊 Comparaison: CV={best_score:.3f} vs Train/Test={simple_accuracy:.3f}")
        
        self.is_trained = True
        
        return {
            'best_model': self.best_model_name,
            'cv_accuracy': best_score,
            'train_test_accuracy': simple_accuracy, 
            'all_cv_scores': model_cv_scores
        }
        
        self.is_trained = True
        
        return {
            'best_model': self.best_model_name,
            'cv_accuracy': best_score,
            'all_cv_scores': model_cv_scores
        }
    
    def predict(self, text: str) -> PredictionResult:
        """Prédiction robuste"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné!")
        
        # Preprocessing
        processed_text = self.preprocessor.preprocess_for_ml(text)
        
        # Gestion des textes vides
        if not processed_text.strip():
            return PredictionResult(
                intent='general_chat',
                confidence=0.5,
                entities={},
                response_template="Je n'ai pas bien compris, pouvez-vous reformuler ?"
            )
        
        entities = self.preprocessor.extract_entities(text)
        
        # Vectorisation
        X = self.vectorizer.transform([processed_text])
        
        # Prédiction
        try:
            predicted_label = self.best_model.predict(X)[0]
            
            # Confiance
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X)[0]
                confidence = probabilities.max()
            else:
                confidence = 0.8
            
            # Décodage
            intent = self.label_encoder.inverse_transform([predicted_label])[0]
            
            return PredictionResult(
                intent=intent,
                confidence=confidence,
                entities=entities,
                response_template=f"Intent: {intent}"
            )
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return PredictionResult(
                intent='general_chat',
                confidence=0.3,
                entities={},
                response_template="Erreur lors de l'analyse de votre demande."
            )

class RobustSemanticSearch:
    """Recherche sémantique """
    
    def __init__(self):
        print("🔍 Initialisation recherche sémantique robuste...")
        
        self.model = None
        self.use_semantic = False
        
        # Essai de plusieurs modèles avec fallback
        models_to_try = [
            'paraphrase-multilingual-MiniLM-L12-v2',
            'distiluse-base-multilingual-cased',
            'all-MiniLM-L6-v2'
        ]
        
        for model_name in models_to_try:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
                self.use_semantic = True
                print(f"✅ Modèle {model_name} chargé avec succès")
                break
            except Exception as e:
                print(f"⚠️  Tentative {model_name} échouée: {e}")
                continue
        
        if not self.use_semantic:
            print("📝 Mode recherche textuelle uniquement")
        
        self.product_embeddings = None
        self.products_df = None
        
    def index_products(self, products_data: List[Dict]):
        """Indexation robuste"""
        print("🔍 Indexation des produits...")
        
        self.products_df = pd.DataFrame(products_data)
        
        if not self.use_semantic:
            print("📝 Indexation textuelle uniquement")
            return
        
        # Descriptions enrichies et mieux structurées
        descriptions = []
        for product in products_data:
            # Composition intelligente de la description
            parts = []
            
            # Nom et catégorie (priorité haute)
            name = product.get('name', '')
            category = product.get('category', '')
            subcategory = product.get('subcategory', '')
            
            if name:
                parts.append(name)
            if category:
                parts.append(category)
            if subcategory:
                parts.append(subcategory)
            
            # Description
            description = product.get('description', '')
            if description:
                parts.append(description)
            
            # Matériaux et couleurs
            materials = product.get('materials', [])
            colors = product.get('colors', [])
            occasions = product.get('occasions', [])
            
            if materials:
                parts.append(' '.join(materials))
            if colors:
                parts.append(' '.join(colors))
            if occasions:
                parts.append(' '.join(occasions))
            
            # Style et tags
            style_profile = product.get('style_profile', [])
            tags = product.get('tags', [])
            
            if style_profile:
                parts.append(' '.join(style_profile))
            if tags:
                parts.append(' '.join(tags))
            
            description = ' '.join(filter(None, parts))
            descriptions.append(description)
        
        try:
            if self.model:
                self.product_embeddings = self.model.encode(descriptions, show_progress_bar=True)
                print(f"   ✅ {len(products_data)} produits indexés avec embeddings")
        except Exception as e:
            print(f"   ⚠️  Erreur indexation sémantique: {e}")
            self.product_embeddings = None
            self.use_semantic = False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Recherche hybride robuste"""
        if self.products_df is None or len(self.products_df) == 0:
            return []
        
        # Recherche sémantique avec fallback robuste
        if self.use_semantic and self.product_embeddings is not None:
            try:
                return self._semantic_search(query, top_k)
            except Exception as e:
                print(f"⚠️  Recherche sémantique échouée: {e}, fallback textuel")
                return self._enhanced_keyword_search(query, top_k)
        
        # Recherche textuelle améliorée
        return self._enhanced_keyword_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Recherche sémantique robuste"""
        if not self.model:
            raise ValueError("Modèle sémantique non disponible")
        
        query_embedding = self.model.encode([query])
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
        
        # Seuil adaptatif
        mean_similarity = similarities.mean()
        std_similarity = similarities.std()
        threshold = max(0.1, mean_similarity - 0.5 * std_similarity)
        
        # Top résultats avec seuil intelligent
        top_indices = similarities.argsort()[-top_k*2:][::-1]  # Prend plus pour filtrer
        
        results = []
        for idx in top_indices:
            if similarities[idx] > threshold and len(results) < top_k:
                product = self.products_df.iloc[idx].to_dict()
                product['similarity_score'] = float(similarities[idx])
                results.append(product)
        
        return results
    
    def _enhanced_keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Recherche textuelle intelligente """
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        
        results = []
        
        # Mots-clés étendus et plus précis
        enhanced_keywords = {
            'sac': ['sac', 'sacoche', 'maroquinerie', 'cabas', 'pochette', 'besace', 'cartable'],
            'chaussure': ['chaussure', 'chaussures', 'escarpin', 'escarpins', 'ballerine', 'ballerines', 'soulier', 'souliers', 'chaussant'],
            'robe': ['robe', 'robes', 'dress'],
            'tailleur': ['tailleur', 'tailleurs', 'costume', 'suit', 'ensemble'],
            'bijou': ['bijou', 'bijoux', 'collier', 'colliers', 'bracelet', 'bague', 'perle', 'perles', 'jewelry'],
            'manteau': ['manteau', 'manteaux', 'veste', 'vestes', 'blouson', 'coat', 'jacket'],
            'vetement': ['vetement', 'vetements', 'habit', 'habits', 'tenue', 'pret-a-porter']
        }
        
        # Score plus nuancé
        for _, product in self.products_df.iterrows():
            score = 0
            product_dict = product.to_dict()
            
            # Texte du produit en minuscules
            product_texts = {
                'name': product_dict.get('name', '').lower(),
                'category': product_dict.get('category', '').lower(),
                'subcategory': product_dict.get('subcategory', '').lower(),
                'description': product_dict.get('description', '').lower(),
                'materials': ' '.join(product_dict.get('materials', [])).lower(),
                'colors': ' '.join(product_dict.get('colors', [])).lower(),
                'occasions': ' '.join(product_dict.get('occasions', [])).lower(),
                'tags': ' '.join(product_dict.get('tags', [])).lower()
            }
            
            all_product_text = ' '.join(product_texts.values())
            
            # Correspondance par catégorie avec poids
            category_match_found = False
            for category, keywords in enhanced_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    # Vérifier correspondance dans les textes produit
                    for field, text in product_texts.items():
                        if any(kw in text for kw in keywords):
                            # Poids selon l'importance du champ
                            field_weights = {
                                'name': 15, 'category': 12, 'subcategory': 10,
                                'description': 8, 'tags': 6, 'materials': 4,
                                'colors': 3, 'occasions': 5
                            }
                            score += field_weights.get(field, 2)
                            category_match_found = True
            
            #  Correspondance de mots individuels
            query_words = [w for w in query_lower.split() if len(w) > 1]
            for word in query_words:
                if word in all_product_text:
                    # Bonus selon le contexte
                    if word in product_texts['name']:
                        score += 8
                    elif word in product_texts['category'] or word in product_texts['subcategory']:
                        score += 6
                    elif word in product_texts['description']:
                        score += 4
                    else:
                        score += 2
            
            # Bonus pour correspondance multi-mots
            if len(query_words) > 1:
                words_found = sum(1 for word in query_words if word in all_product_text)
                coverage = words_found / len(query_words)
                score += coverage * 5
            
            #  Bonus pour correspondance exacte d'expressions
            if len(query_lower) > 3 and query_lower in all_product_text:
                score += 10
            
            if score > 0:
                # Normalisation du score
                max_possible_score = 50  # Score théorique maximum
                normalized_score = min(score / max_possible_score, 1.0)
                product_dict['similarity_score'] = normalized_score
                results.append(product_dict)
        
        # Tri par score décroissant
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

class OptimizedFashionChatbot:
    """Chatbot fashion ultra-optimisé et fiable"""
    
    def __init__(self):
        self.intent_classifier = OptimizedIntentClassifier()
        self.semantic_search = RobustSemanticSearch()
        
    def load_training_data(self, training_file: str) -> Dict:
        """Chargement robuste des données"""
        try:
            with open(training_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Données d'entraînement chargées: {len(data.get('training_data', []))} exemples")
            return data
        except Exception as e:
            print(f"❌ Erreur chargement training data: {e}")
            raise
    
    def load_products_data(self, products_file: str) -> Dict:
        """Chargement robuste des produits"""
        try:
            with open(products_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Données produits chargées: {len(data.get('products', []))} produits")
            return data
        except Exception as e:
            print(f"❌ Erreur chargement products data: {e}")
            raise
    
    def train_full_pipeline(self, training_file: str, products_file: str) -> Dict:
        print("⚡ TF-IDF + Validation Croisée + Recherche hybride robuste")
        
        try:
            # Chargement des données
            training_data = self.load_training_data(training_file)
            products_data = self.load_products_data(products_file)
            
            # Validation des données
            if not training_data.get('training_data'):
                raise ValueError("Aucune donnée d'entraînement trouvée!")
            if not products_data.get('products'):
                raise ValueError("Aucun produit trouvé!")

            print("\n🔤 Construction du vocabulaire automatique...")
            self.intent_classifier.preprocessor.build_vocabulary_from_data(
                training_data['training_data']
            )
            
            # Entraînement
            print("\n📚 Phase d'entraînement...")
            training_results = self.intent_classifier.train(training_data['training_data'])
            
            # Indexation
            print("\n🔍 Phase d'indexation...")
            self.semantic_search.index_products(products_data['products'])
            
            print("\n✅ PIPELINE OPTIMISÉ PRÊT !")
            print(f"🎯 Précision CV: {training_results['cv_accuracy']:.1%}")
            print(f"🏆 Meilleur modèle: {training_results['best_model']}")
            
            return {
                'intent_classification': training_results,
                'products_indexed': len(products_data['products']),
                'semantic_search_enabled': self.semantic_search.use_semantic,
                'status': 'optimized_success'
            }
            
        except Exception as e:
            print(f"❌ Erreur pipeline: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'intent_classification': None,
                'products_indexed': 0
            }

    def process_query(self, user_query: str) -> Dict:
        """Traitement de requête ultra-robuste"""
        try:
            # Validation d'entrée
            if not user_query or not user_query.strip():
                return {
                    'success': True,
                    'intent': 'general_chat',
                    'confidence': 0.5,
                    'entities': {},
                    'recommended_products': [],
                    'response': "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
                }
            
            # Classification avec gestion d'erreur
            try:
                prediction = self.intent_classifier.predict(user_query)
            except Exception as e:
                print(f"⚠️  Erreur classification: {e}")
                return {
                    'success': False,
                    'error': f"Erreur classification: {e}",
                    'fallback_response': "Désolé, je n'ai pas pu analyser votre demande. Pouvez-vous reformuler ?"
                }
            
            # Recherche produits selon l'intention
            products = []
            if prediction.intent in ['product_search', 'style_recommendation', 'stock_check', 'price_inquiry']:
                try:
                    products = self.semantic_search.search(user_query, top_k=3)
                except Exception as e:
                    print(f"⚠️  Erreur recherche: {e}")
                    products = []
            
            #  Réponse structurée et robuste
            response = self._generate_enhanced_response(prediction, products, user_query)
            
            return {
                'success': True,
                'intent': prediction.intent,
                'confidence': prediction.confidence,
                'entities': prediction.entities,
                'recommended_products': products,
                'response': response,
                'debug_info': {
                    'processed_query': self.intent_classifier.preprocessor.preprocess_for_ml(user_query),
                    'search_enabled': self.semantic_search.use_semantic,
                    'products_found': len(products)
                }
            }
            
        except Exception as e:
            print(f"❌ Erreur process_query: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_response': "Une erreur technique s'est produite. Veuillez réessayer."
            }

    def _generate_enhanced_response(self, prediction: PredictionResult, products: List[Dict], query: str) -> str:
        """Génération de réponse intelligente et contextuelle - VERSION FINALE COMPLÈTE"""
        
        import random
        
        # Initialisation obligatoire de base_response
        base_response = ""
        
        # Templates de réponses enrichis
        response_templates = {
            'general_chat': {
                'greetings': ['bonjour', 'salut', 'hello', 'bonsoir'],
                'thanks': ['merci', 'thank', 'remercie'],
                'goodbye': ['au revoir', 'bye', 'aurevoir', 'adieu'],
                'affirmative': ['oui', 'yes', 'parfait', 'ok', 'accord'],
                'negative': ['non', 'no', 'pas'],
                'polite': ['excusez', 'pardon', 'desolee', 'sil vous plait']
            },
            'product_search': [
                "Voici notre sélection exclusive qui pourrait vous intéresser :",
                "J'ai trouvé ces créations Chanel parfaites pour vous :",
                "Découvrez ces pièces iconiques de notre collection :"
            ],
            'price_inquiry': [
                "Voici les informations tarifaires demandées :",
                "Nos prix pour ces créations :",
                "Budget nécessaire pour ces articles :"
            ],
            'stock_check': [
                "État actuel de nos stocks :",
                "Disponibilité en temps réel :",
                "Vérification de notre inventaire :"
            ],
            'style_recommendation': [
                "Mes conseils de style personnalisés :",
                "Je recommande pour votre occasion :",
                "L'alliance parfaite pour vous :"
            ],
            'care_instructions': [
                "Conseils d'entretien de nos experts :",
                "Pour préserver vos créations Chanel :",
                "Instructions de soin recommandées :"
            ]
        }
        
        # Détection contextuelle
        query_lower = query.lower()
        
        if prediction.intent == 'general_chat':
            # Détection fine du contexte
            context_detected = False
            
            for context, keywords in response_templates['general_chat'].items():
                if any(kw in query_lower for kw in keywords):
                    if context == 'greetings':
                        base_response = "Bonjour et bienvenue chez Chanel Maison Heritage ! Comment puis-je vous accompagner dans votre recherche ?"
                    elif context == 'thanks':
                        base_response = "Je vous en prie ! C'est un plaisir de vous conseiller. Puis-je vous aider avec autre chose ?"
                    elif context == 'goodbye':
                        base_response = "Au revoir et merci de votre visite ! N'hésitez pas à revenir pour découvrir nos nouvelles créations."
                    elif context == 'affirmative':
                        base_response = "Parfait ! Comment puis-je continuer à vous assister ?"
                    elif context == 'negative':
                        base_response = "D'accord, puis-je vous proposer autre chose ou vous renseigner différemment ?"
                    elif context == 'polite':
                        base_response = "Aucun souci ! Je suis là pour vous aider. Que puis-je faire pour vous ?"
                    context_detected = True
                    break
            
            if not context_detected:
                base_response = "Comment puis-je vous accompagner dans votre découverte de nos créations ?"
        
        elif prediction.intent == 'care_instructions':
            # Instructions d'entretien détaillées et spécifiques
            care_instructions = {
                'cuir': """
    **Entretien du cuir d'agneau Chanel :**
    • Nettoyage : Utilisez un chiffon doux légèrement humide
    • Protection : Appliquez un produit hydratant spécial cuir tous les 3 mois
    • Stockage : Conservez dans un dustbag à l'abri de la lumière
    • Évitez : L'eau, la chaleur directe et les produits chimiques
    • En cas de tache : Consultez immédiatement nos experts en boutique""",
                
                'chaussures': """
    **Entretien spécialisé pour chaussures Chanel :**
    • Nettoyage quotidien : Brossage délicat avec brosse en crin
    • Cirage : Utilisez un cirage de qualité adapté à la couleur
    • Embauchoirs : Insérez des embauchoirs en bois après chaque port
    • Rotation : Alternez vos paires pour laisser respirer le cuir
    • Réparation : Faites ressemeler chez un cordonnier spécialisé
    • Stockage : Conservez dans leur boîte d'origine avec papier de soie""",
                
                'sac': """
    **Entretien des sacs en cuir Chanel :**
    • Nettoyage mensuel avec un chiffon microfibre légèrement humide
    • Conditionnement du cuir tous les 3-4 mois avec un produit spécialisé
    • Stockage dans le dustbag fourni, rempli de papier pour garder la forme
    • Évitez la surcharge pour préserver la structure
    • Rotation des anses pour éviter l'usure inégale""",
                
                'tweed': """
    **Entretien du tweed Chanel :**
    • Nettoyage à sec uniquement chez un spécialiste textiles de luxe
    • Stockage sur cintre large pour préserver la forme
    • Brossage délicat dans le sens du tissu avec une brosse à vêtements
    • Protection anti-mites avec sachets naturels (lavande, cèdre)
    • Évitez l'exposition prolongée au soleil et à l'humidité""",
                
                'soie': """
    **Entretien de la soie :**
    • Nettoyage à sec professionnel obligatoire
    • Repassage à température douce (110°C max) avec un linge de protection
    • Stockage suspendu dans un endroit sec et aéré
    • Évitez les parfums, déodorants et produits acides""",
                
                'perles': """
    **Entretien des perles de culture :**
    • Nettoyage après chaque port avec un chiffon doux et sec
    • Évitez absolument le contact avec parfums, cosmétiques et transpiration
    • Stockage séparé des autres bijoux dans une pochette douce
    • Enfilage à vérifier annuellement chez un bijoutier"""
            }
            
            # Sélection du template de base
            templates = response_templates.get('care_instructions', ["Conseils d'entretien :"])
            base_response = random.choice(templates)
            
            # Détection intelligente du matériau/produit
            material_found = None
            
            if any(word in query_lower for word in ['chaussure', 'chaussures', 'escarpin', 'ballerine', 'soulier']):
                material_found = 'chaussures'
            elif any(word in query_lower for word in ['sac', 'sacoche', 'maroquinerie']):
                material_found = 'sac'
            elif 'cuir' in query_lower:
                material_found = 'cuir'
            elif 'tweed' in query_lower:
                material_found = 'tweed'
            elif any(word in query_lower for word in ['soie', 'satin']):
                material_found = 'soie'
            elif any(word in query_lower for word in ['perle', 'perles', 'bijou', 'collier']):
                material_found = 'perles'
            
            if material_found and material_found in care_instructions:
                base_response += care_instructions[material_found]
            else:
                base_response += """
    **Conseils d'entretien généraux pour vos créations Chanel :**
    • Suivez toujours les étiquettes d'entretien spécifiques
    • Privilégiez le nettoyage professionnel pour les pièces délicates
    • Stockez vos articles dans leurs dustbags d'origine
    • Évitez l'exposition directe à la lumière et à la chaleur
    • Consultez nos conseillers pour des instructions personnalisées"""
        
        elif prediction.intent == 'stock_check' and not products:
            # Sélection du template de base
            templates = response_templates.get('stock_check', ["État de nos stocks :"])
            base_response = random.choice(templates)
            
            # Gestion complète des stocks par localisation
            location_stocks = {
                'paris': {
                    'name': 'Paris Cambon',
                    'products': [
                        'Tailleur Tweed Iconique : 5 pièces',
                        'Robe Satin Asymétrique : 4 pièces', 
                        'Sac Matelassé Chaîne Dorée : 6 pièces',
                        'Escarpins Bout Rond : 8 pièces',
                        'Collier Perles Signature : 8 pièces',
                        'Ballerines Cuir Nude : 10 pièces',
                        'Manteau Bouclé Long : 3 pièces',
                        'Sac Cabas Grand Format : 4 pièces'
                    ]
                },
                'lyon': {
                    'name': 'Lyon Presqu\'île',
                    'products': [
                        'Tailleur Tweed Iconique : 3 pièces',
                        'Robe Satin Asymétrique : 3 pièces',
                        'Sac Matelassé Chaîne Dorée : 4 pièces',
                        'Escarpins Bout Rond : 6 pièces',
                        'Collier Perles Signature : 5 pièces',
                        'Ballerines Cuir Nude : 8 pièces',
                        'Manteau Bouclé Long : 2 pièces',
                        'Sac Cabas Grand Format : 3 pièces'
                    ]
                },
                'cannes': {
                    'name': 'Cannes Croisette',
                    'products': [
                        'Robe Satin Asymétrique : 3 pièces',
                        'Sac Matelassé Chaîne Dorée : 5 pièces',
                        'Escarpins Bout Rond : 4 pièces',
                        'Collier Perles Signature : 6 pièces',
                        'Ballerines Cuir Nude : 6 pièces',
                        'Tailleur Tweed Iconique : 2 pièces',
                        'Manteau Bouclé Long : 1 pièce'
                    ]
                }
            }
            
            detected_location = None
            for loc_key in location_stocks.keys():
                if loc_key in query_lower:
                    detected_location = loc_key
                    break
            
            if detected_location:
                stock_info = location_stocks[detected_location]
                base_response += f"""
    **Stock disponible à {stock_info['name']} :**

    Voici notre inventaire complet dans cette boutique :
    """
                for product in stock_info['products']:
                    base_response += f"• {product}\n"
                
                base_response += "\n📞 Pour réserver un article ou obtenir plus d'informations, contactez directement la boutique."
            else:
                base_response += """
    **Nos stocks sont mis à jour en temps réel dans nos 3 boutiques :**

    • **Paris Cambon** : Notre flagship avec la collection complète
    • **Lyon Presqu'île** : Sélection premium et nouveautés  
    • **Cannes Croisette** : Collection saisonnière et pièces exclusives

    💡 Précisez une boutique (Paris, Lyon ou Cannes) pour voir le détail des stocks."""
        
        else:
            # Pour toutes les autres intentions, utiliser les templates
            templates = response_templates.get(prediction.intent, ["Voici ce que j'ai trouvé pour vous :"])
            base_response = random.choice(templates)
        
        # Affichage enrichi des produits
        if products and prediction.intent in ['product_search', 'price_inquiry', 'style_recommendation']:
            base_response += "\n"
            
            for i, product in enumerate(products[:3], 1):
                # Informations essentielles
                name = product.get('name', 'Produit')
                price = product.get('price', 0)
                currency = product.get('currency', 'EUR')
                
                # Informations contextuelles
                materials = product.get('materials', [])
                colors = product.get('colors', [])
                occasions = product.get('occasions', [])
                
                # Score de pertinence
                similarity = product.get('similarity_score', 0)
                
                # Construction de la description
                product_line = f"\n{i}. **{name}** - {price} {currency}"
                
                # Ajout d'informations contextuelles
                details = []
                if materials:
                    details.append(f"Matières: {', '.join(materials[:2])}")
                if colors:
                    details.append(f"Couleurs: {', '.join(colors[:2])}")
                if occasions:
                    details.append(f"Occasions: {', '.join(occasions[:2])}")
                
                if details:
                    product_line += f"\n   📝 {' • '.join(details)}"
                
                # Indicateur de pertinence
                if similarity > 0.7:
                    product_line += " ⭐ *Excellent match*"
                elif similarity > 0.5:
                    product_line += " ✨ *Bon match*"
                
                base_response += product_line
        
        elif prediction.intent in ['product_search', 'style_recommendation'] and not products:
            # Cas où aucun produit n'est trouvé
            base_response += "\n\nJe n'ai pas trouvé de produit correspondant exactement à votre recherche."
            base_response += "\nPourriez-vous préciser vos préférences (couleur, occasion, budget) ?"
        
        # Informations contextuelles enrichies
        if prediction.entities:
            context_info = []
            
            # Entités importantes à mettre en valeur
            priority_entities = ['color', 'occasion', 'material', 'product_type', 'price']
            
            for entity_type in priority_entities:
                if entity_type in prediction.entities:
                    values = prediction.entities[entity_type]
                    entity_display = {
                        'color': '🎨 Couleur',
                        'occasion': '✨ Occasion', 
                        'material': '🧵 Matière',
                        'product_type': '👜 Type',
                        'price': '💰 Budget'
                    }
                    
                    display_name = entity_display.get(entity_type, entity_type.title())
                    context_info.append(f"{display_name}: {', '.join(values)}")
            
            if context_info:
                base_response += f"\n\n🔍 **Critères détectés:** {' • '.join(context_info)}"
        
        # Suggestions proactives
        if prediction.intent == 'product_search' and products:
            base_response += "\n\n💡 *Conseil: Dites-moi \"plus d'infos sur le produit 1\" pour obtenir des détails complets.*"
        elif prediction.intent == 'style_recommendation':
            base_response += "\n\n👗 *Astuce: Je peux aussi vous conseiller des accessoires assortis !*"
        elif prediction.intent == 'stock_check' and products:
            # Informations de stock si disponibles
            stock_info = []
            for product in products[:2]:
                stock_locations = product.get('stock_locations', {})
                if stock_locations:
                    total_stock = sum(stock_locations.values())
                    if total_stock > 0:
                        stock_info.append(f"{product['name']}: {total_stock} pièces disponibles")
                    else:
                        stock_info.append(f"{product['name']}: Sur commande")
            
            if stock_info:
                base_response += f"\n\n📦 **Stock:** {' • '.join(stock_info)}"
        
        # Appel à l'action minimal et discret
        confidence = prediction.confidence
    
        if confidence > 0.8:
            base_response += "\n\n❓ Y a-t-il autre chose que je puisse vous proposer ?"
        
        return base_response

    def save_model(self, filepath: str):
        """Sauvegarde robuste du modèle"""
        try:
            import pickle
            model_data = {
                'intent_classifier': self.intent_classifier,
                'vectorizer': self.intent_classifier.vectorizer,
                'label_encoder': self.intent_classifier.label_encoder,
                'products_df': self.semantic_search.products_df,
                'use_semantic': self.semantic_search.use_semantic
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Modèle sauvegardé: {filepath}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    def load_model(self, filepath: str):
        """Chargement robuste du modèle"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.intent_classifier = model_data['intent_classifier']
            self.semantic_search.products_df = model_data['products_df']
            self.semantic_search.use_semantic = model_data.get('use_semantic', False)
            
            print(f"✅ Modèle chargé: {filepath}")
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            raise

# Test principal 
if __name__ == "__main__":
    print("🎯 Validation croisée + Recherche hybride + Gestion d'erreurs")
    
    chatbot = OptimizedFashionChatbot()
    
    try:
        # Entraînement avec gestion d'erreur robuste
        print("\n📚 Phase d'entraînement...")
        results = chatbot.train_full_pipeline(
            'training_data.json',  # Chemins relatifs
            'products.json'
        )
        
        if results['status'] == 'optimized_success':
            print(f"\n🎉 ENTRAÎNEMENT RÉUSSI !")
            print(f"🎯 Précision CV: {results['intent_classification']['cv_accuracy']:.1%}")
            print(f"🔍 Produits indexés: {results['products_indexed']}")
            print(f"🔮 Recherche sémantique: {'✅ Activée' if results['semantic_search_enabled'] else '📝 Textuelle uniquement'}")
            
            # Tests variés avec gestion d'erreur
            test_queries = [
                "Bonjour !",
                "",  # Test cas vide
                "Je cherche un sac noir pour le bureau",
                "Avez-vous des chaussures confortables ?",
                "Combien coûte le tailleur tweed ?",
                "Stock disponible à Paris ?",
                "Que me conseillez-vous pour un mariage ?",
                "Comment entretenir le cuir d'agneau ?",
                "sdkfjhsdkjfh",  # Test cas incompréhensible
                "Merci beaucoup, au revoir !",
                "Des bijoux en perles ?"
            ]
            
            print("\n🧪 TESTS VARIÉS:")
            for i, query in enumerate(test_queries, 1):
                print(f"\n--- Test {i} ---")
                result = chatbot.process_query(query)
                
                if result['success']:
                    print(f"💬 '{query}'")
                    print(f"🎯 Intent: {result['intent']} (confiance: {result['confidence']:.1%})")
                    
                    if result['entities']:
                        print(f"🏷️  Entités: {result['entities']}")
                    
                    if result['recommended_products']:
                        print(f"🛍️  Produits trouvés: {len(result['recommended_products'])}")
                    
                    # Réponse tronquée pour l'affichage
                    response_preview = result['response'][:100]
                    if len(result['response']) > 100:
                        response_preview += "..."
                    print(f"💡 Réponse: {response_preview}")
                    
                    # Debug info
                    debug = result.get('debug_info', {})
                    print(f"🔧 Debug: Query processée='{debug.get('processed_query', 'N/A')[:30]}...', Produits={debug.get('products_found', 0)}")
                else:
                    print(f"❌ Erreur: {result.get('error', 'Inconnue')}")
                    print(f"🔄 Fallback: {result.get('fallback_response', 'Aucun')}")
            
            # Sauvegarde du modèle
            print(f"\n💾 Sauvegarde du modèle...")
            chatbot.save_model('fashion_chatbot_optimized.pkl')
            
        else:
            print(f"❌ ÉCHEC ENTRAÎNEMENT: {results.get('error', 'Erreur inconnue')}")
            
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()