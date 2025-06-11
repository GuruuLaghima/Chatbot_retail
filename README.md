# 🛍️ Chanel Maison Heritage - Chatbot IA pour Conseillers de Vente

## 📋 Vue d'ensemble

**Projet :** Assistant conversationnel intelligent pour conseillers de vente Chanel  
**Objectif :** Améliorer l'expérience client en magasin grâce à un chatbot basé sur ML/NLP  
**Technologies :** Python, FastAPI, scikit-learn, SentenceTransformers, TF-IDF

## 🎯 Problématique Business

Les conseillers de vente Chanel ont besoin d'un assistant intelligent pour :
- 🔍 **Rechercher rapidement** les produits selon les critères clients
- 📦 **Vérifier les stocks** en temps réel dans toutes les boutiques
- 💎 **Proposer des conseils de style** personnalisés selon l'occasion
- 🧽 **Fournir des instructions d'entretien** expertes pour chaque matière

## 🚀 Installation Rapide

### Prérequis
```bash
Python 3.8+
pip (gestionnaire de packages Python)
```

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Structure des fichiers requis
```
projet/
├── api_main.py                 # API FastAPI principale
├── ml_pipeline.py              # Pipeline ML optimisé
├── app.html                    # Interface web moderne
├── test_and_deploy.py          # Tests automatisés
├── ml_evaluation.py            # Évaluation ML détaillée
├── requirements.txt            # Dépendances Python
├── data/
│   ├── products.json           # Base de données produits
│   └── training_data.json      # Données d'entraînement ML
└── README.md                   # Ce fichier
```

## ⚡ Démarrage

### 1. Lancer l'API
```bash
python api_main.py
```
**Résultat attendu :**
```
INFO: Uvicorn running on http://0.0.0.0:8000
✅ API prête !
```

### 2. Accéder à l'interface
Ouvrir dans un navigateur : **http://localhost:8000**

### 3. Tests automatisés (optionnel)
```bash
# Dans un nouveau terminal
python test_and_deploy.py
```

### 4. Évaluation ML détaillée (optionnel)
```bash
python ml_evaluation.py
```

## 🧪 Tests Rapides

### Via l'interface web
1. Cliquer sur les exemples de questions dans la sidebar gauche
2. Ou taper directement :
   - "Je cherche un sac pour le bureau"
   - "Avez-vous du stock à Paris ?"
   - "Que me conseillez-vous pour un mariage ?"

### Via API REST
```bash
# Test basique
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Bonjour"}'

# Test recherche produit
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Je cherche un sac élégant"}'
```

## 🔧 Architecture Technique

### Pipeline ML
```
Input User → Preprocessing → TF-IDF → Classification → Entity Extraction → Response Generation
```

### API Architecture
```
Frontend (HTML/JS) ↔ FastAPI (Python) ↔ ML Pipeline ↔ JSON Database
```

### Composants principaux
- **FastAPI** : API REST moderne avec documentation automatique
- **scikit-learn** : Classification d'intentions (SVM, Random Forest, Logistic Regression)
- **SentenceTransformers** : Recherche sémantique multilingue
- **NLTK** : Preprocessing et tokenisation du français
- **JSON** : Base de données produits et d'entraînement

## 📊 Performances

### Métriques ML
- **Précision** : 86.2%
- **F1-Score** : 85.1%
- **Temps de réponse** : < 1 seconde
- **Grade performance** : A+

### Capacités
- **6 intentions** classifiées automatiquement
- **7 types d'entités** extraites (couleur, matière, occasion...)
- **8 produits** indexés avec recherche sémantique
- **3 boutiques** avec gestion de stocks en temps réel

## 🎯 Fonctionnalités

### Classification d'Intentions
- `product_search` : Recherche de produits
- `price_inquiry` : Demandes de prix
- `stock_check` : Vérification de stocks
- `style_recommendation` : Conseils de style
- `care_instructions` : Instructions d'entretien
- `general_chat` : Conversation générale

### Extraction d'Entités
- **Couleurs** : noir, blanc, rouge, beige...
- **Matières** : cuir, tweed, soie, satin...
- **Types de produits** : sac, chaussure, robe, tailleur...
- **Occasions** : bureau, soirée, mariage, cocktail...
- **Localisation** : Paris, Lyon, Cannes...

### Recherche Intelligente
- **Recherche sémantique** avec SentenceTransformers
- **Fallback textuel** pour robustesse
- **Scoring de pertinence** automatique
- **Recommandations contextuelles**

## 🛠️ Développement

### Ajouter de nouveaux produits
Modifier `data/products.json` avec la structure :
```json
{
  "id": "CMH999",
  "name": "Nouveau Produit",
  "category": "Catégorie",
  "price": 1500.00,
  "materials": ["Matière 1", "Matière 2"],
  "colors": ["Couleur 1", "Couleur 2"],
  "stock_locations": {
    "Paris Cambon": 5,
    "Lyon Presqu'île": 3
  }
}
```

### Enrichir l'entraînement
Ajouter des exemples dans `data/training_data.json` :
```json
{
  "id": 999,
  "text": "Nouvelle phrase d'exemple",
  "intent": "intention_correspondante"
}
```

---

