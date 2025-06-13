import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve, validation_curve
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import du pipeline
from ml_pipeline import OptimizedFashionChatbot, OptimizedTextPreprocessor

class MLModelEvaluator:
    """Évaluation complète du modèle ML"""
    
    def __init__(self, chatbot_ml: OptimizedFashionChatbot):
        self.chatbot_ml = chatbot_ml
        self.evaluation_results = {}
    
    def evaluate_intent_classification(self, test_data: List[Dict]) -> Dict:
        """Évaluation détaillée de la classification d'intentions"""
        print("📊 Évaluation de la classification d'intentions...")
        
        predictions = []
        true_labels = []
        confidences = []
        processing_times = []
        
        for example in test_data:
            try:
                import time
                start_time = time.time()
                
                result = self.chatbot_ml.process_query(example['text'])
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                if result.get('success', True):
                    predictions.append(result['intent'])
                    true_labels.append(example['intent'])
                    confidences.append(result['confidence'])
                else:
                    # Gestion des erreurs
                    predictions.append('unknown')
                    true_labels.append(example['intent'])
                    confidences.append(0.0)
                    
            except Exception as e:
                print(f"⚠️  Erreur sur '{example['text']}': {e}")
                predictions.append('error')
                true_labels.append(example['intent'])
                confidences.append(0.0)
                processing_times.append(0.0)
        
        # Calcul des métriques avec gestion d'erreurs
        try:
            accuracy = accuracy_score(true_labels, predictions)
            
            # Rapport détaillé par classe
            detailed_report = classification_report(
                true_labels, predictions, 
                output_dict=True, 
                zero_division=0
            )
            
            # Matrice de confusion
            unique_labels = sorted(list(set(true_labels + predictions)))
            conf_matrix = confusion_matrix(
                true_labels, predictions, 
                labels=unique_labels
            )
            
        except Exception as e:
            print(f"❌ Erreur calcul métriques: {e}")
            accuracy = 0.0
            detailed_report = {}
            conf_matrix = np.array([[0]])
            unique_labels = []
        
        results = {
            'accuracy': accuracy,
            'mean_confidence': np.mean(confidences) if confidences else 0.0,
            'mean_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'detailed_report': detailed_report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': unique_labels,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'total_samples': len(test_data),
            'successful_predictions': sum(1 for p in predictions if p not in ['unknown', 'error'])
        }
        
        # Métriques supplémentaires
        if detailed_report and 'weighted avg' in detailed_report:
            results.update({
                'precision': detailed_report['weighted avg']['precision'],
                'recall': detailed_report['weighted avg']['recall'],
                'f1_score': detailed_report['weighted avg']['f1-score']
            })
        
        self.evaluation_results['intent_classification'] = results
        
        # Affichage des résultats
        print(f"   ✅ Précision: {accuracy:.3f}")
        print(f"   ✅ Confiance moyenne: {np.mean(confidences):.3f}")
        print(f"   ✅ Temps de traitement moyen: {np.mean(processing_times)*1000:.1f}ms")
        print(f"   ✅ Prédictions réussies: {results['successful_predictions']}/{len(test_data)}")
        
        return results
    
    def evaluate_entity_extraction(self, test_data: List[Dict]) -> Dict:
        """Évaluation de l'extraction d'entités"""
        print("🏷️  Évaluation de l'extraction d'entités...")
        
        extracted_entities = []
        entity_stats = {}
        
        for example in test_data:
            try:
                # Utilisation du preprocessor optimisé
                preprocessor = self.chatbot_ml.intent_classifier.preprocessor
                entities = preprocessor.extract_entities(example['text'])
                extracted_entities.append(entities)
                
                # Statistiques par type d'entité
                for entity_type, values in entities.items():
                    if entity_type not in entity_stats:
                        entity_stats[entity_type] = 0
                    entity_stats[entity_type] += len(values)
                    
            except Exception as e:
                print(f"⚠️  Erreur extraction entités pour '{example['text']}': {e}")
                extracted_entities.append({})
        
        # Analyse des entités trouvées
        total_entities = sum(len(e) for e in extracted_entities)
        entity_types_found = list(set(
            entity_type for entities in extracted_entities 
            for entity_type in entities.keys()
        ))
        
        results = {
            'total_entities_extracted': total_entities,
            'entity_types_found': entity_types_found,
            'entity_statistics': entity_stats,
            'entities_per_query': total_entities / len(test_data) if test_data else 0,
            'coverage': len(entity_types_found),
            'extracted_entities': extracted_entities
        }
        
        self.evaluation_results['entity_extraction'] = results
        
        print(f"   ✅ Entités extraites: {total_entities}")
        print(f"   ✅ Types d'entités: {len(entity_types_found)}")
        print(f"   ✅ Entités par requête: {results['entities_per_query']:.1f}")
        
        return results
    
    def evaluate_semantic_search(self, test_queries: List[str]) -> Dict:
        """Évaluation de la recherche sémantique"""
        print("🔍 Évaluation de la recherche sémantique...")
        
        search_results = []
        relevance_scores = []
        search_types = []
        response_times = []
        
        for query in test_queries:
            try:
                import time
                start_time = time.time()
                
                # Utilisation de l'API de recherche optimisée
                products = self.chatbot_ml.semantic_search.search(query, top_k=5)
                
                processing_time = time.time() - start_time
                response_times.append(processing_time)
                
                search_results.append(products)
                
                # Score de pertinence moyen
                if products:
                    avg_similarity = np.mean([p.get('similarity_score', 0) for p in products])
                    relevance_scores.append(avg_similarity)
                else:
                    relevance_scores.append(0.0)
                
                # Type de recherche utilisé
                search_type = "semantic" if self.chatbot_ml.semantic_search.use_semantic else "textual"
                search_types.append(search_type)
                
            except Exception as e:
                print(f"⚠️  Erreur recherche pour '{query}': {e}")
                search_results.append([])
                relevance_scores.append(0.0)
                search_types.append("error")
                response_times.append(0.0)
        
        results = {
            'mean_relevance_score': np.mean(relevance_scores) if relevance_scores else 0,
            'total_searches': len(test_queries),
            'successful_searches': len([r for r in search_results if r]),
            'average_results_per_query': np.mean([len(r) for r in search_results]),
            'min_similarity': min(relevance_scores) if relevance_scores else 0,
            'max_similarity': max(relevance_scores) if relevance_scores else 0,
            'mean_response_time': np.mean(response_times) if response_times else 0,
            'search_type_distribution': {
                'semantic': search_types.count('semantic'),
                'textual': search_types.count('textual'),
                'error': search_types.count('error')
            },
            'semantic_search_enabled': self.chatbot_ml.semantic_search.use_semantic
        }
        
        self.evaluation_results['semantic_search'] = results
        
        print(f"   ✅ Score de pertinence moyen: {results['mean_relevance_score']:.3f}")
        print(f"   ✅ Recherches réussies: {results['successful_searches']}/{len(test_queries)}")
        print(f"   ✅ Type de recherche: {'Sémantique' if results['semantic_search_enabled'] else 'Textuelle'}")
        
        return results
    
    def benchmark_performance(self, test_queries: List[str], runs: int = 3) -> Dict:
        """Benchmark de performance avec plusieurs runs"""
        print("⚡ Benchmark de performance...")
        
        all_response_times = []
        memory_usage = []
        
        for run in range(runs):
            print(f"   🔄 Run {run + 1}/{runs}")
            run_times = []
            
            for query in test_queries:
                try:
                    import time
                    import psutil
                    import os
                    
                    # Mesure mémoire avant
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    start_time = time.time()
                    self.chatbot_ml.process_query(query)
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    run_times.append(response_time)
                    
                    # Mesure mémoire après
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_usage.append(memory_after - memory_before)
                    
                except Exception as e:
                    print(f"⚠️  Erreur benchmark: {e}")
                    run_times.append(float('inf'))
                    memory_usage.append(0)
            
            all_response_times.extend(run_times)
        
        # Filtrer les valeurs infinies
        valid_times = [t for t in all_response_times if t != float('inf')]
        
        results = {
            'mean_response_time': np.mean(valid_times) if valid_times else 0,
            'median_response_time': np.median(valid_times) if valid_times else 0,
            'max_response_time': np.max(valid_times) if valid_times else 0,
            'min_response_time': np.min(valid_times) if valid_times else 0,
            'std_response_time': np.std(valid_times) if valid_times else 0,
            'queries_per_second': 1 / np.mean(valid_times) if valid_times and np.mean(valid_times) > 0 else 0,
            'total_queries_tested': len(test_queries) * runs,
            'successful_queries': len(valid_times),
            'average_memory_impact': np.mean(memory_usage) if memory_usage else 0,
            'performance_grade': self._calculate_performance_grade(valid_times)
        }
        
        self.evaluation_results['performance'] = results
        
        print(f"   ✅ Temps de réponse moyen: {results['mean_response_time']*1000:.1f}ms")
        print(f"   ✅ Requêtes par seconde: {results['queries_per_second']:.1f}")
        print(f"   ✅ Grade de performance: {results['performance_grade']}")
        
        return results
    
    def _calculate_performance_grade(self, response_times: List[float]) -> str:
        """Calcul du grade de performance"""
        if not response_times:
            return "F"
        
        avg_time = np.mean(response_times)
        
        if avg_time < 0.5:
            return "A+"
        elif avg_time < 1.0:
            return "A"
        elif avg_time < 2.0:
            return "B"
        elif avg_time < 3.0:
            return "C"
        elif avg_time < 5.0:
            return "D"
        else:
            return "F"
    
    def generate_evaluation_report(self) -> str:
        """Génération d'un rapport d'évaluation complet"""
        report = "# 📋 Rapport d'Évaluation ML - Chatbot Fashion Optimisé\n\n"
        report += f"*Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        if 'intent_classification' in self.evaluation_results:
            ic = self.evaluation_results['intent_classification']
            report += f"## 🤖 Classification d'Intentions\n"
            report += f"- **Précision globale**: {ic['accuracy']:.3f} ({ic['accuracy']*100:.1f}%)\n"
            report += f"- **F1-Score**: {ic.get('f1_score', 0):.3f}\n"
            report += f"- **Précision pondérée**: {ic.get('precision', 0):.3f}\n"
            report += f"- **Rappel pondéré**: {ic.get('recall', 0):.3f}\n"
            report += f"- **Confiance moyenne**: {ic['mean_confidence']:.3f}\n"
            report += f"- **Temps de traitement moyen**: {ic['mean_processing_time']*1000:.1f}ms\n"
            report += f"- **Échantillons testés**: {ic['total_samples']}\n"
            report += f"- **Prédictions réussies**: {ic['successful_predictions']}/{ic['total_samples']} ({ic['successful_predictions']/ic['total_samples']*100:.1f}%)\n\n"
        
        if 'entity_extraction' in self.evaluation_results:
            ee = self.evaluation_results['entity_extraction']
            report += f"## 🏷️ Extraction d'Entités\n"
            report += f"- **Entités extraites**: {ee['total_entities_extracted']}\n"
            report += f"- **Types d'entités**: {len(ee['entity_types_found'])}\n"
            report += f"- **Entités par requête**: {ee['entities_per_query']:.1f}\n"
            report += f"- **Types détectés**: {', '.join(ee['entity_types_found'])}\n\n"
            
            if ee['entity_statistics']:
                report += f"### Statistiques par type:\n"
                for entity_type, count in ee['entity_statistics'].items():
                    report += f"- **{entity_type}**: {count} occurrences\n"
                report += "\n"
        
        if 'semantic_search' in self.evaluation_results:
            ss = self.evaluation_results['semantic_search']
            report += f"## 🔍 Recherche Sémantique\n"
            report += f"- **Type de recherche**: {'Sémantique' if ss['semantic_search_enabled'] else 'Textuelle uniquement'}\n"
            report += f"- **Score de pertinence moyen**: {ss['mean_relevance_score']:.3f}\n"
            report += f"- **Recherches réussies**: {ss['successful_searches']}/{ss['total_searches']} ({ss['successful_searches']/ss['total_searches']*100:.1f}%)\n"
            report += f"- **Résultats moyens par requête**: {ss['average_results_per_query']:.1f}\n"
            report += f"- **Temps de réponse moyen**: {ss['mean_response_time']*1000:.1f}ms\n\n"
        
        if 'performance' in self.evaluation_results:
            perf = self.evaluation_results['performance']
            report += f"## ⚡ Performance\n"
            report += f"- **Grade de performance**: {perf['performance_grade']}\n"
            report += f"- **Temps de réponse moyen**: {perf['mean_response_time']*1000:.1f}ms\n"
            report += f"- **Temps de réponse médian**: {perf['median_response_time']*1000:.1f}ms\n"
            report += f"- **Requêtes par seconde**: {perf['queries_per_second']:.1f}\n"
            report += f"- **Écart-type**: {perf['std_response_time']*1000:.1f}ms\n"
            report += f"- **Impact mémoire moyen**: {perf['average_memory_impact']:.1f}MB\n"
            report += f"- **Requêtes testées**: {perf['total_queries_tested']}\n"
            report += f"- **Requêtes réussies**: {perf['successful_queries']}\n\n"
        
        # Recommandations basées sur les résultats
        report += "## 💡 Recommandations d'Optimisation\n"
        
        if 'intent_classification' in self.evaluation_results:
            accuracy = self.evaluation_results['intent_classification']['accuracy']
            if accuracy < 0.8:
                report += "- ⚠️  **Précision faible**: Augmenter les données d'entraînement ou ajuster les hyperparamètres\n"
            elif accuracy > 0.9:
                report += "- ✅ **Excellente précision**: Modèle bien calibré\n"
        
        if 'performance' in self.evaluation_results:
            grade = self.evaluation_results['performance']['performance_grade']
            if grade in ['D', 'F']:
                report += "- ⚠️  **Performance lente**: Optimiser le preprocessing ou utiliser un cache\n"
            elif grade in ['A+', 'A']:
                report += "- ✅ **Performance excellente**: Temps de réponse optimal\n"
        
        report += "\n---\n*Rapport généré par MLModelEvaluator v2.0*"
        
        return report
    
    def plot_confusion_matrix(self, save_path: str = None, figsize: tuple = (12, 10)):
        """Visualisation améliorée de la matrice de confusion"""
        if 'intent_classification' not in self.evaluation_results:
            print("❌ Pas de données de classification disponibles")
            return
        
        ic = self.evaluation_results['intent_classification']
        conf_matrix = np.array(ic['confusion_matrix'])
        class_names = ic['class_names']
        
        # Gestion des matrices vides
        if conf_matrix.size == 0 or len(class_names) == 0:
            print("❌ Matrice de confusion vide")
            return
        
        plt.figure(figsize=figsize)
        
        # Normalisation pour affichage en pourcentages
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
        
        # Heatmap
        sns.heatmap(
            conf_matrix_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Proportion normalisée'}
        )
        
        plt.title('Matrice de Confusion Normalisée - Classification d\'Intentions\n' + 
                 f'Précision globale: {ic["accuracy"]:.3f}', fontsize=14, fontweight='bold')
        plt.xlabel('Prédictions', fontsize=12)
        plt.ylabel('Vraies étiquettes', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Matrice de confusion sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_performance_metrics(self, save_path: str = None, figsize: tuple = (16, 12)):
        """Visualisation complète des métriques de performance"""
        if not self.evaluation_results:
            print("❌ Pas de données d'évaluation disponibles")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Métriques de Performance - Chatbot Fashion IA', fontsize=16, fontweight='bold')
        
        # 1. Métriques de classification
        if 'intent_classification' in self.evaluation_results:
            ic = self.evaluation_results['intent_classification']
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [
                ic['accuracy'], 
                ic.get('precision', 0), 
                ic.get('recall', 0), 
                ic.get('f1_score', 0)
            ]
            
            bars = axes[0, 0].bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[0, 0].set_title('Métriques de Classification')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_ylabel('Score')
            
            # Ajout des valeurs sur les barres
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Distribution des temps de réponse
        if 'performance' in self.evaluation_results:
            perf = self.evaluation_results['performance']
            times = ['Moyen', 'Médian', 'Min', 'Max']
            values = [
                perf['mean_response_time'] * 1000,
                perf['median_response_time'] * 1000,
                perf['min_response_time'] * 1000,
                perf['max_response_time'] * 1000
            ]
            
            bars = axes[0, 1].bar(times, values, color=['#ff7f0e', '#2ca02c', '#17becf', '#e377c2'])
            axes[0, 1].set_title('Temps de Réponse (ms)')
            axes[0, 1].set_ylabel('Temps (ms)')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                               f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 3. Distribution des confiances
        if 'intent_classification' in self.evaluation_results:
            ic = self.evaluation_results['intent_classification']
            confidences = ic.get('confidences', [])
            
            if confidences:
                axes[0, 2].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 2].axvline(np.mean(confidences), color='red', linestyle='--', 
                                  label=f'Moyenne: {np.mean(confidences):.3f}')
                axes[0, 2].set_title('Distribution des Scores de Confiance')
                axes[0, 2].set_xlabel('Score de Confiance')
                axes[0, 2].set_ylabel('Fréquence')
                axes[0, 2].legend()
        
        # 4. Métriques de recherche
        if 'semantic_search' in self.evaluation_results:
            ss = self.evaluation_results['semantic_search']
            
            search_metrics = ['Score Pertinence', 'Résultats/Requête', 'Taux Succès']
            search_values = [
                ss['mean_relevance_score'],
                ss['average_results_per_query'] / 5,  # Normalisation sur 5 max
                ss['successful_searches'] / ss['total_searches']
            ]
            
            bars = axes[1, 0].bar(search_metrics, search_values, color=['#bcbd22', '#17becf', '#ff7f0e'])
            axes[1, 0].set_title('Métriques de Recherche')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_ylabel('Score Normalisé')
            
            for bar, value in zip(bars, search_values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Extraction d'entités
        if 'entity_extraction' in self.evaluation_results:
            ee = self.evaluation_results['entity_extraction']
            entity_stats = ee.get('entity_statistics', {})
            
            if entity_stats:
                entity_types = list(entity_stats.keys())
                entity_counts = list(entity_stats.values())
                
                bars = axes[1, 1].bar(entity_types, entity_counts, color='lightcoral')
                axes[1, 1].set_title('Entités Extraites par Type')
                axes[1, 1].set_ylabel('Nombre d\'Occurrences')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                for bar, count in zip(bars, entity_counts):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(entity_counts) * 0.01,
                                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Grade de performance
        if 'performance' in self.evaluation_results:
            perf = self.evaluation_results['performance']
            grade = perf['performance_grade']
            
            # Couleurs selon le grade
            grade_colors = {
                'A+': '#2ca02c', 'A': '#8cc665', 'B': '#ffcc02', 
                'C': '#ff8c00', 'D': '#ff6b6b', 'F': '#d62728'
            }
            
            axes[1, 2].bar(['Performance'], [1], color=grade_colors.get(grade, '#gray'))
            axes[1, 2].set_title(f'Grade de Performance: {grade}')
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].set_ylabel('Grade')
            axes[1, 2].text(0, 0.5, grade, ha='center', va='center', 
                           fontsize=24, fontweight='bold', color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Graphiques de performance sauvegardés: {save_path}")
        
        plt.show()
    
    def plot_learning_curve(self, save_path: str = None):
        """Courbe d'apprentissage du modèle"""
        print("📈 Génération de la courbe d'apprentissage...")
        
        try:
            # Accès aux données d'entraînement
            if not hasattr(self.chatbot_ml.intent_classifier, 'vectorizer') or not self.chatbot_ml.intent_classifier.is_trained:
                print("❌ Modèle non entraîné, impossible de générer la courbe d'apprentissage")
                return
            
            # Rechargement des données d'entraînement
            training_data_path = 'training_data.json'
            try:
                with open(training_data_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)['training_data']
            except FileNotFoundError:
                print("❌ Fichier training_data.json non trouvé")
                return
            
            # Préparation des données
            X_text, y = self.chatbot_ml.intent_classifier.prepare_data(training_data)
            y_encoded = self.chatbot_ml.intent_classifier.label_encoder.transform(y)
            X_vectorized = self.chatbot_ml.intent_classifier.vectorizer.transform(X_text)
            
            # Génération de la courbe d'apprentissage
            train_sizes, train_scores, val_scores = learning_curve(
                self.chatbot_ml.intent_classifier.best_model,
                X_vectorized, y_encoded,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Calcul des moyennes et écarts-types
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)
            val_scores_std = np.std(val_scores, axis=1)
            
            # Visualisation
            plt.figure(figsize=(12, 8))
            
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                           val_scores_mean + val_scores_std, alpha=0.1, color="g")
            
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
            plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Score de validation")
            
            plt.title('Courbe d\'Apprentissage - Classification d\'Intentions', fontsize=14, fontweight='bold')
            plt.xlabel('Nombre d\'échantillons d\'entraînement')
            plt.ylabel('Score de précision')
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            
            # Annotations
            final_train_score = train_scores_mean[-1]
            final_val_score = val_scores_mean[-1]
            plt.annotate(f'Train final: {final_train_score:.3f}', 
                        xy=(train_sizes[-1], final_train_score),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
            plt.annotate(f'Val final: {final_val_score:.3f}', 
                        xy=(train_sizes[-1], final_val_score),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📈 Courbe d'apprentissage sauvegardée: {save_path}")
            
            plt.show()
            
            # Analyse de la courbe
            print(f"📊 Analyse de la courbe d'apprentissage:")
            print(f"   - Score final d'entraînement: {final_train_score:.3f}")
            print(f"   - Score final de validation: {final_val_score:.3f}")
            
            gap = final_train_score - final_val_score
            if gap > 0.1:
                print(f"   ⚠️  Surapprentissage détecté (écart: {gap:.3f})")
            elif gap < 0.05:
                print(f"   ✅ Bon équilibre biais-variance (écart: {gap:.3f})")
            
        except Exception as e:
            print(f"❌ Erreur génération courbe d'apprentissage: {e}")

class ABTesting:
    """Test A/B pour comparer différentes configurations """
    
    def __init__(self):
        self.test_results = {}
    
    def compare_preprocessing_methods(self, test_queries: List[str]) -> Dict:
        """Comparaison de différentes méthodes de preprocessing"""
        print("🔄 Comparaison des méthodes de preprocessing...")
        
        # Méthode 1: Preprocessing optimisé actuel
        preprocessor_optimized = OptimizedTextPreprocessor()
        
        # Méthode 2: Preprocessing basique 
        class BasicPreprocessor:
            def preprocess_for_ml(self, text: str) -> str:
                import re
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                return ' '.join(text.split())
        
        preprocessor_basic = BasicPreprocessor()
        
        results = {
            'optimized': [],
            'basic': []
        }
        
        for query in test_queries:
            # Test preprocessing optimisé
            processed_opt = preprocessor_optimized.preprocess_for_ml(query)
            entities_opt = preprocessor_optimized.extract_entities(query)
            
            results['optimized'].append({
                'original': query,
                'processed': processed_opt,
                'entities_count': sum(len(v) for v in entities_opt.values()),
                'processed_length': len(processed_opt.split())
            })
            
            # Test preprocessing basique
            processed_basic = preprocessor_basic.preprocess_for_ml(query)
            
            results['basic'].append({
                'original': query,
                'processed': processed_basic,
                'entities_count': 0,  # Pas d'extraction d'entités
                'processed_length': len(processed_basic.split())
            })
        
        # Analyse comparative
        opt_avg_entities = np.mean([r['entities_count'] for r in results['optimized']])
        opt_avg_length = np.mean([r['processed_length'] for r in results['optimized']])
        basic_avg_length = np.mean([r['processed_length'] for r in results['basic']])
        
        comparison = {
            'optimized_method': {
                'avg_entities_extracted': opt_avg_entities,
                'avg_processed_length': opt_avg_length,
                'features': ['Extraction d\'entités', 'Préservation mots fashion', 'Normalisation française']
            },
            'basic_method': {
                'avg_entities_extracted': 0,
                'avg_processed_length': basic_avg_length,
                'features': ['Nettoyage basique uniquement']
            },
            'recommendation': 'optimized' if opt_avg_entities > 0 else 'basic',
            'improvement_factor': opt_avg_entities / max(basic_avg_length / opt_avg_length, 1)
        }
        
        print(f"   ✅ Entités extraites (optimisé): {opt_avg_entities:.1f} par requête")
        print(f"   ✅ Longueur moyenne (optimisé): {opt_avg_length:.1f} mots")
        print(f"   ✅ Longueur moyenne (basique): {basic_avg_length:.1f} mots")
        
        return comparison

def comprehensive_evaluation():
    """Évaluation complète du système ML"""
    print("🎯 ÉVALUATION COMPLÈTE DU SYSTÈME ML ")
    print("=" * 60)
    
    # Chargement du modèle
    chatbot_ml = OptimizedFashionChatbot()
    
    try:
        # Tentative de chargement d'un modèle pré-entraîné
        try:
            chatbot_ml.load_model('fashion_chatbot_optimized.pkl')
            print("✅ Modèle pré-entraîné chargé")
        except FileNotFoundError:
            print("⚠️  Modèle non trouvé, entraînement en cours...")
            results = chatbot_ml.train_full_pipeline(
                'training_data.json',
                'products.json'
            )
            if results['status'] != 'optimized_success':
                print("❌ Échec de l'entraînement")
                return
            print("✅ Modèle entraîné avec succès")
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement/entraînement: {e}")
        return
    
    # Initialisation de l'évaluateur
    evaluator = MLModelEvaluator(chatbot_ml)
    
    # Données de test enrichies et équilibrées
    test_data = [
        # Product search
        {"text": "Je cherche un tailleur noir pour le bureau", "intent": "product_search"},
        {"text": "Avez-vous des sacs en cuir disponibles ?", "intent": "product_search"},
        {"text": "Montrez-moi vos chaussures élégantes", "intent": "product_search"},
        {"text": "Je voudrais voir des bijoux en perles", "intent": "product_search"},
        {"text": "Recherche manteau pour l'hiver", "intent": "product_search"},
        
        # Stock check
        {"text": "Y a-t-il du stock à Paris ?", "intent": "stock_check"},
        {"text": "Le tailleur est-il disponible ?", "intent": "stock_check"},
        {"text": "Vérifiez la disponibilité du sac matelassé", "intent": "stock_check"},
        {"text": "Stock boutique Lyon ?", "intent": "stock_check"},
        
        # Price inquiry
        {"text": "Combien coûte la robe satin ?", "intent": "price_inquiry"},
        {"text": "Quel est le prix du tailleur tweed ?", "intent": "price_inquiry"},
        {"text": "Tarif des escarpins ?", "intent": "price_inquiry"},
        {"text": "Budget nécessaire pour un ensemble complet", "intent": "price_inquiry"},
        
        # Style recommendation
        {"text": "Quelle robe pour un mariage ?", "intent": "style_recommendation"},
        {"text": "Que me conseillez-vous pour un cocktail ?", "intent": "style_recommendation"},
        {"text": "Tenue appropriée pour le bureau", "intent": "style_recommendation"},
        {"text": "Comment assortir ce sac ?", "intent": "style_recommendation"},
        
        # Care instructions
        {"text": "Comment nettoyer le cuir d'agneau ?", "intent": "care_instructions"},
        {"text": "Entretien du tweed conseillé", "intent": "care_instructions"},
        {"text": "Instructions lavage soie", "intent": "care_instructions"},
        {"text": "Comment préserver les perles ?", "intent": "care_instructions"},
        
        # General chat
        {"text": "Bonjour, comment allez-vous ?", "intent": "general_chat"},
        {"text": "Merci beaucoup pour votre aide", "intent": "general_chat"},
        {"text": "Au revoir et bonne journée", "intent": "general_chat"},
        {"text": "Parfait, très bien", "intent": "general_chat"},
        
        # Cas difficiles
        {"text": "sac bureau noir cuir", "intent": "product_search"},  # Style télégraphique
        {"text": "prix????", "intent": "price_inquiry"},  # Ponctuation excessive
        {"text": "je cherche quelque chose d'élégant", "intent": "style_recommendation"},  # Vague
        {"text": "", "intent": "general_chat"},  # Vide
    ]
    
    test_queries = [
        "sac noir bureau", "chaussures confortables", "robe soirée élégante",
        "tailleur classique", "bijoux perles", "manteau hiver", "accessoires mode",
        "collection printemps", "style parisien", "luxe français"
    ]
    
    # Évaluations détaillées
    print("\n" + "="*50)
    print("🤖 CLASSIFICATION D'INTENTIONS")
    intent_results = evaluator.evaluate_intent_classification(test_data)
    
    print("\n" + "="*50)
    print("🏷️  EXTRACTION D'ENTITÉS")
    entity_results = evaluator.evaluate_entity_extraction(test_data)
    
    print("\n" + "="*50)
    print("🔍 RECHERCHE SÉMANTIQUE")
    search_results = evaluator.evaluate_semantic_search(test_queries)
    
    print("\n" + "="*50)
    print("⚡ BENCHMARK DE PERFORMANCE")
    perf_results = evaluator.benchmark_performance(test_queries, runs=2)
    
    # Test A/B du preprocessing
    print("\n" + "="*50)
    print("🔄 TEST A/B PREPROCESSING")
    ab_tester = ABTesting()
    preprocessing_comparison = ab_tester.compare_preprocessing_methods(test_queries[:5])
    
    # Génération du rapport complet
    print("\n" + "="*50)
    print("📄 GÉNÉRATION DU RAPPORT")
    report = evaluator.generate_evaluation_report()
    
    # Ajout des résultats A/B au rapport
    report += "\n## 🔄 Test A/B Preprocessing\n"
    report += f"- **Méthode recommandée**: {preprocessing_comparison['recommendation']}\n"
    report += f"- **Entités extraites (optimisé)**: {preprocessing_comparison['optimized_method']['avg_entities_extracted']:.1f} par requête\n"
    report += f"- **Amélioration**: x{preprocessing_comparison['improvement_factor']:.1f}\n\n"
    
    print(report)
    
    # Sauvegarde du rapport
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'ml_evaluation_report_{timestamp}.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"📄 Rapport ML sauvegardé: {report_file}")
    
    # Génération des visualisations
    print("\n📊 GÉNÉRATION DES VISUALISATIONS...")
    try:
        evaluator.plot_confusion_matrix(f'confusion_matrix_{timestamp}.png')
        evaluator.plot_performance_metrics(f'performance_metrics_{timestamp}.png')
        evaluator.plot_learning_curve(f'learning_curve_{timestamp}.png')
        print("✅ Toutes les visualisations générées")
    except Exception as e:
        print(f"⚠️  Erreur génération visualisations: {e}")
    
    # Sauvegarde des résultats JSON
    all_results = {
        'timestamp': timestamp,
        'intent_classification': intent_results,
        'entity_extraction': entity_results,
        'semantic_search': search_results,
        'performance': perf_results,
        'preprocessing_comparison': preprocessing_comparison,
        'model_info': {
            'best_model': chatbot_ml.intent_classifier.best_model_name if chatbot_ml.intent_classifier.is_trained else None,
            'semantic_search_enabled': chatbot_ml.semantic_search.use_semantic,
            'training_samples': len(test_data)
        }
    }
    
    results_file = f'ml_evaluation_results_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"📊 Résultats JSON sauvegardés: {results_file}")
    
    # Résumé final
    print("\n" + "="*60)
    print("🏆 RÉSUMÉ DE L'ÉVALUATION ML")
    print("="*60)
    
    if intent_results:
        print(f"🤖 Classification: {intent_results['accuracy']:.1%} de précision")
        print(f"   └─ Confiance moyenne: {intent_results['mean_confidence']:.3f}")
        print(f"   └─ Temps moyen: {intent_results['mean_processing_time']*1000:.1f}ms")
    
    if search_results:
        search_type = "Sémantique" if search_results['semantic_search_enabled'] else "Textuelle"
        print(f"🔍 Recherche {search_type}: {search_results['mean_relevance_score']:.3f} score de pertinence")
        print(f"   └─ Succès: {search_results['successful_searches']}/{search_results['total_searches']}")
    
    if perf_results:
        print(f"⚡ Performance: Grade {perf_results['performance_grade']}")
        print(f"   └─ {perf_results['mean_response_time']*1000:.1f}ms de temps de réponse moyen")
        print(f"   └─ {perf_results['queries_per_second']:.1f} requêtes/seconde")
    
    if entity_results:
        print(f"🏷️  Entités: {entity_results['total_entities_extracted']} extraites")
        print(f"   └─ {len(entity_results['entity_types_found'])} types différents")
    
    # Score global
    global_score = 0
    max_score = 0
    
    if intent_results:
        global_score += intent_results['accuracy'] * 40  
        max_score += 40
    
    if search_results:
        search_score = search_results['successful_searches'] / search_results['total_searches']
        global_score += search_score * 30  
        max_score += 30
    
    if perf_results:
        perf_score = 1.0 if perf_results['performance_grade'] in ['A+', 'A'] else 0.7 if perf_results['performance_grade'] == 'B' else 0.5
        global_score += perf_score * 30  
        max_score += 30
    
    final_score = global_score / max_score if max_score > 0 else 0
    
    print(f"\n🎯 SCORE GLOBAL: {final_score:.1%}")
    
    
    return all_results

if __name__ == "__main__":
    # Configuration pour évaluation ML spécialisée
    print("🎯 Analyse détaillée des performances ML/NLP")
    print("="*60)
    
    try:
        # Lancement de l'évaluation complète
        evaluation_results = comprehensive_evaluation()
        
        print("\n🎉 ÉVALUATION ML :")
        print("   📄 ml_evaluation_report_[timestamp].md")
        print("   📊 ml_evaluation_results_[timestamp].json") 
        print("   📈 confusion_matrix_[timestamp].png")
        print("   📊 performance_metrics_[timestamp].png")
        print("   📉 learning_curve_[timestamp].png")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()