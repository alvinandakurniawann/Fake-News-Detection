# fake_news_detector.py
"""
Module untuk deteksi fake news menggunakan model TF-IDF + Logistic Regression
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

class TfidfLogregDetector:
    """Fake News Detection Model TF-IDF + Logistic Regression"""
    
    def __init__(self, model_path: str = None):
        """
        Inisialisasi model deteksi fake news TF-IDF + Logistic Regression
        
        Args:
            model_path: Path ke file model pipeline yang sudah di-training
        """
        self.model: Pipeline = None
        self.model_loaded = False
        self.model_info = {
            'name': 'TF-IDF + Logistic Regression',
            'type': 'tfidf',
            'path': None,
            'status': 'Not Loaded'
        }
        
        if model_path:
            self.load_model(model_path)
    
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model
        
        Returns:
            dict: Dictionary containing model information
        """
        return self.model_info
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a model from the specified path
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            loaded_model = joblib.load(model_path)
            
            # Validasi apakah ini scikit-learn Pipeline dengan step yang benar
            if not isinstance(loaded_model, Pipeline):
                raise TypeError("Model yang dimuat bukan scikit-learn Pipeline.")
            if 'tfidf' not in loaded_model.named_steps or 'logreg' not in loaded_model.named_steps:
                raise ValueError("Pipeline tidak memiliki step 'tfidf' atau 'logreg'.")

            self.model = loaded_model
            self.model_loaded = True
            self.model_info = {
                'name': 'TF-IDF + Logistic Regression',
                'type': 'tfidf',
                'path': model_path,
                'status': 'Loaded'
            }
            print(f"Model TF-IDF+LogReg berhasil dimuat dari: {model_path}")
            return True
        except Exception as e:
            print(f"Gagal memuat model TF-IDF+LogReg: {e}")
            self.model = None
            self.model_loaded = False
            self.model_info['status'] = f'Error: {str(e)}'
            return False
    
    @staticmethod
    def _class_indices(classes) -> tuple:
        """Petakan indeks probabilitas ke (FAKE, REAL) dari classes_
        model, bukan asumsi posisi. Konvensi latih: 1=FAKE, 0=REAL."""
        labels = list(classes)
        lowered = [str(c).lower() for c in labels]

        def find(want_num, want_names):
            if want_num in labels:
                return labels.index(want_num)
            for name in want_names:
                if name in lowered:
                    return lowered.index(name)
            return None

        idx_fake = find(1, ['fake'])
        idx_real = find(0, ['real'])
        if idx_fake is None or idx_real is None or idx_fake == idx_real:
            # Fallback terakhir: asumsi lama [REAL, FAKE]
            idx_fake, idx_real = 1, 0
        return idx_fake, idx_real

    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Melakukan prediksi teks menggunakan model TF-IDF+LogReg.
        
        Args:
            text: Teks atau list teks yang akan diprediksi (setelah preprocessing)
            
        Returns:
            Dict berisi hasil prediksi
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model TF-IDF+LogReg belum dimuat dengan benar")
        
        # Pastikan input berupa list
        texts = [text] if isinstance(text, str) else text
        
        try:
            # Lakukan prediksi menggunakan pipeline
            probabilities = self.model.predict_proba(texts)
            predictions = self.model.predict(texts)

            # Petakan indeks dari classes_ model (robust, bukan asumsi posisi)
            classes = list(self.model.named_steps['logreg'].classes_)
            idx_fake, idx_real = self._class_indices(classes)
            fake_label = classes[idx_fake]

            def scale(i):
                f = float(probabilities[i][idx_fake])
                r = float(probabilities[i][idx_real])
                return f, r

            # Format hasil
            if isinstance(text, str):
                # Single prediction
                fake_prob, real_prob = scale(0)
                prediction = 'FAKE' if predictions[0] == fake_label else 'REAL'

                # Get important words for this prediction
                explanation = self.explain_prediction(text)

                return {
                    'prediction': prediction,
                    'confidence': max(fake_prob, real_prob),
                    'probabilities': {
                        'FAKE': fake_prob,
                        'REAL': real_prob
                    },
                    'important_words': explanation.get('important_words', []),
                    'model_info': self.model_info
                }
            else:
                # Batch prediction
                results = []
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    fake_prob, real_prob = scale(i)

                    # Get important words for this prediction
                    explanation = self.explain_prediction(texts[i])

                    results.append({
                        'text': texts[i],
                        'prediction': 'FAKE' if pred == fake_label else 'REAL',
                        'confidence': max(fake_prob, real_prob),
                        'probabilities': {
                            'FAKE': fake_prob,
                            'REAL': real_prob
                        },
                        'important_words': explanation.get('important_words', []),
                        'model_info': self.model_info
                    })
                return results
            
        except Exception as e:
            error_msg = f"Error saat melakukan prediksi TF-IDF+LogReg: {str(e)}"
            print(error_msg)
            # Jika prediksi gagal, kembalikan hasil error atau nilai default
            if isinstance(text, str):
                return {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': {'FAKE': 0.0, 'REAL': 0.0},
                    'model_info': self.model_info,
                    'error': error_msg
                }
            else:
                return [{
                    'text': t,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': {'FAKE': 0.0, 'REAL': 0.0},
                    'model_info': self.model_info,
                    'error': error_msg
                } for t in texts]
    
    def explain_prediction(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Menghasilkan penjelasan prediksi menggunakan kontribusi kata dalam teks
        terhadap prediksi, berdasarkan bobot TF-IDF dan koefisien model LogReg.
        Hanya berlaku jika model dimuat adalah pipeline TF-IDF+LogReg.
        
        Args:
            text: Teks yang akan dijelaskan (setelah preprocessing)
            num_features: Jumlah fitur teratas yang akan ditampilkan
            
        Returns:
            Dict berisi kata-kata penting dan bobotnya
        """
        try:
            if not self.model_loaded or self.model is None:
                return {
                    'important_words': [],
                    'explanation_method': 'Model not loaded'
                }

            logreg_step = self.model.named_steps.get('logreg')
            tfidf_step = self.model.named_steps.get('tfidf')

            if logreg_step is None or not hasattr(logreg_step, 'coef_'):
                return {
                    'important_words': [],
                    'explanation_method': 'LogReg step not found or no coefficients'
                }
            
            if tfidf_step is None or not hasattr(tfidf_step, 'transform') or not hasattr(tfidf_step, 'get_feature_names_out'):
                return {
                    'important_words': [],
                    'explanation_method': 'TF-IDF step not found or does not support transform/get_feature_names_out'
                }

            # Transformasi teks menggunakan TF-IDF vectorizer
            text_tfidf = tfidf_step.transform([text]) # Input harus berupa list

            # Dapatkan koefisien model LogReg
            coefficients = logreg_step.coef_[0]
            feature_names = tfidf_step.get_feature_names_out()
            
            # Hitung skor kontribusi untuk setiap fitur dalam teks
            # Ini adalah perkalian element-wise antara TF-IDF teks dan koefisien global
            feature_scores = text_tfidf.multiply(coefficients).sum(axis=0).A1

            # Dapatkan indeks fitur terpenting (nilai absolut terbesar) dalam teks
            # Kita hanya peduli dengan fitur yang muncul di teks (yang memiliki nilai TF-IDF > 0)
            text_feature_indices = text_tfidf.indices # Indeks fitur yang ada di teks
            text_feature_scores = feature_scores[text_feature_indices] # Skor kontribusi untuk fitur di teks

            # Urutkan berdasarkan skor absolut
            sorted_indices_in_text_features = np.argsort(np.abs(text_feature_scores))[-num_features:][::-1]

            important_words = []
            for i in sorted_indices_in_text_features:
                original_feature_index = text_feature_indices[i]
                word = feature_names[original_feature_index]
                weight = float(text_feature_scores[i]) # Menggunakan skor kontribusi dari teks

                important_words.append({
                    'word': word,
                    'weight': weight,
                    'impact': 'fake' if weight > 0 else 'real'
                })
            
            return {
                'important_words': important_words,
                'explanation_method': 'Text-Specific Feature Contribution'
            }
            
        except Exception as e:
            print(f"Error saat membuat penjelasan TF-IDF+LogReg berbasis teks: {e}")
            # Kembalikan penjelasan kosong jika terjadi error
            return {
                'important_words': [],
                'explanation_method': f'Error: {str(e)}'
            }