# utils.py
"""
Module untuk fungsi-fungsi utility
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from datetime import datetime


def process_batch_urls(urls: List[str], extractor, preprocessor, detector,
                      preprocessing_steps: List[str]) -> List[Dict[str, Any]]:
    """Process multiple URLs concurrently"""
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit extraction tasks
        future_to_url = {
            executor.submit(extractor.extract_from_url, url): url 
            for url in urls
        }
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                extraction_result = future.result()
                
                if extraction_result['success']:
                    # Preprocess text
                    full_text = f"{extraction_result['title']} {extraction_result['content']}"
                    processed_text = preprocessor.preprocess_pipeline(full_text, preprocessing_steps)
                    
                    # Predict
                    prediction_result = detector.predict(processed_text)
                    
                    # Get important words if available
                    important_words = []
                    if 'important_words' in prediction_result:
                        important_words = prediction_result['important_words']
                    
                    # Combine results (kunci probabilitas uppercase;
                    # .get ganda agar tahan format lama bila model diganti)
                    probs = prediction_result['probabilities']
                    result = {
                        'url': url,
                        'title': extraction_result['title'],
                        'content': extraction_result.get('content', ''),
                        'domain': extraction_result['domain'],
                        'prediction': prediction_result['prediction'],
                        'confidence': prediction_result['confidence'],
                        'fake_probability': probs.get('FAKE', probs.get('fake', 0.0)),
                        'real_probability': probs.get('REAL', probs.get('real', 0.0)),
                        'important_words': important_words,
                        'status': 'success'
                    }
                else:
                    result = {
                        'url': url,
                        'status': 'failed',
                        'error': extraction_result.get('error', 'Unknown error')
                    }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'url': url,
                    'status': 'failed',
                    'error': str(e)
                })
    
    return results