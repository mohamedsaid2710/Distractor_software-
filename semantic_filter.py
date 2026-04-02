#!/usr/bin/env python3
"""Semantic dissimilarity filtering for maze distractors.

Uses fastText embeddings to reject candidates that are semantically
similar to the target word. This ensures distractors come from 
unrelated semantic domains (e.g., "Musikerin" → "Stoffwechsel" not "Sängerin").
"""

import os
import logging
import numpy as np

# Lazy-loaded embedding model
_embedding_model = None
_embedding_lang = None


def _load_fasttext(lang='de'):
    """Load fastText embeddings for the specified language."""
    global _embedding_model, _embedding_lang
    
    if _embedding_model is not None and _embedding_lang == lang:
        return _embedding_model
    
    try:
        import fasttext.util
        import fasttext
        
        # Map language codes
        ft_lang = {'de': 'de', 'en': 'en', 'ar': 'ar'}.get(lang, 'de')
        
        # Check for pre-downloaded model
        model_path = f'cc.{ft_lang}.300.bin'
        
        # Look in common locations
        search_paths = [
            model_path,
            os.path.join(os.path.dirname(__file__), model_path),
            os.path.join(os.path.dirname(__file__), 'models', model_path),
            os.path.expanduser(f'~/.fasttext/{model_path}'),
        ]
        
        found_path = None
        for p in search_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path is None:
            logging.info(f"Downloading fastText model for '{ft_lang}'...")
            fasttext.util.download_model(ft_lang, if_exists='ignore')
            found_path = model_path
        
        logging.info(f"Loading fastText embeddings from {found_path}...")
        _embedding_model = fasttext.load_model(found_path)
        _embedding_lang = lang
        logging.info("FastText embeddings loaded successfully.")
        return _embedding_model
        
    except ImportError:
        logging.warning("fasttext not installed. Install with: pip install fasttext")
        return None
    except Exception as e:
        logging.warning(f"Failed to load fastText: {e}")
        return None


def _load_gensim_fasttext(lang='de'):
    """Fallback: Load fastText via gensim (smaller memory footprint)."""
    global _embedding_model, _embedding_lang
    
    if _embedding_model is not None and _embedding_lang == lang:
        return _embedding_model
    
    try:
        import gensim.downloader as api
        
        # Use smaller pre-trained models
        model_name = {
            'de': 'fasttext-wiki-news-subwords-300',  # Falls back to multilingual
            'en': 'fasttext-wiki-news-subwords-300',
        }.get(lang, 'fasttext-wiki-news-subwords-300')
        
        logging.info(f"Loading gensim model: {model_name}...")
        _embedding_model = api.load(model_name)
        _embedding_lang = lang
        return _embedding_model
        
    except Exception as e:
        logging.warning(f"Failed to load gensim model: {e}")
        return None


def get_word_vector(word, lang='de'):
    """Get the embedding vector for a word."""
    model = _load_fasttext(lang)
    if model is None:
        return None
    
    try:
        # fastText can handle OOV words via subword embeddings
        return model.get_word_vector(word.lower())
    except Exception:
        return None


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def semantic_similarity(word1, word2, lang='de'):
    """Compute semantic similarity between two words (-1 to 1)."""
    vec1 = get_word_vector(word1, lang)
    vec2 = get_word_vector(word2, lang)
    
    sim = cosine_similarity(vec1, vec2)
    return sim


def is_semantically_dissimilar(target, candidate, threshold=0.35, lang='de'):
    """Check if candidate is semantically dissimilar enough from target.
    
    Args:
        target: The target word
        candidate: The distractor candidate
        threshold: Maximum allowed similarity (lower = more strict)
        lang: Language code
        
    Returns:
        True if candidate is dissimilar enough (good distractor)
        False if candidate is too similar (bad distractor)
    """
    sim = semantic_similarity(target, candidate, lang)
    return sim < threshold


def filter_by_semantic_dissimilarity(target, candidates, threshold=0.35, lang='de'):
    """Filter candidates to keep only semantically dissimilar ones.
    
    Args:
        target: The target word
        candidates: List of candidate distractors
        threshold: Maximum allowed similarity
        lang: Language code
        
    Returns:
        List of candidates that are semantically dissimilar to target
    """
    model = _load_fasttext(lang)
    if model is None:
        logging.debug("No embedding model available, skipping semantic filter")
        return candidates
    
    target_vec = get_word_vector(target, lang)
    if target_vec is None:
        return candidates
    
    filtered = []
    for cand in candidates:
        text = cand.text if hasattr(cand, 'text') else str(cand)
        cand_vec = get_word_vector(text, lang)
        sim = cosine_similarity(target_vec, cand_vec)
        
        if sim < threshold:
            filtered.append(cand)
    
    # If filtering removed too many, relax threshold
    if len(filtered) < len(candidates) * 0.1 and len(filtered) < 50:
        logging.debug(f"Semantic filter too strict, relaxing threshold from {threshold} to {threshold + 0.15}")
        return filter_by_semantic_dissimilarity(target, candidates, threshold + 0.15, lang)
    
    return filtered


def batch_filter_semantic(target, candidates, threshold=0.35, lang='de'):
    """Efficiently filter candidates using batch vector operations.
    
    This is faster than filtering one by one.
    """
    model = _load_fasttext(lang)
    if model is None:
        return candidates
    
    try:
        target_vec = model.get_word_vector(target.lower())
    except Exception:
        return candidates
    
    if target_vec is None:
        return candidates
    
    # Batch get all vectors at once - fastText is optimized for this
    cand_vecs = []
    valid_indices = []
    for i, cand in enumerate(candidates):
        try:
            text = cand.text if hasattr(cand, 'text') else str(cand)
            vec = model.get_word_vector(text.lower())
            if vec is not None:
                cand_vecs.append(vec)
                valid_indices.append(i)
        except Exception:
            continue
    
    if not cand_vecs:
        return candidates
    
    # Stack into matrix for batch computation
    cand_matrix = np.vstack(cand_vecs)
    
    # Normalize
    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    cand_norms = cand_matrix / (np.linalg.norm(cand_matrix, axis=1, keepdims=True) + 1e-9)
    
    # Batch cosine similarity - single matrix operation
    similarities = np.dot(cand_norms, target_norm)
    
    # Filter using vectorized comparison
    mask = similarities < threshold
    filtered = [candidates[valid_indices[i]] for i, keep in enumerate(mask) if keep]
    
    # Relaxation if too strict
    if len(filtered) < len(candidates) * 0.1 and len(filtered) < 50:
        return batch_filter_semantic(target, candidates, threshold + 0.15, lang)
    
    return filtered


# Convenience function for integration
def apply_semantic_filter(target_word, candidates, params):
    """Apply semantic filtering based on params configuration.
    
    Params:
        semantic_filter: True/False - enable/disable
        semantic_threshold: float (default 0.35) - max similarity allowed
        language: 'de', 'en', 'ar'
    """
    if not params.get('semantic_filter', False):
        return candidates
    
    threshold = float(params.get('semantic_threshold', 0.35))
    lang = params.get('language', 'de')
    
    return batch_filter_semantic(target_word, candidates, threshold, lang)
