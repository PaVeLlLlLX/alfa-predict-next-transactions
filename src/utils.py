import torch
import numpy as np
import random
import logging
import os
import sys


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(name="alfa_solution"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
 
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler("log.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def hamming_score_weighted(y_true, y_pred, category_weights=None):
    """
    Calculates weighted Hamming score for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        Ground truth binary matrix.
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted binary matrix.
    category_weights : np.ndarray, shape (n_classes,), optional
        Weights for each class. If None, all weights are 1.

    Returns
    -------
    float
        Weighted Hamming score (average per sample).
    """
    n_samples, n_classes = y_true.shape
    if category_weights is None:
        category_weights = np.ones(n_classes)
    else:
        category_weights = np.array(category_weights)
        if category_weights.shape[0] != n_classes:
            raise ValueError("category_weights must have the same length as the number of classes")

    intersection = y_true * y_pred
    union = (y_true + y_pred) > 0
    
    # Apply weights to intersection and union for each sample
    weighted_intersection = np.sum(intersection * category_weights, axis=1)
    weighted_union = np.sum(union * category_weights, axis=1)
    
    # Handle special case where both true and pred are all zeros
    both_zero = (np.sum(y_true, axis=1) == 0) & (np.sum(y_pred, axis=1) == 0)
    
    # Calculate scores, avoiding division by zero
    scores = np.where(weighted_union > 0,
                      weighted_intersection / weighted_union,
                      1.0
                     )
    
    # Set score to 1.0 for samples where both true and pred are all zeros
    scores = np.where(both_zero, 1.0, scores)
    
    return np.mean(scores)


