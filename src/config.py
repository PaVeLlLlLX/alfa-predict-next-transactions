import torch
import numpy as np
import os

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TRAIN_SOURCE = [
    os.path.join(DATA_DIR, 'train/train_1.parquet'), 
    os.path.join(DATA_DIR, 'train/train_2.parquet')
]
TEST_SOURCE = os.path.join(DATA_DIR, 'test.csv')
MCC_MAP_PATH = os.path.join(DATA_DIR, 'mcc_to_cat_mapping.xlsx')

TRAIN_FINAL = os.path.join(DATA_DIR, 'train_final.parquet')
VAL_FINAL = os.path.join(DATA_DIR, 'validation_final.parquet')
TEST_PARQUET_TEMP = os.path.join(DATA_DIR, 'test_temp.parquet')

SVD_PATH = os.path.join(ARTIFACTS_DIR, 'tfidf_svd.joblib')
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, 'cat_vocabs.json')
MODEL_CHECKPOINT = os.path.join(ARTIFACTS_DIR, 'best_ftt_model.pt')
VAL_PREDS_PATH = os.path.join(ARTIFACTS_DIR, 'val_predictions.npz')
THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, 'best_thresholds.json') 
TEST_PROBS_PATH = os.path.join(ARTIFACTS_DIR, 'test_probs_ftt.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-4
WD = 1e-5
N_SVD = 16
SAMPLE_SIZE = 2_000_000

CATS_TARGET = [
    'alkogol', 'arenda_avto', 'avto', 'azs', 'blagotvoritelnost', 'dom_i_remont', 'fastfud',
    'juvelirnye_izdelija', 'kafe_i_restorany', 'knigi', 'kommunalnye_uslugi', 'krasota', 'kredity',
    'obrazovanie', 'odezhda_i_obuv', 'prochie_rashody', 'puteshestvija', 'razvlechenija', 'sber_ecosystem',
    'shtrafy_i_nalogi', 'sportivnye_tovary', 'supermarkety', 'svjaz_internet_i_tv', 'tabak', 'taksi',
    'tehnika', 'transport', 'tsifrovye_tovary', 'tsvety', 'yandex_ecosystem', 'zdorove', 'zhivotnye'
]

WEIGHTS = np.array([
    0.37, 0.66, 0.41, 0.34, 0.63, 0.39, 0.24, 0.72, 0.38, 0.5 , 0.57,
    0.45, 0.68, 0.62, 0.41, 0.35, 0.52, 0.38, 0.52, 0.55, 0.6 , 0.08,
    0.47, 0.45, 0.5 , 0.52, 0.26, 0.41, 0.49, 0.28, 0.33, 0.53
])

NUM_COLS = [
    "age", "life_time_days", "cat_entropy", 
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "hedonism_share", "obligatory_share", "auto_share"
]
NUM_COLS.extend([f"svd_{i}" for i in range(N_SVD)])

CAT_COLS = [
    'gender', 'clientsegment', 'day_period',
    'adminarea', 'period_segment', "last_cat", "last_transition"
]

HEDONISM_SET = {'fastfud', 'kafe_i_restorany', 'razvlechenija', 'puteshestvija', 'krasota', 'juvelirnye_izdelija', 'tsvety'}
OBLIGATORY_SET = {'supermarkety', 'kommunalnye_uslugi', 'svjaz_internet_i_tv', 'dom_i_remont', 'zdorove', 'prochie_rashody'}
AUTO_SET = {'azs', 'avto', 'arenda_avto', 'shtrafy_i_nalogi', 'taksi', 'transport'}