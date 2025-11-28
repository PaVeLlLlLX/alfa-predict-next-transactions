import sys
import os
import pytest
import pandas as pd
import numpy as np
import torch

sys.path.append(os.getcwd())

from src.dataset import CategoricalEncoder, DataProcessor
from src import config

class MockSVD:
    def transform(self, X):
        return np.random.rand(len(X), config.N_SVD)

MOCK_MCC_DICT = {
    '5411': 'supermarkety',
    '5814': 'fastfud',
    '1111': 'unknown_cat'
}

MOCK_VOCABS = {
    'gender': ['M', 'F'],
    'clientsegment': ['MASS', 'VIP'],
    'day_period': ['morning', 'evening'],
    'adminarea': ['Msk', 'Spb'],
    'last_cat': ['supermarkety', 'fastfud'],
    'last_transition': ['supermarkety->fastfud']
}

def test_categorical_encoder_logic():
    encoder = CategoricalEncoder(MOCK_VOCABS)
    
    # Тест известных значений
    series = pd.Series(['M', 'F'])
    encoded = encoder.encode_series(series, 'gender')
    expected = np.array([0, 1]) 
    assert np.array_equal(encoded, expected), f"Expected {expected}, got {encoded}"
    
    # Тест неизвестных значений
    series_unknown = pd.Series(['F', 'Alien', 'Predator'])
    encoded_unknown = encoder.encode_series(series_unknown, 'gender')
    expected_unknown = np.array([1, 0, 0])
    assert np.array_equal(encoded_unknown, expected_unknown), "Unknown values should be 0"

def test_feature_engineering_correctness():
    df = pd.DataFrame({
        'prev_mcc_seq': ['5411 5814', ''], # 1 Обычный, 2 Пустой
        'period_start': ['2022-01-01 12:00:00', pd.NaT],
        'age': [30, None],
        'life_time_days': [100, None],
        'gender': ['M', 'F'],
        'clientsegment': ['MASS', 'VIP'],
        'day_period': ['morning', 'evening'],
        'adminarea': ['Msk', 'Spb'],
        'target': [0, 1] # Заглушка
    })
    
    processor = DataProcessor(MockSVD(), MOCK_MCC_DICT)
    processed_df = processor.process_chunk(df)
    
    # Строка 0: '5411'(supermarkety - oblig) + '5814'(fastfud - hedonism)
    # Ожидаем: hedonism = 0.5, obligatory = 0.5
    hedonism = processed_df.iloc[0]['hedonism_share']
    obligatory = processed_df.iloc[0]['obligatory_share']
    
    assert np.isclose(hedonism, 0.5), f"Hedonism share wrong: {hedonism}"
    assert np.isclose(obligatory, 0.5), f"Obligatory share wrong: {obligatory}"
    
    assert processed_df.iloc[1]['hedonism_share'] == 0.0
    
    # 12:00 - Sin должен быть около 0, Cos -1
    h_sin = processed_df.iloc[0]['hour_sin']
    h_cos = processed_df.iloc[0]['hour_cos']
    
    # Используем atol т.к. float32
    assert np.isclose(h_sin, 0.0, atol=1e-4)
    assert np.isclose(h_cos, -1.0, atol=1e-4)
    
    # Должны появиться svd_0 ... svd_15
    for i in range(config.N_SVD):
        assert f'svd_{i}' in processed_df.columns

def test_data_integrity_and_types():
    df = pd.DataFrame({
        'prev_mcc_seq': ['5411'],
        'period_start': ['2022-01-01'],
        'age': [25],
        'life_time_days': [100],
        'gender': ['M'],
        'clientsegment': ['MASS'],
        'day_period': ['morning'],
        'adminarea': ['Msk'],
        'target': [0]
    })
    
    processor = DataProcessor(MockSVD(), MOCK_MCC_DICT)
    df_proc = processor.process_chunk(df)
    
    # Проверка что float
    for col in config.NUM_COLS:
        assert pd.api.types.is_float_dtype(df_proc[col]), f"Column {col} is not float!"
        
    # Проверка на отсутствие NaNs
    assert not df_proc[config.NUM_COLS].isnull().values.any(), "NaNs found in numerical columns!"
    assert not df_proc[config.CAT_COLS].isnull().values.any(), "NaNs found in categorical columns!"

    
def test_transitions_logic():
    df = pd.DataFrame({
        'prev_mcc_seq': [
            '5411 5814 1111',  # 3 транзакции -> '5814->1111' -> 'fastfud->unknown_cat'
            '5411',            # 1 транзакция -> Недостаточно -> '__NONE__'
            ''                 # 0 транзакций -> '__NONE__'
        ],
        # Остальные поля для заглушки
        'period_start': [pd.NaT]*3, 'age': [0]*3, 'life_time_days': [0]*3,
        'gender': ['M']*3, 'clientsegment': ['S']*3, 'day_period': ['D']*3, 'adminarea': ['A']*3
    })

    processor = DataProcessor(MockSVD(), MOCK_MCC_DICT)
    df_proc = processor.process_chunk(df)
    
    transitions = df_proc['last_transition'].values
    
    # MOCK_MCC_DICT: 5411->supermarkety, 5814->fastfud, 1111->unknown_cat
    
    # Длинная последовательность
    assert transitions[0] == "fastfud->unknown_cat", f"Got: {transitions[0]}"
    
    # Короткая последовательность
    assert transitions[1] == "__NONE__"
    
    # Пустая
    assert transitions[2] == "__NONE__"

def test_normalization_logic():
    df = pd.DataFrame({
        'age': [50, 25, 0],
        'life_time_days': [10000, 5000, 0],
        'prev_mcc_seq': ['']*3, 'period_start': [pd.NaT]*3,
        'gender': ['M']*3, 'clientsegment': ['S']*3, 'day_period': ['D']*3, 'adminarea': ['A']*3
    })
    
    processor = DataProcessor(MockSVD(), MOCK_MCC_DICT)
    df_proc = processor.process_chunk(df)
    
    # age / 100
    assert np.isclose(df_proc['age'][0], 0.5)
    assert np.isclose(df_proc['age'][1], 0.25)
    
    # life_time_days / 10000
    assert np.isclose(df_proc['life_time_days'][0], 1.0)
    assert np.isclose(df_proc['life_time_days'][1], 0.5)

def test_period_segment_creation():
    df = pd.DataFrame({
        'day_period': ['morning', 'evening'],
        'clientsegment': ['MASS', 'VIP'],
        'prev_mcc_seq': ['']*2, 'period_start': [pd.NaT]*2, 'age': [0]*2, 'life_time_days': [0]*2,
        'gender': ['M']*2, 'adminarea': ['A']*2
    })
    
    processor = DataProcessor(MockSVD(), MOCK_MCC_DICT)
    df_proc = processor.process_chunk(df)
    
    assert df_proc['period_segment'][0] == "morning_MASS"
    assert df_proc['period_segment'][1] == "evening_VIP"

def test_entropy_logic():
    # Категории одинаковые - энтропия 0
    # Разные - энтропия > 0
    df = pd.DataFrame({
        'prev_mcc_seq': [
            '5411 5411 5411', # Одинаковые
            '5411 5814'       # Разные
        ],
        'period_start': [pd.NaT]*2, 'age': [0]*2, 'life_time_days': [0]*2,
        'gender': ['M']*2, 'clientsegment': ['S']*2, 'day_period': ['D']*2, 'adminarea': ['A']*2
    })
    
    processor = DataProcessor(MockSVD(), MOCK_MCC_DICT)
    df_proc = processor.process_chunk(df)
    
    assert np.isclose(df_proc['cat_entropy'][0], 0.0)
    assert df_proc['cat_entropy'][1] > 0.0


if __name__ == "__main__":
    # Для python tests/test_pipeline.py
    sys.exit(pytest.main(["-v", __file__]))