import sys
import os
from typing import List, Dict
sys.path.append(os.getcwd())

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import joblib
import json
import gc
from tqdm import tqdm
from collections import Counter
from src import config, utils


logger = utils.get_logger()

def split_data() -> None:
    logger.info("1. Сплит данных по времени (Train/Val)")
    all_timestamps: List[pd.DataFrame] = []

    for file_path in config.TRAIN_SOURCE:
        pf = pq.ParquetFile(file_path)
        for i in tqdm(range(pf.num_row_groups), desc=f"Reading timestamps {file_path}"):
            ts_chunk = pf.read_row_group(i, columns=['period_start']).to_pandas()
            ts_chunk['period_start'] = pd.to_datetime(ts_chunk['period_start'], errors='coerce')
            all_timestamps.append(ts_chunk)
    
    timestamps_df = pd.concat(all_timestamps, ignore_index=True)
    del all_timestamps
    gc.collect()

    split_timestamp = timestamps_df['period_start'].quantile(0.9) # 1 - 0.1
    logger.info(f"\nВременная отсечка (90%): {split_timestamp}")
    del timestamps_df
    gc.collect()

    logger.info("Подготовка схемы")
    pf_sample = pq.ParquetFile(config.TRAIN_SOURCE[0])
    df_sample = pf_sample.read_row_group(0).to_pandas()
    df_sample['period_start'] = pd.to_datetime(df_sample['period_start'], errors='coerce')
    corrected_schema = pa.Table.from_pandas(df_sample, preserve_index=False).schema
    del df_sample, pf_sample

    if os.path.exists(config.TRAIN_FINAL): os.remove(config.TRAIN_FINAL)
    if os.path.exists(config.VAL_FINAL): os.remove(config.VAL_FINAL)

    logger.info("Запись файлов")
    with pq.ParquetWriter(config.TRAIN_FINAL, schema=corrected_schema) as train_w, \
         pq.ParquetWriter(config.VAL_FINAL, schema=corrected_schema) as val_w:
        
        for file_path in config.TRAIN_SOURCE:
            pf = pq.ParquetFile(file_path)
            for i in tqdm(range(pf.num_row_groups), desc=f"Processing {file_path}"):
                df_chunk = pf.read_row_group(i).to_pandas()
                df_chunk['period_start'] = pd.to_datetime(df_chunk['period_start'], errors='coerce')
                
                train_part = df_chunk[df_chunk['period_start'] < split_timestamp]
                val_part = df_chunk[df_chunk['period_start'] >= split_timestamp]
                
                if not train_part.empty:
                    train_w.write_table(pa.Table.from_pandas(train_part, schema=corrected_schema, preserve_index=False))
                if not val_part.empty:
                    val_w.write_table(pa.Table.from_pandas(val_part, schema=corrected_schema, preserve_index=False))
                del df_chunk, train_part, val_part
                gc.collect()

def train_svd() -> None:
    logger.info("\n2. Обучение SVD")
    pf = pq.ParquetFile(config.TRAIN_FINAL)
    df = pf.read_row_group(0, columns=['prev_mcc_seq']).to_pandas()
    if len(df) < config.SAMPLE_SIZE and pf.num_row_groups > 1:
        df = pd.concat([df, pf.read_row_group(1, columns=['prev_mcc_seq']).to_pandas()])
    
    corpus: List[str] = df['prev_mcc_seq'].fillna('').astype(str).tolist()
    pipe = make_pipeline(
        TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1, 1), max_features=5000),
        TruncatedSVD(n_components=config.N_SVD, random_state=42)
    )
    pipe.fit(corpus)
    joblib.dump(pipe, config.SVD_PATH)
    logger.info(f"SVD сохранен: {config.SVD_PATH}")

def build_vocab() -> None:
    logger.info("\n3. Сбор словарей категорий")
    mcc_map = pd.read_excel(config.MCC_MAP_PATH)
    mcc_dict: Dict[str, str] = dict(zip(mcc_map["mcc"].astype(str), mcc_map["eng_cat"]))
    
    counters: Dict[str, Counter] = {c: Counter() for c in config.CAT_COLS}
    files = [config.TRAIN_FINAL, config.VAL_FINAL]
    
    for f in files:
        pf = pq.ParquetFile(f)
        cols = ['adminarea', 'gender', 'clientsegment', 'day_period', 'prev_mcc_seq']
        for i in tqdm(range(pf.num_row_groups), desc=f"Vocab scan {f}"):
            df = pf.read_row_group(i, columns=cols).to_pandas()
            
            # Версия процессинга только для категорий
            seq = df['prev_mcc_seq'].fillna('').astype(str).str.split()
            df['seq_cat'] = seq.apply(lambda lst: [mcc_dict.get(x, "__UNK__") for x in lst])
            df['last_cat'] = df['seq_cat'].apply(lambda lst: lst[-1] if lst else "__MISSING__").astype(str)
            df['last_transition'] = df['seq_cat'].apply(lambda lst: f"{lst[-2]}->{lst[-1]}" if len(lst) >=2 else "__NONE__").astype(str)
            df['period_segment'] = df['day_period'].astype(str) + "_" + df['clientsegment'].astype(str)
            
            for col in config.CAT_COLS:
                if col in df.columns:
                    counters[col].update(df[col].fillna('__MISSING__').astype(str).values)
            del df

    final_vocabs: Dict[str, List[str]] = {}
    limits = {"last_transition": 2000, "adminarea": 500, "default": 1000}
    
    for col in config.CAT_COLS:
        limit = limits.get(col, limits["default"])
        common = [val for val, _ in counters[col].most_common(limit)]

        if "__MISSING__" in common: 
            common.remove("__MISSING__")
            
        final_vocabs[col] = ["__MISSING__", "__RARE__"] + sorted(common)
    
    with open(config.VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_vocabs, f, ensure_ascii=False)
    logger.info(f"Словари сохранены: {config.VOCAB_PATH}")

if __name__ == "__main__":
    split_data()
    train_svd()
    build_vocab()