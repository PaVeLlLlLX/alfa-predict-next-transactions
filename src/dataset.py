import pandas as pd
import numpy as np
import torch
import gc
from collections import Counter
from torch.utils.data import IterableDataset
from typing import List, Dict, Any, Optional, Iterator, Union
import pyarrow.parquet as pq
from src import config

class CategoricalEncoder:
    def __init__(self, vocabs: Dict[str, List[str]]):
        self.mapping: Dict[str, Dict[str, int]] = {}
        
        for col, vals in vocabs.items():
            # маппинг {Value: Index}
            self.mapping[col] = {v: i for i, v in enumerate(vals)}
            
    def get_mapper(self, col):
        return self.mapping.get(col)
    
    def encode_series(self, series: pd.Series, col_name: str) -> np.ndarray:
        mapper = self.mapping.get(col_name)
        if mapper:
            rare_idx = mapper.get("__RARE__", 0)
            return series.map(mapper).fillna(rare_idx).astype(np.int64).values
        else:
            return np.zeros(len(series), dtype=np.int64)


class DataProcessor:
    def __init__(self, svd_pipeline: Any, mcc_dict: Dict[str, str]):
        self.svd = svd_pipeline
        self.mcc_dict = mcc_dict

    def process_chunk(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        raw_seqs: List[str] = df_chunk['prev_mcc_seq'].fillna('').astype(str).tolist()

        seq_s = df_chunk['prev_mcc_seq'].fillna('').astype(str)
        seq = seq_s.str.split()

        del seq_s
        # Конвертация MCC → категории
        def map_to_cat(lst: List[str]) -> List[str]:
            return [self.mcc_dict.get(x, "__UNK__") for x in lst]

        df_chunk['seq_cat'] = seq.apply(map_to_cat)

        del seq
        df_chunk['period_start'] = pd.to_datetime(df_chunk['period_start'], errors='coerce')

        ps_hour = df_chunk['period_start'].dt.hour.fillna(0)
        ps_dow = df_chunk['period_start'].dt.dayofweek.fillna(0)
        ps_month = df_chunk['period_start'].dt.month.fillna(1) # Месяц от 1 до 12

        df_chunk['hour_sin'] = np.sin(2 * np.pi * ps_hour / 24).astype('float16')
        df_chunk['hour_cos'] = np.cos(2 * np.pi * ps_hour / 24).astype('float16')
        
        df_chunk['dow_sin'] = np.sin(2 * np.pi * ps_dow / 7).astype('float16')
        df_chunk['dow_cos'] = np.cos(2 * np.pi * ps_dow / 7).astype('float16')
        
        df_chunk['month_sin'] = np.sin(2 * np.pi * ps_month / 12).astype('float16')
        df_chunk['month_cos'] = np.cos(2 * np.pi * ps_month / 12).astype('float16')

        def calc_share(lst: List[str], target_set: set) -> float:
            if not lst: return 0.0
            cnt = sum(1 for x in lst if x in target_set)
            return cnt / len(lst)

        df_chunk['hedonism_share'] = df_chunk['seq_cat'].apply(lambda x: calc_share(x, config.HEDONISM_SET)).astype('float16')
        df_chunk['obligatory_share'] = df_chunk['seq_cat'].apply(lambda x: calc_share(x, config.OBLIGATORY_SET)).astype('float16')
        df_chunk['auto_share'] = df_chunk['seq_cat'].apply(lambda x: calc_share(x, config.AUTO_SET)).astype('float16')


        df_chunk['last_cat'] =  df_chunk['seq_cat'].apply(lambda lst: lst[-1] if lst else "__MISSING__")

        # Энтропия категорий
        def entropy(lst: List[str]) -> float:
            if not lst:
                return 0.0
            cnt = Counter(lst)
            total = len(lst)
            vals = np.array([c/total for c in cnt.values()])
            return -np.sum(vals * np.log(vals + 1e-9))

        df_chunk['cat_entropy'] = df_chunk['last_cat'].apply(entropy).astype('float16')
        df_chunk["cat_entropy"] = df_chunk["cat_entropy"] / 3.5 
 
        def transitions(lst: List[str]) -> str:
            if len(lst) < 2: return "__NONE__"
            return f"{lst[-2]}->{lst[-1]}"

        df_chunk['last_transition'] = df_chunk['seq_cat'].apply(transitions)
        df_chunk = df_chunk.drop(columns=['prev_mcc_seq'])
        df_chunk['age'] = pd.to_numeric(df_chunk['age'].fillna(0), downcast='integer')
        df_chunk['life_time_days'] = pd.to_numeric(df_chunk['life_time_days'].fillna(0), downcast='integer')
        df_chunk['age_log'] = np.log1p(df_chunk['age'].clip(lower=0))
        df_chunk['life_time_days_log'] = np.log1p(df_chunk['life_time_days'].clip(lower=0))
        df_chunk['period_segment'] = df_chunk['day_period'].astype(str) + "_" + df_chunk['clientsegment'].astype(str)
        df_chunk["age"] = df_chunk["age"] / 100
        df_chunk["life_time_days"] = df_chunk["life_time_days"] / 10000

        try:
            svd_features = self.svd.transform(raw_seqs)
        except:
            svd_features = np.zeros((len(df_chunk), config.N_SVD))
        
        for i in range(config.N_SVD):
            df_chunk[f'svd_{i}'] = svd_features[:, i].round(3)


        for col in config.CAT_COLS:
            if col in df_chunk.columns:
                df_chunk[col] = df_chunk[col].fillna('__MISSING__').astype(str)

        return df_chunk
    

class TabularStreamingDataset(IterableDataset):
    def __init__(self, parquet_files: List[str], encoder: CategoricalEncoder, processor: DataProcessor, is_test: bool = False):
        super().__init__()
        self.files = parquet_files
        self.encoder = encoder
        self.processor = processor
        self.is_test = is_test
        self.cols_to_read = [
            'adminarea', 'gender', 'age', 'clientsegment', 'life_time_days', 
            'day_period', 'prev_mcc_seq', 'target', 'period_start'
        ]
        if is_test:
             self.cols_to_read = [c for c in self.cols_to_read if c != 'target']

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files_to_read = self.files
        else:
            per_worker = int(np.ceil(len(self.files) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.files))
            files_to_read = self.files[start:end]

        for file_path in files_to_read:
            pq_file = pq.ParquetFile(file_path)
            for i in range(pq_file.num_row_groups):
                df = pq_file.read_row_group(i, columns=self.cols_to_read).to_pandas()
                
                if self.is_test and 'target' not in df.columns:
                    df['target'] = 0 # Dummy target for test

                df = self.processor.process_chunk(df)
                
                x_num = df[config.NUM_COLS].values.astype(np.float32)
                x_cat_list = [self.encoder.encode_series(df[col], col) for col in config.CAT_COLS]
                x_cat = np.stack(x_cat_list, axis=1)
                
                labels = df['target'].tolist()
                
                for j in range(len(df)):
                    yield {
                        'x_num': torch.tensor(x_num[j]),
                        'x_cat': torch.tensor(x_cat[j]),
                        'labels': torch.tensor(labels[j], dtype=torch.float32)
                    }
                del df, x_num, x_cat
                gc.collect()


                