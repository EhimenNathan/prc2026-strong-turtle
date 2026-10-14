"""Read only requested state rows, one column at a time."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from config import WORK_DIR

def read_state(ids, filenames):
    frames = []
    for filename in filenames:
        file = pq.ParquetFile(Path(WORK_DIR) / filename)
        columns = [name for name in file.schema_arrow.names if name != 'id']
        row_ids = file.read(columns=['id']).column(0).to_numpy().astype('int64')
        _, first = np.unique(row_ids,return_index=True)
        selected = np.isin(row_ids,ids) & np.isin(np.arange(len(row_ids)),first)
        matrix = np.empty((int(selected.sum()),len(columns)),np.float32)
        for j, name in enumerate(columns):
            matrix[:,j] = file.read(columns=[name]).column(0).to_numpy(zero_copy_only=False)[selected].astype(np.float32)
        frames.append(pd.DataFrame(matrix,index=row_ids[selected],columns=columns))
    result = pd.concat(frames)
    return result[~result.index.duplicated()].reindex(ids)
