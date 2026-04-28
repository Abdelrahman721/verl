import glob
import os

import pandas as pd

directory = "/data/abdelrahman/verl/data/full_mix/train"
parquet_files = sorted(glob.glob(os.path.join(directory, "*.parquet")))

for path in parquet_files:
    df = pd.read_parquet(path)
    print(f"{os.path.basename(path)}: {len(df)} rows, {len(df.columns)} columns")
