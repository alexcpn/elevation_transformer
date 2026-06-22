import glob
import pyarrow.parquet as pq

files = sorted(glob.glob("/data/itm_loss/*.parquet"))
total = 0

for f in files:
    n = pq.ParquetFile(f).metadata.num_rows
    total += n

print("files:", len(files))
print("total rows:", total)