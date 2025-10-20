import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

# 1. 定义文件路径
file_path = 'data/DrugLikeSMILSE-debug/train/data-00000-of-00001.arrow'

# 2. 以兼容模式读取文件
with open(file_path, "rb") as f:
    try:
        # 优先尝试作为 File 格式读取 (用于磁盘存储)
        reader = ipc.open_file(f)
    except pa.ArrowInvalid:
        # 如果失败，则回退到 Stream 格式 (用于流式传输)
        f.seek(0) # 必须将文件指针移回开头
        reader = ipc.open_stream(f)
    
    # 3. 从 reader 中一次性读取所有数据为 Arrow Table，然后转为 Pandas DataFrame
    df = reader.read_all().to_pandas()

# 4. 打印 DataFrame 的头部信息
print(df.head())