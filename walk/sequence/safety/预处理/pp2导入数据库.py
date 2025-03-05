import pandas as pd
from sqlalchemy import create_engine
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取CSV文件
csv_file = r"D:\毕设\数据集\Place Pulse 2.0\metadata\final_data.csv"  # 替换为你的CSV文件路径
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    logging.error(f"读取CSV文件失败: {e}")
    exit(1)

# 创建数据库连接（使用pymysql）
conn_string = 'mysql+pymysql://root:123@localhost:3306/scene?charset=utf8'

# 创建SQLAlchemy引擎
engine = create_engine(conn_string)

# 定义表名
table_name = 'pp2'

# 定义每批次的大小
batch_size = 1000  # 可根据需要调整

def insert_data_in_batches(df, engine, table_name, batch_size):
    total_rows = len(df)
    logging.info(f"总行数: {total_rows}")

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        chunk_df = df.iloc[start:end]
        
        try:
            with engine.begin() as connection:
                chunk_df.to_sql(name=table_name, con=connection, if_exists='append', index=False, method='multi')
            logging.info(f"已成功插入批次 {start + 1} 到 {end}")
        except Exception as e:
            logging.error(f"插入批次 {start + 1} 到 {end} 失败: {e}")
            # 强制关闭当前连接并重新创建引擎
            engine.dispose()
            engine = create_engine(conn_string)
            logging.info("重新尝试数据导入...")
            insert_data_in_batches(chunk_df, engine, table_name, batch_size)
            break

# 调用函数分批次插入数据
insert_data_in_batches(df, engine, table_name, batch_size)

# 关闭数据库连接
engine.dispose()