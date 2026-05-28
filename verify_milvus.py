"""Milvus 连接验证脚本。"""
from src.milvus_client import MilvusClient

VECTOR_DIM = 1536  # text-embedding-v1 向量维度
COLLECTION = "medical_chunks"

# 从 config.yml 读取 milvus 配置
from src.config_loader import get_milvus_config

cfg = get_milvus_config()
print(f"Milvus 配置: host={cfg['host']}, port={cfg['port']}")

client = MilvusClient(
    host=cfg["host"],
    port=cfg["port"],
    collection_name=COLLECTION,
    vector_dim=VECTOR_DIM,
)

# 验证连接
client.connect()
print("✅ Milvus 连接成功")

# 检查 Collection 是否存在
exists = client.collection_exists()
print(f"Collection '{COLLECTION}' {'已存在' if exists else '不存在'}")

# 如果存在则查看统计信息
if exists:
    stats = client.get_collection_stats()
    print(f"Collection 统计: {stats}")
else:
    print("提示：运行 create_collection() 即可创建 Collection")

client.disconnect()
print("✅ 已断开连接，验证完成")
