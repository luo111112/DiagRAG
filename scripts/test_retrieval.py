"""检索功能验证脚本。

用法:
    python scripts/test_retrieval.py "高血压的常见症状"
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import get_milvus_config, get_embedding_config
from src.embedding_client import DashScopeEmbeddingClient
from src.milvus_client import MilvusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        query = "高血压的常见症状"
        logger.info("未提供查询参数，使用默认查询: %s", query)
    else:
        query = sys.argv[1]

    # 1. 加载配置
    milvus_cfg = get_milvus_config()
    logger.info("Milvus 配置: host=%s, port=%s, collection=%s",
                milvus_cfg["host"], milvus_cfg["port"], milvus_cfg["collection_name"])

    # 2. 连接 Milvus
    client = MilvusClient(
        host=milvus_cfg["host"],
        port=milvus_cfg["port"],
        collection_name=milvus_cfg["collection_name"],
        vector_dim=milvus_cfg["vector_dim"],
    )
    client.connect()
    logger.info("Milvus 连接成功")

    # 3. 检查 Collection 状态
    stats = client.get_collection_stats()
    logger.info("Collection 统计: %s", stats)
    if stats.get("num_entities", 0) == 0:
        logger.warning("Collection 为空，请先运行 build_milvus.py 导入数据")

    # 4. Embedding
    emb_client = DashScopeEmbeddingClient()
    logger.info("Embedding 模型已初始化")

    logger.info("正在为查询生成向量: %s", query)
    query_vector = emb_client.embed_text(query)
    logger.info("向量生成完成，维度=%d，前5维: %s", len(query_vector), query_vector[:5])

    # 5. 检索（稠密向量语义搜索）
    top_k = 5
    logger.info("执行向量检索，top_k=%d", top_k)
    results = client.search(query_vector=query_vector, top_k=top_k)

    # 6. 打印结果
    print("\n" + "=" * 60)
    print(f"查询: {query}")
    print(f"检索结果共 {len(results)} 条:")
    print("=" * 60)
    for i, hit in enumerate(results, 1):
        text_preview = hit["text"][:120].replace("\n", " ")
        print(f"\n[结果 {i}] id={hit['id']}  score={hit['score']:.4f}")
        print(f"  text: {text_preview}{'...' if len(hit['text']) > 120 else ''}")
        if hit.get("metadata"):
            print(f"  metadata: {hit['metadata']}")

    # 7. 额外尝试混合检索（向量+BM25）
    logger.info("执行混合检索（向量 + BM25 RRF 融合）")
    hybrid_results = client.hybrid_search(
        query_text=query,
        query_vector=query_vector,
        top_k=top_k,
    )
    print("\n" + "=" * 60)
    print(f"混合检索结果 (RRF 融合):")
    print("=" * 60)
    for i, hit in enumerate(hybrid_results, 1):
        text_preview = hit["text"][:120].replace("\n", " ")
        print(f"\n[混合结果 {i}] id={hit['id']}  rrf_score={hit.get('rrf_score', 0):.4f}")
        print(f"  text: {text_preview}{'...' if len(hit['text']) > 120 else ''}")

    client.disconnect()
    logger.info("Milvus 连接已关闭，测试完成")


if __name__ == "__main__":
    main()
