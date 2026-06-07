"""Direct RAG chain test bypassing all network dependencies."""
import sys, os, traceback
sys.path.insert(0, '/app')

# Set env like the API does
os.environ['DASHSCOPE_API_KEY'] = 'sk-23ab824edf4e4136acfc12803e2891b8'
os.environ['MILVUS_HOST'] = 'milvus-standalone'
os.environ['MILVUS_PORT'] = '19530'

print("Step 1: Loading config and clients...")
from src.config_loader import load_config
from src.embedding_client import DashScopeEmbeddingClient
from src.milvus_client import MilvusClient
from src.llm_client import DashScopeLLMClient
from src.rag_chain import RAGChain

cfg = load_config()
mc = cfg.get('milvus', {})

print("Step 2: Connecting to Milvus...")
milvus = MilvusClient(
    host=os.environ.get('MILVUS_HOST', mc.get('host', 'milvus-standalone')),
    port=int(os.environ.get('MILVUS_PORT', mc.get('port', 19530))),
    collection_name=mc.get('collection_name', 'medical_chunks'),
    vector_dim=int(mc.get('vector_dim', 1536)),
)
milvus.connect()
print("  Milvus connected OK")

print("Step 3: Initializing clients...")
embed = DashScopeEmbeddingClient(api_key=os.environ['DASHSCOPE_API_KEY'])
llm = DashScopeLLMClient(api_key=os.environ['DASHSCOPE_API_KEY'])

print("Step 4: Building RAG chain (with semantic cache disabled)...")
chain = RAGChain(embed, milvus, llm)
# Monkey-patch to disable cache since it has connection issues
if chain.semantic_cache:
    chain.semantic_cache = None
    print("  Semantic cache disabled")

print("Step 5: Running RAG answer...")
try:
    result = chain.answer('急性心肌梗死的典型症状有哪些？')
    print("\n=== SUCCESS ===")
    print("Answer preview:", result.get('answer', '')[:300])
    print("Cache hit:", result.get('cache_hit'))
    print("Sources count:", len(result.get('sources', [])))
except Exception as e:
    print("\n=== FAILED ===")
    print("Error type:", type(e).__name__)
    print("Error message:", str(e))
    traceback.print_exc()
