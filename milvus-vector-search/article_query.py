# 通过pymilvus接口查询向量,从article中查询
import time
import numpy as np

from pymilvus import DataType, CollectionSchema, FieldSchema, Collection
from pymilvus import connections, utility

search_latency_fmt = "search latency = {:.4f}s"

connections.connect()

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]

schema = CollectionSchema(fields, "article schema")

print("Create collection `article`")
article = Collection("article", schema, consistency_level="Strong")

article.load()

# search vectors
# 通过pymilvus接口查询向量,从article中查询

rng = np.random.default_rng(seed=20000)

vectors_to_search = rng.random((1, 8))
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = article.search(vectors_to_search, "embeddings", search_params, limit=1, output_fields=["id", "title", "content"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, title field: {hit.entity.get('title')}")
print(search_latency_fmt.format(end_time - start_time))

