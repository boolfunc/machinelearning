# 通过pymilvus在milvus上创建article schema，包含三个字段，分别是id，title，content

from pymilvus import DataType, CollectionSchema, FieldSchema, Collection
from pymilvus import connections, utility

import numpy as np

connections.connect()

# 1. create collection
# We're going to create a collection with 4 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "id"    |   INT64  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "title"   |    VarChar |                  |      "a varchar field"       |
# +-+------------+------------+------------------+------------------------------+
# |3|"content"   | VarChar    |                  |  "varchar field"             |
# +-+------------+------------+------------------+------------------------------+
# |4|"embeddings"| FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+------------+------------+------------------+------------------------------+

# 如果article collection已经存在，删除

if utility.has_collection("article"):
    print("Delete collection `article`")
    Collection("article").drop()

print(utility.has_collection("article"))

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]

schema = CollectionSchema(fields, "article schema")

print("Create collection `article`")
article = Collection("article", schema, consistency_level="Strong")

# 2. insert data
# We're going to insert some entities into the collection.
# The entities consist of 4 fields:
#   - "id": key 
#   - "title": title of the article
#   - "content": content of the article
#   - "embeddings": embeddings of the article

rng = np.random.default_rng(seed=20000)


articles = [
    ["How to train your dragon", "How to train your dragon 2", "How to train your dragon 3", "How to train your dragon 4"],
    ["How to train your dragon content", "How to train your dragon 2 content", "How to train your dragon 3", "How to train your dragon 4 content"],
    rng.random((4, 8))
]

insert_result = article.insert(articles)
print("Insert {} articles into collection `article`.".format(insert_result.insert_count))
    


# 3. create index
# We're going to create an index for the collection.
# The index is a float vector field named "embeddings" with L2 metric type.

print("Create index for collection `article`.")
article.create_index("embeddings", {"metric_type": "L2"})

print("finished to create index")