from text2vec import SentenceModel

model = SentenceModel()

vec = model.encode('hello world')

print(vec.shape)

print(vec)