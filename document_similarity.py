from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Narendra Modi is the Prime Minister of India known for his charismatic leadership and economic reforms.",

"Sonia Gandhi is the former president of the Indian National Congress famous for her resilience and political acumen.",

"Rahul Gandhi, also known as the face of the Indian youth in politics, has advocated for social change and transparency.",

"Amit Shah is known for his strategic planning and organizational skills in Indian politics.",

"Arvind Kejriwal is the Chief Minister of Delhi known for his activism and focus on education and healthcare reforms."
]

query = 'tell me about narendra modi'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)