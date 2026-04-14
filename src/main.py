from data.processor import build_dataset
from model.predictor import recommend_movies
from model.trainer import train_model

# cria dataset
X, y = build_dataset()

# treina o modelo
model = train_model(X, y)

# faz as recomendações
recs = recommend_movies(model, user_id=5, top_k=5)

for r in recs:
    print(r)
