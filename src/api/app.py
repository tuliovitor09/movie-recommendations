from fastapi import FastAPI
from src.model.predictor import recommend_movies
from src.model.trainer import train_model
from src.data.processor import build_dataset
from src.domain.user import UserInput

app = FastAPI()

X, y = build_dataset()
model = train_model(X, y)


@app.post("/recommend")
def recommend(user: UserInput):
    user_data = {"idade": user.idade, "estado": user.estado}

    recommendations = recommend_movies(model, user_data)

    return {"usuario": user.nome, "recomendacoes": recommendations}
