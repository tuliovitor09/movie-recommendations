import numpy as np
import pandas as pd
from src.data.loader import load_movies, load_users
from src.data.processor import create_mappings, encode_movie, encode_user


def recommend_movies(model, user, top_k=5):

    # carregar usuarios
    users = load_users()
    movies = load_movies()

    df_users = pd.DataFrame(users)
    df_movies = pd.DataFrame(movies)

    # criar mappings
    states_map, genders_map = create_mappings(df_users, df_movies)

    # vetor do usuario
    user_vector = encode_user(user, states_map)

    scores = []

    # testar todos os filmes
    for _, movie in df_movies.iterrows():
        movie_vector = encode_movie(movie, genders_map)

        # juntar user + movie
        input_vector = np.concatenate((user_vector, movie_vector)).reshape(1, -1)

        # prever
        prediction = model.predict(input_vector, verbose=0)[0][0]

        scores.append(
            {
                "movie_id": movie["movie_id"],
                "titulo": movie["nome"],
                "score": f"{prediction * 100:.2f}%",
            }
        )

    # ordenar por score
    scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)

    # retornar top_k
    return scores_sorted[:top_k]
