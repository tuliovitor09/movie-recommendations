import pandas as pd
import numpy as np
from .loader import load_users, load_movies, load_interactions


# carrega os dados e converte para DataFrames do pandas
def load_data_frames():
    users = load_users()
    movies = load_movies()
    interactions = load_interactions()

    users_df = pd.DataFrame(users)
    movies_df = pd.DataFrame(movies)
    interactions_df = pd.DataFrame(interactions)

    return users_df, movies_df, interactions_df


# cria mapeamentos para estados e gêneros
# extrair categorias únicas
# criar mapeamento de texto para índice
# exemplo: {'SP': 0, 'RJ': 1, 'MG': 2} para estados e {'Ação': 0, 'Comédia': 1} para gêneros
def create_mappings(users_df, movies_df):
    states = users_df["estado"].unique()
    genders = movies_df["genero"].unique()

    states_map = {state: i for i, state in enumerate(states)}
    genders_map = {gender: i for i, gender in enumerate(genders)}

    return states_map, genders_map


# transforma um usuario em um vetor numérico usando idade e estado
def encode_user(user, states_map):
    age = user["idade"] / 100

    # cria um vetor cheio de zeros do tamanho do mapeamento de estados
    # e coloca 1 na posição correspondente ao estado do usuário
    state_vector = np.zeros(len(states_map))
    state_vector[states_map[user["estado"]]] = 1

    return np.concatenate(([age], state_vector))


# transforma um filme em um vetor numérico usando gênero, ano de lançamento e duração
def encode_movie(movie, genders_map):

    # cria um vetor cheio de zeros do tamanho do mapeamento de gêneros
    # e coloca 1 na posição correspondente ao gênero do filme
    gender_vector = np.zeros(len(genders_map))
    gender_vector[genders_map[movie["genero"]]] = 1

    year_norm = (movie["ano"] - 1980) / 50
    duration_norm = movie["duracao"] / 200

    return np.concatenate((gender_vector, [year_norm, duration_norm]))


def build_dataset():
    # carrega os dataframes
    df_users, df_movies, df_interactions = load_data_frames()

    # cria os encodings a partir do dataframe. Exemplos:
    # estado_map = {"SP": 0, "RJ": 1}
    # genero_map = {"Ação": 0, "Drama": 1}
    states_map, genders_map = create_mappings(df_users, df_movies)

    # transforma a coluna id em um índice do novo dicionário
    # converte o dataframe em dicionário
    users_dict = df_users.set_index("user_id").to_dict(orient="index")
    movies_dict = df_movies.set_index("movie_id").to_dict(orient="index")

    # inicializa os dataset: X entrada e Y saída
    X = []
    Y = []

    # _, ignora o índice e apenas percorre a lista
    #
    for _, interaction in df_interactions.iterrows():
        user = users_dict[interaction["user_id"]]
        movie = movies_dict[interaction["movie_id"]]

        # transforma os dados em texto em um vetor de números normalizado
        user_vector = encode_user(user, states_map)
        movie_vector = encode_movie(movie, genders_map)

        # junta tudo. Cada linha um exemplo de treino
        # y = gostou/não gostou
        X.append(np.concatenate((user_vector, movie_vector)))
        Y.append(interaction["rating"])

    # converte uma lista em numpy
    return np.array(X), np.array(Y)
