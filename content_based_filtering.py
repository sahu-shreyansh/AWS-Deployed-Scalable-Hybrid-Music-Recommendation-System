import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz
import os

# Cleaned Data Path
CLEANED_DATA_PATH = "data/cleaned_data.csv"

# Separate column lists for clarity and robust preprocessing
numerical_cols = ["duration_ms","loudness","tempo", "danceability","energy","speechiness",
                  "acousticness","instrumentalness","liveness","valence"]

categorical_cols = ['artist', "time_signature", "key", 'year']

# Text column for TF-IDF Vectorization
tfidf_col = 'tags'


def train_transformer(data):
    """
    Trains a ColumnTransformer on the provided data and saves the transformer to a file.
    The ColumnTransformer applies the following transformations:
    - One-Hot Encoding on categorical columns.
    - Standard Scaling on all numerical columns.
    - TF-IDF Vectorization on the text column.
    
    This version is more robust by using a pipeline for numerical and categorical features.

    Parameters:
    data (pd.DataFrame): The input data to be transformed.
    Returns:
    None
    Saves:
    transformer.joblib: The trained ColumnTransformer object.
    """
    
    # Create the preprocessing steps
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    
    # The `TfidfVectorizer` requires a 1D array, so we'll handle it separately
    text_transformer = TfidfVectorizer(max_features=85)
    
    # Use ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            # Apply StandardScaler to all numerical columns
            ('num', numerical_transformer, numerical_cols),
            # Apply OneHotEncoder to all categorical columns
            ('cat', categorical_transformer, categorical_cols),
            # Apply TfidfVectorizer to the text column
            ('tfidf', text_transformer, [tfidf_col])
        ],
        # 'passthrough' will keep any columns not specified above
        remainder='drop',
        n_jobs=-1
    )

    # fit the transformer
    preprocessor.fit(data)

    # save the transformer
    joblib.dump(preprocessor, "transformer.joblib")
    

def transform_data(data):
    """
    Transforms the input data using a pre-trained transformer.
    Args:
        data (array-like): The data to be transformed.
    Returns:
        array-like: The transformed data.
    """
    # load the transformer
    transformer = joblib.load("transformer.joblib")
    
    # transform the data
    transformed_data = transformer.transform(data)
    
    return transformed_data


def save_transformed_data(transformed_data,save_path):
    """
    Save the transformed data to a specified file path.

    Parameters:
    transformed_data (scipy.sparse.csr_matrix): The transformed data to be saved.
    save_path (str): The file path where the transformed data will be saved.

    Returns:
    None
    """
    # save the transformed data
    save_npz(save_path, transformed_data)


def calculate_similarity_scores(input_vector, data):
    """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.
    Args:
        input_vector (array-like): The input vector for which similarity scores are to be calculated.
        data (array-like): The dataset against which the similarity scores are to be calculated.
    Returns:
        array-like: An array of similarity scores.
    """
    # calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, data)
    
    return similarity_scores


def content_recommendation(song_name,artist_name,songs_data, transformed_data, k=10):
    """
    Recommends top k songs similar to the given song based on content-based filtering.

    Parameters:
    song_name (str): The name of the song to base the recommendations on.
    artist_name (str): The name of the artist of the song.
    songs_data (DataFrame): The DataFrame containing song information.
    transformed_data (ndarray): The transformed data matrix for similarity calculations.
    k (int, optional): The number of similar songs to recommend. Default is 10.

    Returns:
    DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """
    # convert song name to lowercase
    song_name = song_name.lower()
    # convert the artist name to lowercase
    artist_name = artist_name.lower()
    # filter out the song from data
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    # get the index of song
    song_index = song_row.index[0]
    # generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)
    # calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data).flatten()
    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list


def main(data_path):
    """
    Test the recommendations for a given song using content-based filtering.

    Parameters:
    data_path (str): The path to the CSV file containing the song data.

    Returns:
    None: Prints the top k recommended songs based on content similarity.
    """
    # load the data
    data = pd.read_csv(data_path)
    # clean the data
    data_content_filtering = data_for_content_filtering(data)
    
    for col in categorical_cols:
        if col in data_content_filtering.columns:
            data_content_filtering[col] = data_content_filtering[col].astype('category')
    
    # train the transformer
    train_transformer(data_content_filtering)
    # transform the data
    transformed_data = transform_data(data_content_filtering)
    #save transformed data
    save_transformed_data(transformed_data,"data/transformed_data.npz")
    
if __name__ == "__main__":
    main(CLEANED_DATA_PATH)
