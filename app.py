
import streamlit as st
import pandas as pd
import numpy as np
from numpy import load
from scipy.sparse import load_npz
from joblib import load as joblib_load
from content_based_filtering import content_recommendation
from collaborative_filtering import collaborative_recommendation
from hybrid_recommendations import HybridRecommenderSystem
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# --------- CSS for modern design and spacing ---------
st.markdown("""
<style>
.main-header {
    font-size: 3rem; font-weight: bold; text-align: center;
    background: linear-gradient(90deg, #1DB954, #1ed760);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem; border-radius: 15px;
    color: white; text-align: center;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    margin-bottom: 0.8rem;
}
.song-card {
    background: linear-gradient(135deg, #fff 60%, #e2ffe9 100%);
    padding: 1.25rem 1.2rem 1.2rem 1.2rem; border-radius: 16px;
    margin: 1.1rem 0;
    border-left: 5px solid #1DB954;
    box-shadow: 0 5px 18px rgba(29, 185, 84, 0.10);
}
.song-title { font-size: 1.15rem; font-weight: 700; color: #1DB954; margin-bottom: .18em;}
.song-artist { font-size: 1rem;  color: #444;  margin-bottom: .4em;}
.song-match { font-size: 0.93rem; color: #22bb77; font-weight: 600; }
.status-badges { margin: 0.1rem 0 0.5rem 0;}
.status-badge {
    padding: 0.25rem 0.85rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; display:inline-block;
    margin-right: 0.5rem; margin-bottom:0.22em;
}
.badge-liked { background: #fff0f1; color: #ff6b6b;}
.badge-disliked { background: #e0f5ff; color: #3498db;}
.badge-queued { background: #f3f0ff; color: #6c5ce7; }
.now-playing {
    background: linear-gradient(90deg, #1DB954, #1ed760);
    color: white; padding: 0.45rem 1.2rem; border-radius: 18px;
    font-size: 0.99rem; font-weight: 500;
    margin-bottom: 0.65rem; display:inline-block;
}
.next-up {
    background: #eafaf2;
    color: #1DB954; padding: 0.36rem 1.05rem; border-radius: 17px;
    font-size: 0.92rem; font-weight: 600;
    margin-bottom: 0.52rem; display:inline-block;
}
.audio-preview { margin-top:0.7rem; }
.audio-available { color: #1DB954; font-weight: 500;}
.audio-unavailable { color: #9ca3af; font-style: italic;}
/* BUTTON: style ALL st.button globally for spacing and appearance */
.stButton > button {
    margin-right: 13px; margin-left: 2px;
    padding: 0.5rem 1.2rem !important;
    border-radius: 16px !important;
    background: linear-gradient(90deg, #1DB954, #1ed760) !important;
    color: white; font-weight: 600; font-size:1rem;
    border: none; box-shadow: 0 1.5px 7px rgba(29,185,84,.09);
    transition: all 0.18s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #38e44e, #4cffde) !important;
    color: #212121;
}
</style>
""", unsafe_allow_html=True)


# ------ SESSION STATE ------
for key, default in {
    'actions': {},
    'playlists': {},
    'recommendation_history': [],
    'user_preferences': {},
    'liked_songs': set(),
    'disliked_songs': set(),
    'queued_songs': set(),
    'selected_mood': "Custom",
    'current_recommendations': pd.DataFrame(),
    'playlist_actions': {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------- CALLBACKS --------
def like_callback(song_key):
    st.session_state.liked_songs.add(song_key)
    st.session_state.disliked_songs.discard(song_key)
def dislike_callback(song_key):
    st.session_state.disliked_songs.add(song_key)
    st.session_state.liked_songs.discard(song_key)
def queue_callback(song_key):
    st.session_state.queued_songs.add(song_key)
def playlist_callback(song_key, rec):
    st.session_state.playlist_actions[song_key] = {
        'name': rec['name'],
        'artist': rec['artist'],
        'url': rec.get('spotify_preview_url', '')
    }

# DATA LOADING
@st.cache_data
def load_app_data():
    try:
        return {
            'songs_data': pd.read_csv("data/cleaned_data.csv"),
            'transformed_data': load_npz("data/transformed_data.npz"),
            'track_ids': load("data/track_ids.npy", allow_pickle=True),
            'filtered_data': pd.read_csv("data/collab_filtered_data.csv"),
            'interaction_matrix': load_npz("data/interaction_matrix.npz"),
            'transformed_hybrid_data': load_npz("data/transformed_hybrid_data.npz"),
            'transformer': joblib_load('transformer.joblib')
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
data = load_app_data()
if data is None:
    st.stop()
songs_data = data['songs_data']
transformed_data = data['transformed_data']
track_ids = data['track_ids']
filtered_data = data['filtered_data']
interaction_matrix = data['interaction_matrix']
transformed_hybrid_data = data['transformed_hybrid_data']
transformer = data['transformer']

MOOD_PRESETS = {
    "😊 Happy": {"danceability": 0.8, "energy": 0.75, "valence": 0.9, "tempo": 130.0, "acousticness": 0.2, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -5.0, "key": 2},
    "😢 Sad": {"danceability": 0.3, "energy": 0.25, "valence": 0.15, "tempo": 75.0, "acousticness": 0.7, "instrumentalness": 0.3, "liveness": 0.1, "loudness": -15.0, "key": 9},
    "🔥 Energetic": {"danceability": 0.9, "energy": 0.95, "valence": 0.8, "tempo": 150.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.4, "loudness": -3.0, "key": 7},
    "😴 Relaxed": {"danceability": 0.4, "energy": 0.3, "valence": 0.6, "tempo": 85.0, "acousticness": 0.6, "instrumentalness": 0.4, "liveness": 0.15, "loudness": -12.0, "key": 5},
    "💪 Workout": {"danceability": 0.85, "energy": 0.9, "valence": 0.75, "tempo": 140.0, "acousticness": 0.15, "instrumentalness": 0.1, "liveness": 0.3, "loudness": -4.0, "key": 1},
    "🎉 Party": {"danceability": 0.95, "energy": 0.85, "valence": 0.85, "tempo": 125.0, "acousticness": 0.1, "instrumentalness": 0.05, "liveness": 0.5, "loudness": -3.5, "key": 4},
    "🧘 Meditation": {"danceability": 0.2, "energy": 0.15, "valence": 0.65, "tempo": 60.0, "acousticness": 0.8, "instrumentalness": 0.7, "liveness": 0.05, "loudness": -20.0, "key": 3}
}
def get_mood_values(selected_mood):
    return MOOD_PRESETS.get(selected_mood, {"danceability": 0.5, "energy": 0.5, "valence": 0.5, "tempo": 120.0, "acousticness": 0.5, "instrumentalness": 0.5, "liveness": 0.2, "loudness": -10.0, "key": 0})

def recommend_by_available_features(user_preferences, songs_df, num_recs, exclude_songs=None):
    available_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'loudness', 'key']
    existing_features = [f for f in available_features if f in songs_df.columns]
    if not existing_features: return pd.DataFrame()
    filtered_songs = songs_df.copy()
    if exclude_songs:
        exclude_names = [song.split('_')[0] for song in exclude_songs]
        filtered_songs = filtered_songs[~filtered_songs['name'].isin(exclude_names)]
    user_vector = np.array([user_preferences.get(feature, 0.5) for feature in existing_features])
    song_vectors = filtered_songs[existing_features].values
    song_vectors = np.nan_to_num(song_vectors, nan=0.0)
    similarity_scores = cosine_similarity(user_vector.reshape(1, -1), song_vectors)[0]
    top_indices = np.argsort(similarity_scores)[-num_recs:][::-1]
    recs = filtered_songs.iloc[top_indices][['name','artist','spotify_preview_url']].copy()
    recs['similarity_score'] = similarity_scores[top_indices]
    return recs.reset_index(drop=True)

def display_static_recommendations(recommendations, show_scores=False):
    if recommendations.empty:
        st.warning("🎵 No recommendations found. Try adjusting your preferences!")
        return
    st.session_state.current_recommendations = recommendations
    for ind, rec in recommendations.iterrows():
        song_name = rec['name'].title()
        artist_name = rec['artist'].title()
        song_key = f"{rec['name']}_{rec['artist']}"
        st.markdown('<div class="song-card">', unsafe_allow_html=True)
        if ind == 0:
            st.markdown('<div class="now-playing">🎵 Currently Playing</div>', unsafe_allow_html=True)
        elif ind == 1:
            st.markdown('<div class="next-up">🎶 Next Up</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="song-title">{song_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="song-artist">by {artist_name}</div>', unsafe_allow_html=True)
        if show_scores and 'similarity_score' in rec:
            st.markdown(f'<div class="song-match">✨ Match: {rec["similarity_score"]:.1%}</div>', unsafe_allow_html=True)
        status_html = '<div class="status-badges">'
        if song_key in st.session_state.liked_songs:
            status_html += '<span class="status-badge badge-liked">💚 Liked</span>'
        if song_key in st.session_state.disliked_songs:
            status_html += '<span class="status-badge badge-disliked">💔 Disliked</span>'
        if song_key in st.session_state.queued_songs:
            status_html += '<span class="status-badge badge-queued">✅ Queued</span>'
        status_html += '</div>'
        st.markdown(status_html, unsafe_allow_html=True)
        cols = st.columns(4)
        if cols[0].button("❤️ Like", key=f"like_{ind}_{song_key}"):
            like_callback(song_key)
        if cols[1].button("👎 Pass", key=f"dislike_{ind}_{song_key}"):
            dislike_callback(song_key)
        if cols[2].button("➕ Queue", key=f"queue_{ind}_{song_key}"):
            queue_callback(song_key)
        if cols[3].button("📋 Playlist", key=f"playlist_{ind}_{song_key}"):
            playlist_callback(song_key, rec)
        st.markdown('<div class="audio-preview">', unsafe_allow_html=True)
        try:
            if pd.notna(rec['spotify_preview_url']) and rec['spotify_preview_url'].strip():
                st.markdown('<div class="audio-available">🔊 Audio Preview Available</div>', unsafe_allow_html=True)
                st.audio(rec['spotify_preview_url'])
            else:
                st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="audio-unavailable">🔇 Audio preview not available</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

st.sidebar.markdown('<h1 style="color: #1DB954;">🎵 Navigation</h1>', unsafe_allow_html=True)
page = st.sidebar.selectbox("Choose Page", [
    "🏠 Home", "📊 Analytics", "📝 Playlists", "⚙️ Settings"
])

# ===== ANALYTICS SUPPORT FUNCTIONS ======
def get_user_song_df(song_keys, songs_data):
    df = []
    for key in song_keys:
        try:
            name, artist = key.split("_", 1)
            row = songs_data[
                (songs_data["name"].str.lower().str.strip() == name.lower().strip()) &
                (songs_data["artist"].str.lower().str.strip() == artist.lower().strip())
            ]
            if len(row) > 0:
                df.append(row.iloc[0])
        except Exception:
            continue
    return pd.DataFrame(df) if df else pd.DataFrame()

def get_top(series, n=5):
    return series.value_counts().head(n)

def radar_chart(features, liked_means, disliked_means=None, queued_means=None):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[liked_means.get(feat,0) for feat in features],
        theta=features,
        fill='toself',
        name='Liked Songs'
    ))
    if disliked_means:
        fig.add_trace(go.Scatterpolar(
            r=[disliked_means.get(feat,0) for feat in features],
            theta=features,
            fill='toself',
            name='Disliked Songs'
        ))
    if queued_means:
        fig.add_trace(go.Scatterpolar(
            r=[queued_means.get(feat,0) for feat in features],
            theta=features,
            fill='toself',
            name='Queued Songs'
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )
    return fig

# ===== PAGES =====
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🎵 Hybrid Music Recommender</h1>', unsafe_allow_html=True)
    st.write('### 🎧 Discover your next favorite song')
    col1, col2 = st.columns(2)
    with col1:
        song_name = st.text_input('🎵 Enter a song name:')
    with col2:
        artist_name = st.text_input('🎤 Enter the artist name:')
    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.selectbox('📊 Number of recommendations:', [5, 10, 15, 20], index=1)
    with col2:
        recommender_type = st.selectbox('🤖 Algorithm:', ['Feature Sliders', 'Content-Based Filtering', 'Collaborative Filtering', 'Hybrid Recommender System'])
    with col3:
        exclude_heard = st.checkbox("🚫 Exclude liked/disliked songs", value=True)

    st.sidebar.header("🎵 Audio Features")
    st.sidebar.subheader("😊 Quick Mood Presets")
    mood_options = ["Custom"] + list(MOOD_PRESETS.keys())
    new_mood = st.sidebar.selectbox("Choose a mood:", mood_options, key="mood_selector")
    if new_mood != st.session_state.selected_mood:
        st.session_state.selected_mood = new_mood
        if new_mood != "Custom":
            st.rerun()
    mood_values = get_mood_values(st.session_state.selected_mood)
    if st.session_state.selected_mood != "Custom":
        st.sidebar.markdown(f'<div style="background: linear-gradient(90deg,#1DB954,#1ed760); color:white; font-weight:600; border-radius:16px; display:inline-block; padding:0.18rem 1.2rem;">🎭 {st.session_state.selected_mood} Active</div>', unsafe_allow_html=True)
    st.sidebar.subheader("🎛️ Core Features")
    danceability = st.sidebar.slider('💃 Danceability', 0.0, 1.0, float(mood_values["danceability"]))
    energy = st.sidebar.slider('⚡ Energy', 0.0, 1.0, float(mood_values["energy"]))
    valence = st.sidebar.slider('😊 Valence', 0.0, 1.0, float(mood_values["valence"]))
    tempo = st.sidebar.slider('🥁 Tempo', 60.0, 200.0, float(mood_values["tempo"]))
    with st.sidebar.expander("🎛️ Advanced Features"):
        acousticness = st.slider('🎸 Acousticness', 0.0, 1.0, float(mood_values["acousticness"]))
        instrumentalness = st.slider('🎹 Instrumentalness', 0.0, 1.0, float(mood_values["instrumentalness"]))
        liveness = st.slider('🎤 Liveness', 0.0, 1.0, float(mood_values["liveness"]))
        loudness = st.slider('🔊 Loudness', -60.0, 0.0, float(mood_values["loudness"]))
        key = st.selectbox('🎼 Key', list(range(12)), index=int(mood_values["key"]))

    st.write("**🎯 Current Preferences:**")
    pref_col1, pref_col2, pref_col3, pref_col4 = st.columns(4)
    with pref_col1:
        st.metric("💃 Danceability", f"{danceability:.2f}")
    with pref_col2:
        st.metric("⚡ Energy", f"{energy:.2f}")
    with pref_col3:
        st.metric("😊 Valence", f"{valence:.2f}")
    with pref_col4:
        st.metric("🥁 Tempo", f"{tempo:.0f} BPM")

    if st.button('🎵 Get Recommendations', type="primary", use_container_width=True):
        exclude_list = list(st.session_state.liked_songs) + list(st.session_state.disliked_songs) if exclude_heard else []
        with st.spinner('🎵 Finding your perfect songs...'):
            if recommender_type == 'Feature Sliders':
                user_preferences = {
                    'danceability': danceability, 'energy': energy, 'valence': valence,
                    'acousticness': acousticness, 'instrumentalness': instrumentalness,
                    'liveness': liveness, 'tempo': tempo, 'loudness': loudness, 'key': key
                }
                recommendations = recommend_by_available_features(user_preferences, songs_data, k, exclude_list)
                if not recommendations.empty:
                    st.success(f'🎵 Found songs matching your criteria!')
                    display_static_recommendations(recommendations, show_scores=True)
            elif recommender_type == 'Content-Based Filtering' and song_name and artist_name:
                song_name_low, artist_name_low = song_name.lower().strip(), artist_name.lower().strip()
                if ((songs_data["name"] == song_name_low) & (songs_data['artist'] == artist_name_low)).any():
                    recommendations = content_recommendation(song_name_low, artist_name_low, songs_data, transformed_data, k)
                    st.success(f'🎵 Found songs similar to **{song_name}** by **{artist_name}**!')
                    display_static_recommendations(recommendations)
                else:
                    st.warning("Song not found in database")
            elif recommender_type == 'Collaborative Filtering' and song_name and artist_name:
                song_name_low, artist_name_low = song_name.lower().strip(), artist_name.lower().strip()
                if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
                    recommendations = collaborative_recommendation(song_name_low, artist_name_low, track_ids, filtered_data, interaction_matrix, k)
                    st.success(f'🎵 Collaborative recommendations for **{song_name}** by **{artist_name}**')
                    display_static_recommendations(recommendations)
                else:
                    st.warning("Song not found in collaborative dataset")
            elif recommender_type == 'Hybrid Recommender System' and song_name and artist_name:
                song_name_low, artist_name_low = song_name.lower().strip(), artist_name.lower().strip()
                if ((filtered_data["name"] == song_name_low) & (filtered_data['artist'] == artist_name_low)).any():
                    diversity = st.slider("🔀 Diversity", 1, 9, 5)
                    content_based_weight = 1 - (diversity / 10)
                    recommender = HybridRecommenderSystem(number_of_recommendations=k, weight_content_based=content_based_weight)
                    recommendations = recommender.give_recommendations(song_name_low, artist_name_low, filtered_data, track_ids, transformed_hybrid_data, interaction_matrix)
                    st.success(f'🎵 Hybrid recommendations!')
                    display_static_recommendations(recommendations)
                else:
                    st.warning("Song not found in hybrid dataset")
            else:
                st.warning("Please select an algorithm and fill all required fields.")

    elif not st.session_state.current_recommendations.empty:
        st.markdown("### 🎵 Your Last Recommendations")
        display_static_recommendations(st.session_state.current_recommendations, show_scores=True)

elif page == "📊 Analytics":
    # ==== FULL ANALYTICS PAGE WITHOUT BAR CHART ====

    st.header("📊 Your Music Analytics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("❤️ Liked Songs", len(st.session_state.liked_songs))
    with col2:
        st.metric("👎 Disliked Songs", len(st.session_state.disliked_songs))
    with col3:
        st.metric("🎵 Queued Songs", len(st.session_state.queued_songs))
    with col4:
        total_inter = len(st.session_state.liked_songs) + len(st.session_state.disliked_songs) + len(st.session_state.queued_songs)
        hit_rate = (len(st.session_state.liked_songs) / total_inter * 100) if total_inter else 0
        st.metric("✅ Hit Rate", f"{hit_rate:.1f}%")

    # Get DataFrames for user actions
    liked_df = get_user_song_df(st.session_state.liked_songs, songs_data)
    disliked_df = get_user_song_df(st.session_state.disliked_songs, songs_data)
    queued_df = get_user_song_df(st.session_state.queued_songs, songs_data)

    # Top Artists, Genres, Diversity Metrics
    if not liked_df.empty:
        st.subheader("🎼 Your Favorite Genres & Artists")
        top_artists = get_top(liked_df["artist"])
        st.markdown("**Top Artists:**")
        for artist, count in top_artists.items():
            st.write(f"- {artist.title()} ({count})")
        if 'genre' in liked_df.columns:
            top_genres = get_top(liked_df["genre"])
            st.markdown("**Top Genres:**")
            for genre, count in top_genres.items():
                st.write(f"- {genre.title()} ({count})")
        st.markdown("**Diversity:**")
        artist_div = liked_df["artist"].nunique()
        genre_div = liked_df["genre"].nunique() if "genre" in liked_df else "n/a"
        st.write(f"- Distinct Artists: {artist_div}")
        if genre_div != "n/a":
            st.write(f"- Distinct Genres: {genre_div}")

    # Audio Feature Means
    core_features = [
        "danceability", "energy", "valence", "acousticness",
        "instrumentalness", "liveness", "tempo", "loudness"
    ]
    liked_means = liked_df[core_features].mean().to_dict() if not liked_df.empty else {}
    disliked_means = disliked_df[core_features].mean().to_dict() if not disliked_df.empty else {}
    queued_means = queued_df[core_features].mean().to_dict() if not queued_df.empty else {}

    st.subheader("🎚️ Average Audio Features")
    def feature_means_table(core_features, liked_means, disliked_means, queued_means):
        rows = []
        for f in core_features:
            rows.append({
                "Feature": f.title(),
                "Liked": round(liked_means.get(f, 0), 2) if liked_means else "-",
                "Disliked": round(disliked_means.get(f, 0), 2) if disliked_means else "-",
                "Queued": round(queued_means.get(f, 0), 2) if queued_means else "-",
            })
        return pd.DataFrame(rows)
    if not liked_df.empty or not disliked_df.empty or not queued_df.empty:
        st.dataframe(feature_means_table(core_features, liked_means, disliked_means, queued_means), use_container_width=True)
    else:
        st.info("No feature data for your actions yet.")

    # Radar chart if liked/disliked available
    if not liked_df.empty and not disliked_df.empty:
        st.plotly_chart(radar_chart(core_features, liked_means, disliked_means, queued_means))

    if not liked_df.empty:
        st.subheader("❤️ Your Liked Songs")
        for i, row in liked_df.iterrows():
            st.write(f"{i+1}. **{row['name'].title()}** by *{row['artist'].title()}*"
                     + (f" | _Genre:_ {row['genre']}" if 'genre' in row else "")
            )
            if pd.notna(row.get("spotify_preview_url", "")) and row["spotify_preview_url"].strip():
                st.audio(row["spotify_preview_url"])

    if not disliked_df.empty:
        st.subheader("💔 Your Disliked Songs")
        for i, row in disliked_df.iterrows():
            st.write(f"{i+1}. **{row['name'].title()}** by *{row['artist'].title()}*")

    if not queued_df.empty:
        st.subheader("🎵 Your Queued Songs")
        for i, row in queued_df.iterrows():
            st.write(f"{i+1}. **{row['name'].title()}** by *{row['artist'].title()}*")

elif page == "📝 Playlists":
    st.header("📝 My Playlists")
    col1, col2 = st.columns([2, 1])
    with col1:
        new_playlist_name = st.text_input("🆕 Create New Playlist")
    with col2:
        if st.button("➕ Create"):
            if new_playlist_name and new_playlist_name not in st.session_state.playlists:
                st.session_state.playlists[new_playlist_name] = []
                st.success(f"✅ Created playlist: {new_playlist_name}")
    if st.session_state.playlists:
        for playlist_name, songs in st.session_state.playlists.items():
            with st.expander(f"📋 {playlist_name} ({len(songs)} songs)"):
                if songs:
                    for i, song in enumerate(songs):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{song['name']}** by {song['artist']}")
                        with col2:
                            if song['url']:
                                st.audio(song['url'])
                else:
                    st.write("No songs yet. Add some from recommendations!")

elif page == "⚙️ Settings":
    st.header("⚙️ Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🗑️ Clear Liked"):
            st.session_state.liked_songs.clear()
            st.success("Liked songs cleared!")
    with col2:
        if st.button("🗑️ Clear Disliked"):
            st.session_state.disliked_songs.clear()
            st.success("Disliked songs cleared!")
    with col3:
        if st.button("🗑️ Clear Queue"):
            st.session_state.queued_songs.clear()
            st.success("Queue cleared!")
    with col4:
        if st.button("🗑️ Clear Playlists"):
            st.session_state.playlists.clear()
            st.success("Playlists cleared!")

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #1DB954; font-weight: bold;">'
    '🎵 Hybrid Music Recommender'
    '</div>',
    unsafe_allow_html=True
)
