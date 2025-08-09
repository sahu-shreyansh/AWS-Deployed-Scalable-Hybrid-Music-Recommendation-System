# 🎵 AWS-Deployed Scalable Hybrid Recommendation System

A real-time music recommendation system built using a hybrid approach that combines collaborative filtering and content-based filtering. The system is optimized for scalability, deployed on AWS, and capable of delivering personalized music recommendations based on user listening behavior and track metadata.

---

## 🚀 Features

- 🔄 **Hybrid Recommendation Engine**: Combines collaborative and content-based filtering for better accuracy.
- 💾 **Scalable Data Pipelines**: Efficient handling of large datasets using pandas, SciPy, and Dask.
- 🌐 **AWS Deployment**: Hosted using AWS EC2/S3 for scalability and real-time access.
- 🎯 **Content Filtering Module**: Uses TF-IDF and song metadata (genre, artist, tempo, etc.).
- 👥 **Collaborative Filtering Module**: Leverages user-item interaction matrices with matrix factorization or nearest-neighbor models.
- 🧠 **Personalized Output**: Generates top-N similar songs for a given track or user.
- 💡 **Streamlit Web App Interface**: Simple UI for testing and interacting with the recommender.

---

## 📦 Dataset

This project uses a publicly available dataset that combines Spotify metadata and Last.fm user listening history.

- **Source**: [🎧 Million Song Dataset - Spotify + Last.fm (Kaggle)](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm)

> 🔑 Note: You may need to log into Kaggle to access or download the dataset.  
> The raw files are not included in this repository due to size and licensing constraints.

---

## 📁 Project Structure

```

📂 Project
│
├── data/
│   ├── music\_info.csv
│   ├── User Listening History.csv
│   ├── interaction\_matrix.npz
│   ├── collab\_filtered\_data.csv
│   └── transformed\_hybrid\_data.npz
│
├── content\_based\_filtering.py
├── collaborative\_filtering.py
├── hybrid\_recommender.py
├── streamlit\_app.py
│
├── models/
│   └── saved\_vectorizers, encoders, etc.
│
├── .gitignore
├── requirements.txt
└── README.md

````

---

## ⚙️ Tech Stack

- **Python 3.x**
- **pandas, NumPy, SciPy, scikit-learn**
- **Streamlit** (for UI)
- **AWS EC2 / S3** (for hosting and storage)
- **Dask** (for handling large datasets)
- **Git, DVC** (for version control and data tracking)

---

## 🧠 How It Works

1. **Data Preprocessing**  
   Clean and transform metadata (e.g., genre, tempo, popularity) and user interaction history.

2. **Content-Based Filtering**  
   Uses TF-IDF on text features and cosine similarity on numerical/encoded features.

3. **Collaborative Filtering**  
   Generates a sparse user-item interaction matrix and applies similarity-based recommendation.

4. **Hybrid Recommender**  
   Combines both filtering methods with weighted scoring.

5. **Streamlit Front-End**  
   Users can input a song name and get top-N similar songs.

---

## 🛠️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hybrid-recommender.git
cd hybrid-recommender
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## ☁️ AWS Deployment

* App is deployed on an AWS EC2 instance.
* Data and models are stored on S3 buckets.
* Streamlit is reverse-proxied using Nginx (optional for production).

---

## 📊 Sample Output

> **Input song**: *Blinding Lights*
> **Recommended songs**:
>
> * Save Your Tears
> * In Your Eyes
> * Don’t Start Now
> * Physical
> * Levitating

---

## 🧪 Future Enhancements

* Integrate Spotify API for real-time listening data
* Use deep learning models (e.g., autoencoders)
* Improve cold-start handling for new users/songs
* Dockerize and deploy with CI/CD
* Add user authentication and playlists

---

## 🙋‍♂️ Author

**Pawan Bonde**
Big Data & ML Engineer | CDAC PG-DBDA
📧 [pssbonde@gmail.com](mailto:pssbonde@gmail.com)

---
=======
# AWS-Deployed-Scalable-Hybrid-Music-Recommendation-System

