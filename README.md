# Spotify Track-to-Playlist Recommendation System

## Overview
This project leverages the Spotify Million Playlist Dataset (MPD) to build an advanced machine learning pipeline. While traditional recommendation systems predict tracks for a user, this engine flips the paradigm: it predicts the optimal **Playlists for a specific Track**. We achieved this due to the parallel nature of the ALS algorithm by flipping the matrix variable assignment of "users" to tracks and "items" to playlists, using sparse.csr_matrix((data, (tracks, playlists))) 
                            ^user    ^items to predict
                            
This Item-to-User (Track-to-Playlist) approach simulates a real-world music industry tool designed to help independent artists and record labels pitch their music to the most relevant algorithmic and editorial curators.

For the purpose of this dataset, we referenced the Spotify 2018 Recsys Challenge, as well as GH user bnsreenu's tutorial on ALS Recommender systems.

Here is where you can source the data for the ongoing challenge (requires AICrowd profile, and to click 'Participate' on the following page):
[Download the Dataset Here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
[ACM Challenge Documentation](https://www.recsyschallenge.com/2018/)

References on ALS Recommender Systems:
[Github: bnsreenu/Recommender-Systems](https://github.com/bnsreenu/Recommender-Systems)
[Youtube: On ALS Recommender Systems](https://www.youtube.com/playlist?list=PLZsOBAyNTZwaeoQXRbx3fIN6-eCcatT3p)

## Key Features & Technical Architecture
* **High-Performance ETL:** A custom pipeline that processes nested JSON metadata into memory-efficient `.parquet` files using PyArrow.
* **Optimized Data Structures:** Utilizes a custom `Tracks x Playlists` Compressed Sparse Row (CSR) matrix, allowing for rapid $O(1)$ track query times and preventing memory bottlenecking during matrix transpositions.
* **Alternating Least Squares (ALS):** Implements collaborative filtering via matrix factorization using the `implicit` C++ backend to extract latent audio and curator features.
* **Inference Engine:** Ranks recommendations using both raw ALS confidence scores and calculated Cosine Similarity to ensure thematic fit across a diverse pool of playlist sizes.

## Experimentation & Model Evaluation
This project relies on strict quantitative ranking metrics to prove model efficacy rather than standard classification accuracy.

* **Evaluation Metrics:** The model's predictive power is evaluated using **MAP@K** (Mean Average Precision) and **NDCG@K** (Normalized Discounted Cumulative Gain). These metrics heavily penalize the model if the true target playlists are not ranked at the absolute top of the recommendation list.
* **Business-Logic Baseline:** To prove algorithmic lift, the ALS model is rigorously tested against a non-personalized "Follower Count" baseline. This mimics a naive industry strategy of simply pitching a song to the playlists with the highest follower counts. 
* **MLflow Tracking:** Hyperparameter tuning (factors, regularization, iterations) is centralized using MLflow. Every model run automatically logs these parameters alongside the resulting MAP and NDCG scores to ensure reproducible experiment tracking. Model artifacts and matrix translations are serialized and saved via `joblib`.

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. The required libraries include:
**`implicit`, `pandas`, `numpy`, `scipy`, `pyarrow`, `mlflow`, `joblib`, `tqdm`**

### Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-repo/spotify-recsys.git](https://github.com/your-repo/spotify-recsys.git)
   cd spotify-recsys

## How to Run (currently)


1. Open the Pipeline:
Launch Jupyter and open ALS Script.ipynb.

2. Current Notebook Modular Pipeline:
Execute the notebook sequentially. The modules will:

### **Module 0: ETL (Extract, Transform, Load)**
JSON Processing: Iterates through the raw Spotify JSON files to extract track and playlist information.

**NOTE: As is, you will need to point the data_path to a local folder after downloading and extracting the MPD Dataset linked above.

Memory Optimization: Converts the heavy, nested JSON data into three distinct, flattened Parquet files (mpd_interactions, mpd_track_metadata, and mpd_playlist_metadata) which are significantly faster to read and require less RAM.

Here, you can change the amount of data being loaded from the example dataset linked above
Future scope: Need to implement a data folder and remove / isolate this module for useability.

### **Module 1: Sparse Matrix Construction**
Data Slicing: Loads the interaction data from the Parquet files; it includes a DEV_MODE toggle to load a smaller subset (e.g., 50,000 playlists) for faster testing.

Categorical Encoding: Maps long Spotify URIs to integer IDs so the machine learning model can process them mathematically.

Matrix Creation: Builds a Tracks x Playlists Compressed Sparse Row (CSR) matrix. This specific orientation is optimized for querying a song to find playlist leads.

### **Module 2: ALS Engine Definition**
Model Initialization: Defines the training function for the Alternating Least Squares (ALS) algorithm.

Hyperparameter Setup: Configures the "latent factors" (hidden musical features), regularization, and iterations that the model will use to learn the relationships between tracks and curators.

### **Module 3: Generate Evaluation Framework and Train Algorithm**
3.1 Popularity Baseline: Calculates a "dumb" baseline by recommending the most followed playlists in the dataset regardless of the song's vibe, providing a benchmark to prove the AI's effectiveness.

3.2 Native Evaluation: Uses ranking metrics—MAP@K and NDCG@K—to measure how accurately the ALS model can predict which playlists a hidden set of tracks actually belongs to.

#### **Once you have trained the model, if you are not experimenting with hyper parameters, you need only re-run the following cells, changing your desired track ID for playlist suggestions.** 

### **Module 4: Execution & MLflow Tracking**
Orchestration: This is the main control center that triggers the matrix building, baseline calculation, and ALS training in sequence.

Experiment Logging: Automatically logs every run to MLflow, capturing the hyperparameters used and the resulting evaluation scores so you can compare different model versions in a dashboard.

Persistence: Saves the trained model and translation dictionaries to disk using joblib for later use in the inference engine.

### **Module 5: Inference & Lead Generation** 
Recommendation Logic: Defines the recommend_playlists_for_track function, which uses the trained model to find the most mathematically similar playlists for a given song.

Business Filtering: Filters the raw results to only include playlists that meet a minimum "Follower Count" requirement to ensure the leads are worth an artist's time.

User Interface: Calculates Cosine Similarity percentages for each recommendation, providing a "Match %" that is easy for a human to interpret.


### **Experimentation: View MLflow Tracking:**
To view the experiment dashboard and compare model runs, open your terminal in the project directory and run:

Bash
mlflow ui
Navigate to http://127.0.0.1:5000 in your web browser.

## Team Members
- Raturi, Karthik
- Tapp, Cole
- Ubana, Jonh Ver
- Nejman, Jonathon
- Myers, Grey
