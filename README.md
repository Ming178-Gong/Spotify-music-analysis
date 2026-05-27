# 🎵 Spotify Music Data Analysis

> A beginner data science project exploring the relationships between audio features and song popularity using Python and machine learning — powered by 114,000 Spotify tracks.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Analysis Pipeline](#analysis-pipeline)
- [Machine Learning Model](#machine-learning-model)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

This project dives into the **Spotify Tracks Dataset** to understand what makes a song popular. Starting from raw data exploration all the way through a predictive machine learning model, it walks through a complete end-to-end data analysis workflow — ideal for anyone learning data science fundamentals.

**Key questions explored:**
- Which audio features correlate most strongly with song popularity?
- How do different genres differ in energy, danceability, and tempo?
- Can we predict a song's popularity from its audio characteristics alone?

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) |
| **Author** | Maharshi Pandya |
| **Size** | ~114,000 songs |
| **File** | `dataset.csv` |

### Audio Features Included

| Feature | Description |
|---|---|
| `popularity` | Score from 0–100 based on recent streams |
| `danceability` | How suitable a track is for dancing (0.0–1.0) |
| `energy` | Perceptual measure of intensity and activity (0.0–1.0) |
| `loudness` | Overall loudness in decibels (dB) |
| `tempo` | Estimated beats per minute (BPM) |
| `acousticness` | Confidence measure of whether the track is acoustic (0.0–1.0) |
| `instrumentalness` | Predicts whether a track has no vocals (0.0–1.0) |
| `speechiness` | Detects spoken words in a track (0.0–1.0) |
| `valence` | Musical positivity conveyed by a track (0.0–1.0) |
| `track_genre` | Genre classification label |

---

## Analysis Pipeline

### 1. Data Cleaning
- Load CSV with Pandas and inspect structure (`head`, `columns`, `dtypes`)
- Detect and handle null values
- Remove duplicate entries to ensure data integrity

### 2. Exploratory Data Analysis (EDA)
- Compute summary statistics per genre
- Group tracks by genre and compare mean feature values
- Identify trends across different musical styles

### 3. Data Visualization
- **Scatter plots** — visualize relationships between pairs of audio features
- **Bar charts** — compare feature averages across genres
- **Correlation heatmap** — identify which features move together using Seaborn

### 4. Correlation Analysis
- Build a full correlation matrix across all numeric features
- Highlight the strongest predictors of `popularity`

---

## Machine Learning Model

### Model: Linear Regression

A **Linear Regression** model is trained to predict song popularity from audio features.

**Input features (8 predictors):**

```
danceability · energy · loudness · speechiness
acousticness · instrumentalness · valence · tempo
```

**Target variable:** `popularity`

### Training Setup

| Parameter | Value |
|---|---|
| Train / Test split | 80% / 20% |
| Random state | 42 |
| Evaluation metric | Mean Squared Error (MSE) |

### Workflow

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
```

After training, feature coefficients are extracted to rank how much each audio attribute influences the predicted popularity score.

---

## Tech Stack

| Library | Purpose |
|---|---|
| [Python 3.8+](https://www.python.org/) | Core language |
| [Pandas](https://pandas.pydata.org/) | Data loading, cleaning, and manipulation |
| [Matplotlib](https://matplotlib.org/) | Base plotting and visualizations |
| [Seaborn](https://seaborn.pydata.org/) | Statistical plots and correlation heatmaps |
| [scikit-learn](https://scikit-learn.org/) | Machine learning model and evaluation |
| [Jupyter Notebook](https://jupyter.org/) | Interactive development environment |

---

## Project Structure

```
Spotify-music-analysis/
├── music-analysis.ipynb           # Main analysis notebook (EDA → ML)
├── dataset.csv                    # Spotify Tracks Dataset (~114k rows)
├── The_commands_explaintions.txt  # Code reference and command notes
├── LICENSE                        # MIT License
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ming178-Gong/Spotify-music-analysis.git
cd Spotify-music-analysis

# 2. Install dependencies
pip install pandas matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook music-analysis.ipynb
```

The notebook is self-contained — run all cells from top to bottom to reproduce the full analysis and model training.

---

## Acknowledgements

- Dataset provided by [Maharshi Pandya](https://www.kaggle.com/maharshipandya) on Kaggle
- Built with open-source Python data science libraries

---

## License

Distributed under the [MIT License](LICENSE). Free to use, modify, and share.
