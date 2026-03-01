# 🛡️ EchoShield: Uncovering Coordinated Inauthentic Behaviour on Social Platforms

> **Behavioural Analytics Hackathon — Problem Statement 3**  
> Detecting fake/bot-driven social media engagement using behavioural pattern analysis and machine learning.

---

## 📌 Project Overview

Social media platforms are increasingly polluted by bots and coordinated fake engagement that distorts genuine behavioural signals. **EchoShield** is a behavioural analytics system that:

- 🔍 Differentiates **organic vs artificial** social media engagement
- 🤖 Detects **coordinated inauthentic behaviour** patterns
- 📊 Outputs an **Authenticity Score (0–100)** and **Bot Probability** per account
- 🧠 Explains the key **behavioural anomalies** driving each prediction

---

## 🗂️ Repository Structure

```
EchoShield/
├── data/
│   ├── echoshield_dataset.csv        # Synthetic dataset (5000 records)
│   └── echoshield_predictions.csv    # Model output with scores
├── plots/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── model_evaluation.png
│   ├── feature_importance.png
│   └── authenticity_scores.png
├── generate_dataset.py               # Synthetic data generation script
├── EchoShield.ipynb                  # Main analysis & model notebook
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📦 Dataset Information

| Property | Details |
|---|---|
| **Dataset Type** | Synthetic |
| **Total Records** | 5,000 |
| **Bot Records** | 2,000 (40%) |
| **Organic Records** | 3,000 (60%) |
| **Features** | 14 raw + 5 engineered = 19 total |

### Why Synthetic?
No publicly available labelled behavioural dataset exists that captures the full range of coordinated inauthentic behaviour signals (network overlap, posting regularity, linguistic consistency) at the granularity needed for this task. Real platform data is private and proprietary.

### How It Was Generated
The dataset was simulated using domain-informed statistical rules:
- **Bots** follow tight Gaussian distributions for posting intervals, high network overlap, high hashtag reuse, near-instant response times
- **Organic users** follow exponential distributions for activity gaps, varied language, natural follower ratios
- Distributions are grounded in academic literature on bot behaviour detection

### Feature Description

| Feature | Description |
|---|---|
| `post_interval_std` | Std deviation of time between posts (bots = very low) |
| `likes_per_hour` | Average likes given per hour |
| `comments_per_hour` | Average comments per hour |
| `repost_ratio` | Ratio of reposts to original content |
| `follower_following_ratio` | Follower count / following count |
| `account_age_days` | Age of the account in days |
| `profile_completeness` | Profile fill score (0–1) |
| `unique_word_ratio` | Ratio of unique words in posts (low = repetitive/scripted) |
| `avg_sentiment_score` | Average sentiment of posts (-1 to 1) |
| `burst_flag` | Whether account exhibits engagement bursts (0/1) |
| `network_overlap_score` | Overlap with known suspicious accounts (0–1) |
| `night_activity_ratio` | Proportion of activity during night hours |
| `hashtag_reuse_rate` | Rate of repetitive hashtag usage |
| `avg_response_time_sec` | Average time to respond/react to content |

### Engineered Features

| Feature | Logic |
|---|---|
| `engagement_intensity` | likes_per_hour + comments_per_hour |
| `automation_score` | 1 / (post_interval_std + 1) × 1000 |
| `social_credibility` | follower_following_ratio × profile_completeness |
| `linguistic_bot_signal` | 1 − unique_word_ratio |
| `coordination_index` | network_overlap_score × hashtag_reuse_rate |

---

## 🤖 Model Approach

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest (200 trees) | ~99% | ~0.999 |
| Gradient Boosting (150 trees) | ~98% | ~0.998 |

**Best Model:** Random Forest Classifier  
**Key Hyperparameters:** n_estimators=200, max_depth=15

---

## 📊 Output Format

Each account receives:
- **Authenticity Score** (0–100): Higher = more likely organic
- **Bot Probability** (0.0–1.0): Probability of being a bot
- **Risk Label**: `Likely Organic` / `Uncertain` / `Likely Bot`

---

## 🔑 Key Behavioural Insights

1. **Posting regularity** (low std dev in intervals) is the #1 bot signal
2. **Network coordination** combined with hashtag reuse strongly flags inauthentic campaigns
3. **Near-instant response times** (1–10 sec) are impossible for humans at scale
4. **Linguistic repetitiveness** (low unique word ratio) reveals scripted content
5. **Night activity ratios** expose 24/7 automated operation

---

## 🚀 Getting Started

```bash
# 1. Clone repository
git clone https://github.com/yourusername/EchoShield.git
cd EchoShield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python generate_dataset.py

# 4. Open and run the notebook
jupyter notebook EchoShield.ipynb
```

---

## 📋 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## ⚠️ Important Notes

- All data is **synthetically generated** — no real user data was used
- The model is a **proof-of-concept** for the hackathon context
- Real-world deployment would require real platform data and additional privacy considerations

---

## 👥 Team

**EchoShield Team** — Behavioural Analytics Hackathon 2025

---

*Built for the OrgX Behavioural Analytics Hackathon*
