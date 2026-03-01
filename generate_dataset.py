"""
EchoShield: Synthetic Dataset Generator
Generates fake social media engagement data for bot vs organic detection.
Records: 5000 | Type: Synthetic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N = 5000
N_BOT = 2000
N_ORGANIC = 3000

def generate_timestamps(n, is_bot=False):
    base = datetime(2024, 1, 1)
    if is_bot:
        # Bots post at very regular intervals
        intervals = np.random.choice([60, 120, 300], size=n)  # fixed intervals in seconds
        timestamps = [base + timedelta(seconds=int(sum(intervals[:i+1]))) for i in range(n)]
    else:
        # Humans post irregularly
        intervals = np.random.exponential(scale=3600, size=n)
        timestamps = [base + timedelta(seconds=int(sum(intervals[:i+1]))) for i in range(n)]
    return timestamps

def generate_records(n, is_bot):
    label = 1 if is_bot else 0
    records = []
    for _ in range(n):
        if is_bot:
            post_interval_std = np.random.uniform(0, 30)          # very regular
            likes_per_hour = np.random.randint(80, 200)
            comments_per_hour = np.random.randint(50, 150)
            repost_ratio = np.random.uniform(0.7, 1.0)
            follower_following_ratio = np.random.uniform(0.01, 0.3)
            account_age_days = np.random.randint(1, 180)
            profile_completeness = np.random.uniform(0.1, 0.4)
            unique_word_ratio = np.random.uniform(0.1, 0.3)       # repetitive language
            avg_sentiment_score = np.random.uniform(-0.1, 0.1)    # neutral/scripted
            burst_flag = np.random.choice([0, 1], p=[0.2, 0.8])   # frequent bursts
            network_overlap_score = np.random.uniform(0.6, 1.0)   # coordinated network
            night_activity_ratio = np.random.uniform(0.4, 0.9)    # active at odd hours
            hashtag_reuse_rate = np.random.uniform(0.6, 1.0)
            avg_response_time_sec = np.random.uniform(1, 10)       # instant replies
        else:
            post_interval_std = np.random.uniform(200, 2000)
            likes_per_hour = np.random.randint(1, 40)
            comments_per_hour = np.random.randint(0, 20)
            repost_ratio = np.random.uniform(0.0, 0.4)
            follower_following_ratio = np.random.uniform(0.5, 5.0)
            account_age_days = np.random.randint(180, 3650)
            profile_completeness = np.random.uniform(0.5, 1.0)
            unique_word_ratio = np.random.uniform(0.5, 1.0)
            avg_sentiment_score = np.random.uniform(-1.0, 1.0)
            burst_flag = np.random.choice([0, 1], p=[0.85, 0.15])
            network_overlap_score = np.random.uniform(0.0, 0.4)
            night_activity_ratio = np.random.uniform(0.0, 0.3)
            hashtag_reuse_rate = np.random.uniform(0.0, 0.4)
            avg_response_time_sec = np.random.uniform(60, 86400)

        records.append({
            "post_interval_std": round(post_interval_std, 2),
            "likes_per_hour": likes_per_hour,
            "comments_per_hour": comments_per_hour,
            "repost_ratio": round(repost_ratio, 3),
            "follower_following_ratio": round(follower_following_ratio, 3),
            "account_age_days": account_age_days,
            "profile_completeness": round(profile_completeness, 2),
            "unique_word_ratio": round(unique_word_ratio, 2),
            "avg_sentiment_score": round(avg_sentiment_score, 3),
            "burst_flag": burst_flag,
            "network_overlap_score": round(network_overlap_score, 3),
            "night_activity_ratio": round(night_activity_ratio, 3),
            "hashtag_reuse_rate": round(hashtag_reuse_rate, 3),
            "avg_response_time_sec": round(avg_response_time_sec, 2),
            "is_bot": label
        })
    return records

bot_records = generate_records(N_BOT, is_bot=True)
organic_records = generate_records(N_ORGANIC, is_bot=False)

all_records = bot_records + organic_records
random.shuffle(all_records)

df = pd.DataFrame(all_records)
df.index.name = "user_id"
df.reset_index(inplace=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/echoshield_dataset.csv", index=False)

print(f"✅ Dataset generated: {len(df)} records")
print(f"   Bots: {df['is_bot'].sum()} | Organic: {(df['is_bot']==0).sum()}")
print(f"   Saved to: data/echoshield_dataset.csv")
print(df.head())
