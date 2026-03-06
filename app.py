import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EchoShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0D1B2A; }
    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; }

    h1, h2, h3, h4 { color: #FFFFFF !important; }

    .metric-card {
        background: #112233;
        border: 1px solid #1E6FA5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2.2em; font-weight: bold; color: #00C9C8; }
    .metric-label { font-size: 0.9em; color: #8BA3B8; margin-top: 4px; }

    .risk-high   { background:#2A0D0D; border:2px solid #E8394A; border-radius:10px; padding:16px; text-align:center; }
    .risk-medium { background:#2A1F0D; border:2px solid #F5A623; border-radius:10px; padding:16px; text-align:center; }
    .risk-low    { background:#0D2A1A; border:2px solid #00C896; border-radius:10px; padding:16px; text-align:center; }

    .risk-high-text   { color:#E8394A; font-size:1.6em; font-weight:bold; }
    .risk-medium-text { color:#F5A623; font-size:1.6em; font-weight:bold; }
    .risk-low-text    { color:#00C896; font-size:1.6em; font-weight:bold; }

    .insight-box {
        background: #081520;
        border-left: 4px solid #00C9C8;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #E8F4FD;
        font-size: 0.95em;
    }
    .stSlider > div > div { color: #00C9C8; }
    section[data-testid="stSidebar"] { background-color: #0A1628 !important; }
    .stSelectbox label, .stSlider label { color: #8BA3B8 !important; }
</style>
""", unsafe_allow_html=True)


# ── DATA & MODEL (cached) ──────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    N_BOT, N_ORG = 2000, 3000
    def make(n, bot):
        if bot:
            return pd.DataFrame({
                'post_interval_std':       np.random.uniform(0,30,n),
                'likes_per_hour':          np.random.randint(80,200,n),
                'comments_per_hour':       np.random.randint(50,150,n),
                'repost_ratio':            np.random.uniform(0.7,1.0,n),
                'follower_following_ratio':np.random.uniform(0.01,0.3,n),
                'account_age_days':        np.random.randint(1,180,n),
                'profile_completeness':    np.random.uniform(0.1,0.4,n),
                'unique_word_ratio':       np.random.uniform(0.1,0.3,n),
                'avg_sentiment_score':     np.random.uniform(-0.1,0.1,n),
                'burst_flag':              np.random.choice([0,1],n,p=[0.2,0.8]),
                'network_overlap_score':   np.random.uniform(0.6,1.0,n),
                'night_activity_ratio':    np.random.uniform(0.4,0.9,n),
                'hashtag_reuse_rate':      np.random.uniform(0.6,1.0,n),
                'avg_response_time_sec':   np.random.uniform(1,10,n),
                'is_bot': 1
            })
        else:
            return pd.DataFrame({
                'post_interval_std':       np.random.uniform(200,2000,n),
                'likes_per_hour':          np.random.randint(1,40,n),
                'comments_per_hour':       np.random.randint(0,20,n),
                'repost_ratio':            np.random.uniform(0.0,0.4,n),
                'follower_following_ratio':np.random.uniform(0.5,5.0,n),
                'account_age_days':        np.random.randint(180,3650,n),
                'profile_completeness':    np.random.uniform(0.5,1.0,n),
                'unique_word_ratio':       np.random.uniform(0.5,1.0,n),
                'avg_sentiment_score':     np.random.uniform(-1.0,1.0,n),
                'burst_flag':              np.random.choice([0,1],n,p=[0.85,0.15]),
                'network_overlap_score':   np.random.uniform(0.0,0.4,n),
                'night_activity_ratio':    np.random.uniform(0.0,0.3,n),
                'hashtag_reuse_rate':      np.random.uniform(0.0,0.4,n),
                'avg_response_time_sec':   np.random.uniform(60,86400,n),
                'is_bot': 0
            })
    df = pd.concat([make(N_BOT,True), make(N_ORG,False)]).sample(frac=1, random_state=42).reset_index(drop=True)
    df['engagement_intensity'] = df['likes_per_hour'] + df['comments_per_hour']
    df['automation_score']     = (1/(df['post_interval_std']+1))*1000
    df['social_credibility']   = df['follower_following_ratio']*df['profile_completeness']
    df['linguistic_bot_signal']= 1 - df['unique_word_ratio']
    df['coordination_index']   = df['network_overlap_score']*df['hashtag_reuse_rate']
    return df

@st.cache_resource
def train_model(df):
    FEATS = [
        'post_interval_std','likes_per_hour','comments_per_hour','repost_ratio',
        'follower_following_ratio','account_age_days','profile_completeness',
        'unique_word_ratio','avg_sentiment_score','burst_flag','network_overlap_score',
        'night_activity_ratio','hashtag_reuse_rate','avg_response_time_sec',
        'engagement_intensity','automation_score','social_credibility',
        'linguistic_bot_signal','coordination_index'
    ]
    X, y = df[FEATS], df['is_bot']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    return rf, sc, FEATS, X_test, X_test_sc, y_test

df   = generate_data()
rf, sc, FEATS, X_test, X_test_sc, y_test = train_model(df)
y_prob = rf.predict_proba(X_test_sc)[:,1]


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ EchoShield")
    st.markdown("<p style='color:#8BA3B8;font-size:0.85em;'>Uncovering Coordinated Inauthentic Behaviour</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview", "🔍 Live Account Checker", "📊 Dataset Analysis", "📈 Model Performance"])
    st.markdown("---")
    st.markdown("<p style='color:#8BA3B8;font-size:0.8em;'>OrgX Behavioural Analytics Hackathon 2025<br>Problem Statement 3</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🛡️ EchoShield Dashboard")
    st.markdown("<p style='color:#8BA3B8;'>Behavioural Analytics System for Fake Engagement Detection</p>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">5,000</div><div class="metric-label">Total Accounts</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">99%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">0.999</div><div class="metric-label">ROC-AUC Score</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">19</div><div class="metric-label">Behavioural Features</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔍 How EchoShield Works")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-box">📥 <b>Step 1 — Data Collection</b><br>Collect 14 raw behavioural signals per account including posting patterns, network structure, and language metrics.</div>
        <div class="insight-box">⚙️ <b>Step 2 — Feature Engineering</b><br>Engineer 5 composite signals: automation score, coordination index, social credibility, engagement intensity, linguistic bot signal.</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">🤖 <b>Step 3 — Model Prediction</b><br>Random Forest classifier (200 trees) trained on 4,000 accounts predicts bot probability for each account.</div>
        <div class="insight-box">📊 <b>Step 4 — Risk Output</b><br>Each account receives an Authenticity Score (0–100), Bot Probability, and Risk Label for action.</div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#112233')
        ax.set_facecolor('#112233')
        sizes  = [3000, 2000]
        colors = ['#00C896', '#E8394A']
        labels = ['Organic (60%)', 'Bot (40%)']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.0f%%', startangle=140,
                                           textprops={'color':'white','fontsize':11})
        for at in autotexts: at.set_color('white')
        ax.set_title('Class Distribution', color='white', fontsize=13, pad=12)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#112233')
        ax.set_facecolor('#112233')
        bots = df[df['is_bot']==1]['post_interval_std']
        orgs = df[df['is_bot']==0]['post_interval_std'].clip(upper=500)
        ax.hist(bots, bins=30, color='#E8394A', alpha=0.7, label='Bot')
        ax.hist(orgs, bins=30, color='#00C896', alpha=0.7, label='Organic')
        ax.set_title('Posting Regularity (Std Dev)', color='white', fontsize=13)
        ax.set_xlabel('Std Dev of Post Interval (sec)', color='#8BA3B8')
        ax.set_ylabel('Count', color='#8BA3B8')
        ax.tick_params(colors='#8BA3B8')
        for spine in ax.spines.values(): spine.set_edgecolor('#1E3A5A')
        ax.legend(facecolor='#0D1B2A', labelcolor='white')
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE ACCOUNT CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Account Checker":
    st.markdown("# 🔍 Live Account Checker")
    st.markdown("<p style='color:#8BA3B8;'>Adjust the behavioural sliders to simulate an account and get an instant bot prediction.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ⏱️ Activity Patterns")
        post_interval_std     = st.slider("Post Interval Std Dev (sec)", 0, 2000, 500)
        night_activity_ratio  = st.slider("Night Activity Ratio", 0.0, 1.0, 0.2)
        burst_flag            = st.selectbox("Engagement Burst Activity?", [0,1], format_func=lambda x: "Yes" if x else "No")
        avg_response_time_sec = st.slider("Avg Response Time (sec)", 1, 86400, 3600)

        st.markdown("#### 💬 Engagement Metrics")
        likes_per_hour    = st.slider("Likes Per Hour", 0, 200, 20)
        comments_per_hour = st.slider("Comments Per Hour", 0, 150, 10)
        repost_ratio      = st.slider("Repost Ratio", 0.0, 1.0, 0.2)

    with col2:
        st.markdown("#### 🕸️ Network & Profile")
        follower_following_ratio = st.slider("Follower/Following Ratio", 0.01, 5.0, 1.0)
        account_age_days         = st.slider("Account Age (days)", 1, 3650, 365)
        profile_completeness     = st.slider("Profile Completeness", 0.0, 1.0, 0.7)
        network_overlap_score    = st.slider("Network Overlap Score", 0.0, 1.0, 0.2)

        st.markdown("#### 📝 Language & Content")
        unique_word_ratio    = st.slider("Unique Word Ratio", 0.0, 1.0, 0.6)
        avg_sentiment_score  = st.slider("Avg Sentiment Score", -1.0, 1.0, 0.0)
        hashtag_reuse_rate   = st.slider("Hashtag Reuse Rate", 0.0, 1.0, 0.2)

    # Engineered features
    engagement_intensity  = likes_per_hour + comments_per_hour
    automation_score      = (1 / (post_interval_std + 1)) * 1000
    social_credibility    = follower_following_ratio * profile_completeness
    linguistic_bot_signal = 1 - unique_word_ratio
    coordination_index    = network_overlap_score * hashtag_reuse_rate

    input_data = pd.DataFrame([[
        post_interval_std, likes_per_hour, comments_per_hour, repost_ratio,
        follower_following_ratio, account_age_days, profile_completeness,
        unique_word_ratio, avg_sentiment_score, burst_flag, network_overlap_score,
        night_activity_ratio, hashtag_reuse_rate, avg_response_time_sec,
        engagement_intensity, automation_score, social_credibility,
        linguistic_bot_signal, coordination_index
    ]], columns=FEATS)

    input_sc   = sc.transform(input_data)
    bot_prob   = rf.predict_proba(input_sc)[0][1]
    auth_score = round((1 - bot_prob) * 100, 1)

    if   bot_prob >= 0.7: risk, css, emoji = "Likely Bot",     "risk-high",   "🤖"
    elif bot_prob >= 0.3: risk, css, emoji = "Uncertain",      "risk-medium", "⚠️"
    else:                 risk, css, emoji = "Likely Organic", "risk-low",    "✅"

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{auth_score}</div><div class="metric-label">Authenticity Score (0–100)</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{bot_prob:.3f}</div><div class="metric-label">Bot Probability</div></div>', unsafe_allow_html=True)
    with r3:
        text_class = risk.lower().replace(" ", "-")
        st.markdown(f'<div class="{css}"><div class="risk-{text_class.split("-")[0] if "organic" not in text_class else "low"}-text">{emoji} {risk}</div></div>', unsafe_allow_html=True)

    # Gauge chart
    fig, ax = plt.subplots(figsize=(6,3), facecolor='#112233')
    ax.set_facecolor('#112233')
    color = '#E8394A' if bot_prob>=0.7 else ('#F5A623' if bot_prob>=0.3 else '#00C896')
    ax.barh(['Bot Probability'], [bot_prob], color=color, height=0.4)
    ax.barh(['Bot Probability'], [1-bot_prob], left=[bot_prob], color='#1E3A5A', height=0.4)
    ax.set_xlim(0,1)
    ax.axvline(0.3, color='#F5A623', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(0.7, color='#E8394A', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'Bot Probability: {bot_prob:.3f}', color='white', fontsize=13)
    ax.tick_params(colors='#8BA3B8')
    for spine in ax.spines.values(): spine.set_edgecolor('#1E3A5A')
    ax.set_xlabel('Probability', color='#8BA3B8')
    st.pyplot(fig)
    plt.close()

    # Top behavioural signals
    st.markdown("### 🔑 Key Behavioural Signals for This Account")
    importances = rf.feature_importances_
    feat_vals   = input_data.values[0]
    top_idx     = np.argsort(importances)[::-1][:6]
    sig_col1, sig_col2 = st.columns(2)
    for i, idx in enumerate(top_idx):
        box = sig_col1 if i % 2 == 0 else sig_col2
        with box:
            st.markdown(f'<div class="insight-box"><b>{FEATS[idx]}</b><br>Value: <span style="color:#00C9C8">{feat_vals[idx]:.3f}</span> &nbsp;|&nbsp; Importance: <span style="color:#F5A623">{importances[idx]:.3f}</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Analysis":
    st.markdown("# 📊 Dataset Analysis")
    st.markdown("<p style='color:#8BA3B8;'>Explore behavioural feature distributions across bot and organic accounts.</p>", unsafe_allow_html=True)
    st.markdown("---")

    selected = st.selectbox("Select Feature to Explore", [
        'post_interval_std','likes_per_hour','network_overlap_score',
        'hashtag_reuse_rate','avg_response_time_sec','unique_word_ratio',
        'night_activity_ratio','automation_score','coordination_index'
    ])

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#112233')
        ax.set_facecolor('#112233')
        df[df['is_bot']==0][selected].hist(ax=ax, bins=30, color='#00C896', alpha=0.7, label='Organic')
        df[df['is_bot']==1][selected].hist(ax=ax, bins=30, color='#E8394A', alpha=0.7, label='Bot')
        ax.set_title(f'{selected} Distribution', color='white', fontsize=12)
        ax.tick_params(colors='#8BA3B8')
        for spine in ax.spines.values(): spine.set_edgecolor('#1E3A5A')
        ax.legend(facecolor='#0D1B2A', labelcolor='white')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#112233')
        ax.set_facecolor('#112233')
        data = [df[df['is_bot']==0][selected].values, df[df['is_bot']==1][selected].values]
        bp = ax.boxplot(data, patch_artist=True, labels=['Organic','Bot'])
        bp['boxes'][0].set_facecolor('#00C896')
        bp['boxes'][1].set_facecolor('#E8394A')
        for item in ['whiskers','caps','medians','fliers']:
            for patch in bp[item]: patch.set_color('white')
        ax.set_title(f'{selected} Boxplot', color='white', fontsize=12)
        ax.tick_params(colors='#8BA3B8')
        for spine in ax.spines.values(): spine.set_edgecolor('#1E3A5A')
        st.pyplot(fig)
        plt.close()

    st.markdown("### 🔥 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,7), facecolor='#112233')
    ax.set_facecolor('#112233')
    corr = df[FEATS + ['is_bot']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, annot_kws={'size':7},
                ax=ax, cbar_kws={'shrink':0.8})
    ax.set_title('Feature Correlation Heatmap', color='white', fontsize=14, pad=12)
    ax.tick_params(colors='white')
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

    st.markdown("# 📈 Model Performance")
    st.markdown("<p style='color:#8BA3B8;'>Evaluation metrics and visualizations for the Random Forest classifier.</p>", unsafe_allow_html=True)
    st.markdown("---")

    y_pred = rf.predict(X_test_sc)

    m1, m2, m3, m4 = st.columns(4)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy_score(y_test,y_pred)*100:.1f}%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{roc_auc_score(y_test,y_prob):.3f}</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{precision_score(y_test,y_pred):.2f}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{recall_score(y_test,y_pred):.2f}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#112233')
        ax.set_facecolor('#112233')
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Organic','Bot'], yticklabels=['Organic','Bot'],
                    annot_kws={'size':14,'color':'white'})
        ax.set_title('Confusion Matrix', color='white', fontsize=13)
        ax.tick_params(colors='white')
        ax.set_xlabel('Predicted', color='#8BA3B8')
        ax.set_ylabel('Actual', color='#8BA3B8')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### ROC Curve")
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#112233')
        ax.set_facecolor('#112233')
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color='#00C9C8', lw=2, label=f'AUC = {auc:.3f}')
        ax.plot([0,1],[0,1],'--', color='#8BA3B8', label='Random')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#00C9C8')
        ax.set_title('ROC Curve', color='white', fontsize=13)
        ax.tick_params(colors='#8BA3B8')
        for spine in ax.spines.values(): spine.set_edgecolor('#1E3A5A')
        ax.legend(facecolor='#0D1B2A', labelcolor='white')
        ax.set_xlabel('False Positive Rate', color='#8BA3B8')
        ax.set_ylabel('True Positive Rate', color='#8BA3B8')
        st.pyplot(fig)
        plt.close()

    st.markdown("#### 🏆 Feature Importance")
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#112233')
    ax.set_facecolor('#112233')
    imp_df = pd.DataFrame({'Feature': FEATS, 'Importance': rf.feature_importances_}).sort_values('Importance')
    colors = ['#E8394A' if i >= len(imp_df)-5 else '#1E6FA5' for i in range(len(imp_df))]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors)
    ax.set_title('Feature Importance (Red = Top 5)', color='white', fontsize=13)
    ax.tick_params(colors='#8BA3B8')
    for spine in ax.spines.values(): spine.set_edgecolor('#1E3A5A')
    ax.set_xlabel('Importance Score', color='#8BA3B8')
    st.pyplot(fig)
    plt.close()
