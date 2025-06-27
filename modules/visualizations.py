import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_strategy_distribution(df):
    st.subheader("📈 Strategy Distribution")
    strategy_counts = df['strategy'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(strategy_counts, labels=strategy_counts.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

def plot_clicks_vs_relevance(df):
    st.subheader("📉 Clicks vs Relevance Score")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="relevance_score", y="Clicks", hue="strategy", ax=ax2)
    st.pyplot(fig2)
