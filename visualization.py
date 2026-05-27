import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

def plot_sentiment_distribution(df, sentiment_column='Sentiment'):
    """
    Returns a Plotly pie chart of sentiment distribution.
    """
    sentiment_counts = df[sentiment_column].value_counts().reset_index()
    sentiment_counts.columns = [sentiment_column, 'Count']
    
    color_map = {
        'Positive': '#00CC96',
        'Neutral': '#636EFA',
        'Negative': '#EF553B'
    }
    
    fig = px.pie(
        sentiment_counts, 
        names=sentiment_column, 
        values='Count',
        title='Sentiment Distribution',
        color=sentiment_column,
        color_discrete_map=color_map,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_wordcloud(text_series, title="Word Cloud"):
    """
    Returns a matplotlib figure containing a WordCloud.
    """
    text = " ".join(review for review in text_series)
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    
    return fig

def plot_model_comparison(metrics_dict):
    """
    Plots a bar chart comparing accuracy of different models.
    metrics_dict should be like: {'Logistic Regression': 0.85, 'Naive Bayes': 0.80, ...}
    """
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    
    fig = go.Figure([go.Bar(
        x=models, 
        y=accuracies,
        text=[f"{val:.2f}" for val in accuracies],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)]
    )])
    
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Models",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1])
    )
    
    return fig
