"""
Advanced Visualization Module for the project
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ReviewVisualizer:
    """Create advanced visualizations for review analysis"""
    
    def __init__(self):
        self.setup_styles()
    
    def setup_styles(self):
        """Setup visualization styles"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_sentiment_dashboard(self, sentiment_df: pd.DataFrame) -> go.Figure:
        """Create comprehensive sentiment dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Sentiment Distribution by Bank',
                'Average Sentiment Scores',
                'Rating vs Sentiment Correlation',
                'Sentiment Over Time',
                'Confidence Distribution',
                'Sentiment by Rating'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "histogram"}, {"type": "bar"}]
            ]
        )
        
        # Sentiment distribution by bank
        bank_sentiment = pd.crosstab(sentiment_df['bank'], sentiment_df['sentiment_label'])
        for i, sentiment in enumerate(bank_sentiment.columns):
            fig.add_trace(
                go.Bar(name=sentiment, x=bank_sentiment.index, y=bank_sentiment[sentiment]),
                row=1, col=1
            )
        
        # Average sentiment by bank
        avg_sentiment = sentiment_df.groupby('bank')['sentiment_score'].mean()
        fig.add_trace(
            go.Bar(x=avg_sentiment.index, y=avg_sentiment.values, name='Avg Sentiment'),
            row=1, col=2
        )
        
        # Rating vs sentiment
        for bank in sentiment_df['bank'].unique():
            bank_data = sentiment_df[sentiment_df['bank'] == bank]
            fig.add_trace(
                go.Scatter(x=bank_data['rating'], y=bank_data['sentiment_score'],
                          mode='markers', name=bank, opacity=0.6),
                row=1, col=3
            )
        
        # Sentiment over time
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        monthly_sentiment = sentiment_df.groupby([
            pd.Grouper(key='date', freq='M'), 'bank'
        ])['sentiment_score'].mean().unstack()
        
        for bank in monthly_sentiment.columns:
            fig.add_trace(
                go.Scatter(x=monthly_sentiment.index, y=monthly_sentiment[bank],
                          mode='lines+markers', name=f'{bank} Trend'),
                row=2, col=1
            )
        
        # Confidence distribution
        fig.add_trace(
            go.Histogram(x=sentiment_df['confidence'], name='Confidence'),
            row=2, col=2
        )
        
        # Sentiment by rating
        rating_sentiment = pd.crosstab(sentiment_df['rating'], sentiment_df['sentiment_label'])
        for sentiment in rating_sentiment.columns:
            fig.add_trace(
                go.Bar(name=sentiment, x=rating_sentiment.index, y=rating_sentiment[sentiment]),
                row=2, col=3
            )
        
        fig.update_layout(height=800, title_text="Comprehensive Sentiment Analysis Dashboard")
        return fig
    
    def create_theme_wordclouds(self, thematic_results: Dict, max_words: int = 50) -> plt.Figure:
        """Create word clouds for each bank's themes"""
        n_banks = len(thematic_results)
        fig, axes = plt.subplots(1, n_banks, figsize=(20, 6))
        
        if n_banks == 1:
            axes = [axes]
        
        for i, (bank, analysis) in enumerate(thematic_results.items()):
            # Combine all keywords
            all_keywords = []
            themes = analysis.get('themes_identified', {})
            for keywords in themes.values():
                all_keywords.extend(keywords)
            
            if all_keywords:
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    max_words=max_words,
                    colormap='viridis'
                ).generate(' '.join(all_keywords))
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{bank}\nKey Themes', fontsize=14, fontweight='bold')
                axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_theme_comparison_heatmap(self, thematic_results: Dict) -> plt.Figure:
        """Create heatmap comparing theme strengths across banks"""
        theme_categories = [
            'User Interface & Experience',
            'Transaction Performance', 
            'Account Access & Security',
            'Customer Support',
            'App Reliability & Bugs',
            'Feature Requests'
        ]
        
        comparison_data = []
        for bank, analysis in thematic_results.items():
            themes = analysis.get('themes_identified', {})
            for category in theme_categories:
                strength = len(themes.get(category, []))
                comparison_data.append({'Bank': bank, 'Category': category, 'Strength': strength})
        
        comparison_df = pd.DataFrame(comparison_data)
        pivot_df = comparison_df.pivot(index='Category', columns='Bank', values='Strength')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='d', linewidths=0.5)
        plt.title('Theme Strength Comparison Across Banks', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()