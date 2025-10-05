"""Main analysis orchestration script."""
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src import config
from src.data_loading import load_and_merge_data
from src.similarity_analysis import create_hybrid_similarity_system, analyze_similarity_improvements, visualize_similarity_comparison
from src.era_analysis import analyze_era_evolution
from src.reputation_analysis import analyze_reputation_vs_others
from src.preference_analysis import analyze_personal_preferences
from src.topic_modeling import improved_lda_topic_modeling, improved_bertopic_modeling
from src.visualization import visualize_era_evolution, create_era_audio_profile, visualize_topics_comprehensive
from src.classification import era_classifier, cluster_songs
from src.utils import print_section_header
from src.berttopic import bertopic_lyrics_pipeline
import os
import pandas as pd

def main():
    print_section_header("TAYLOR SWIFT ANALYSIS: SIMILARITY & ERA EVOLUTION")
    if not os.path.exists('data/merged_df.csv'):
        # 1. Load data
        merged_df = load_and_merge_data(
            config.DATA_PATH,
            config.SPOTIFY_CSV,
            config.ALBUM_SONG_CSV
        )
        merged_df.to_csv('data/merged_df.csv')
    else: merged_df = pd.read_csv('data/merged_df.csv')
    
    # 3. Era Evolution
    #print_section_header("ERA EVOLUTION ANALYSIS")
    #df_with_eras, era_stats = analyze_era_evolution(merged_df)
    
    # 4. Topic Modeling
    print_section_header("TOPIC MODELING")
    bert_model, bert_topics, df_with_topics, bert_labels = bertopic_lyrics_pipeline( merged_df )

if __name__ == "__main__":
    main()
