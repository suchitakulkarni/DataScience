"""Main analysis orchestration script."""
import sys, os
from pathlib import Path
import pandas as pd

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


def main():
    print_section_header("TAYLOR SWIFT ANALYSIS: SIMILARITY & ERA EVOLUTION")
    
    # 1. Load data
    if not os.path.exists(f"{config.RESULTS_DIR}/merged_records.csv"):
        merged_df = load_and_merge_data(
            config.DATA_DIR,
            config.SPOTIFY_CSV,
            config.ALBUM_SONG_CSV
        )
        merged_df.to_csv(f"{config.RESULTS_DIR}/merged_records.csv", index=False)
    else:
        merged_df = pd.read_csv(f"{config.RESULTS_DIR}/merged_records.csv")
    
    # 2. Similarity Analysis
    print_section_header("SIMILARITY ANALYSIS")
    similarity_results = create_hybrid_similarity_system(merged_df)
    interesting_cases = analyze_similarity_improvements(similarity_results)
    visualize_similarity_comparison(similarity_results, song_idx=0)
    
    # 3. Era Evolution
    print_section_header("ERA EVOLUTION ANALYSIS")
    if not os.path.exists(config.RESULTS_DIR+"/songs_with_eras.csv"):
        df_with_eras, era_stats = analyze_era_evolution(merged_df)
        if df_with_eras is not None:
            df_with_eras.to_csv(f"{config.RESULTS_DIR}/songs_with_eras.csv", index=False)
    else: df_with_eras = pd.read_csv(config.RESULTS_DIR+"/songs_with_eras.csv")
    visualize_era_evolution(df_with_eras)
    create_era_audio_profile(df_with_eras)
    
    # 4. Topic Modeling
    print_section_header("TOPIC MODELING")
    '''lda_model, X_counts, vectorizer, df_with_topics, lda_labels = improved_lda_topic_modeling(
        df_with_eras, 
        n_topics_range=config.TOPIC_RANGE
    )
    bert_model, bert_topics, df_with_topics, bert_labels = improved_bertopic_modeling(
        df_with_topics,
        n_topics_range=config.TOPIC_RANGE
    )
    visualize_topics_comprehensive(
        df_with_eras, lda_model, bert_model, 
        lda_labels, bert_labels, X_counts
    )'''
    bert_model, bert_topics, probs, df_with_topics, bert_labels = bertopic_lyrics_pipeline(
        df_with_eras, visualize=True, min_cluster_size=4
    )
    df_with_topics.to_csv(f"{config.RESULTS_DIR}/songs_with_topics.csv")
    print("*"*60)
    print("improved BERT modelling?")
    #print (bert_labels)
    #sys.exit()

    # 5. ML Analysis
    print_section_header("ML ANALYSIS")
    clf = era_classifier(df_with_eras, similarity_results['lyric_similarity'])
    clusters = cluster_songs(similarity_results['lyric_similarity'],df_with_eras, n_clusters=5)
    
    # 6. Reputation Analysis
    print_section_header("REPUTATION ANALYSIS")
    rep_results = analyze_reputation_vs_others(df_with_eras)
    if rep_results is not None:
        rep_results.to_csv(f"{config.RESULTS_DIR}/reputation_feature_differences.csv", index=False)
    
    # 7. Personal Preferences
    favorite_songs = ["Delicate", "All Too Well", "Style", "Cruel Summer", "22"]
    print_section_header("PERSONAL PREFERENCE ANALYSIS")
    alignment_df, overall_score = analyze_personal_preferences(
        df_with_eras, rep_results, favorite_songs
    )
    if alignment_df is not None:
        alignment_df.to_csv(f"{config.RESULTS_DIR}/personal_alignment.csv", index=False)
        print(f"Your Overall Reputation Alignment Score: {overall_score:.3f}")
    
    # 8. Agentic Options
    print("\n" + "="*80)
    print("AGENTIC ANALYSIS OPTIONS (Ollama)")
    print("="*80)
    print(f"\nUsing model: {config.MODEL}")
    print("\nAvailable agentic features:")
    print("  1. Conversational Analysis Assistant")
    print("  2. Agentic Recommendation System")
    print("  3. Multi-Agent Song Analysis")
    print("  4. Tool-Using Agent")
    print("  5. Memory-Enhanced Agent")
    print("\nRun the agent demo with:")
    print("  python demo_agents.py")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
