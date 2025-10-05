from src.preprocessing import preprocess_lyrics_enhanced
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.config as config

import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def get_or_create_embeddings(df, embedding_model_name, embeddings_path="results/song_embeddings.npz"):
    """Load embeddings if they exist and match model; otherwise compute and save new ones."""
    meta_path = embeddings_path.replace(".npz", "_meta.json")
    if os.path.exists(embeddings_path) and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta.get("embedding_model_name") == embedding_model_name:
            print(f"Loading cached embeddings from {embeddings_path}")
            data = np.load(embeddings_path, allow_pickle=True)
            embeddings = data["embeddings"]
            return embeddings
        else:
            print(
                f"Model name mismatch - expected {meta['embedding_model_name']}, got {embedding_model_name}. Recomputing...")

    print(f"Computing new embeddings using model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(df["clean_lyrics"].tolist(), show_progress_bar=True)

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.savez(embeddings_path, embeddings=embeddings)

    meta = {
        "embedding_model_name": embedding_model_name,
        "num_documents": len(df),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings and metadata to {embeddings_path}")
    return embeddings


def bertopic_lyrics_pipeline(df, embedding_model_name="all-mpnet-base-v2",
                             min_cluster_size=5, diversity_threshold=0.5,
                             visualize=True):
    """
    Optimized BERTopic pipeline for song lyrics.

    Args:
        df (pd.DataFrame): Must contain a 'lyrics' column.
        embedding_model_name (str): SentenceTransformer model name.
        min_cluster_size (int): HDBSCAN minimum cluster size.
        diversity_threshold (float): Threshold to merge similar topics (lower = more merging).
        visualize (bool): Whether to save visualizations.

    Returns:
        best_model (BERTopic)
        topics (list[int])
        probs (np.ndarray)
        df (pd.DataFrame) with topic assignments and strengths
        topic_labels (list[str])
    """
    # ------------------------
    # 1. Preprocess lyrics
    # ------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING LYRICS")
    print("=" * 60)
    df["clean_lyrics"] = df["lyrics"].astype(str).apply(preprocess_lyrics_enhanced)

    # ------------------------
    # 2. Sentence embeddings
    # ------------------------
    print("\nEmbedding lyrics with:", embedding_model_name)
    embeddings = get_or_create_embeddings(df, embedding_model_name)

    # ------------------------
    # 3. UMAP for dimensionality reduction
    # ------------------------
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # ------------------------
    # 4. HDBSCAN clustering
    # ------------------------
    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='leaf',  # Changed from 'eom' - more aggressive clustering
        min_samples=1,  # Added - more lenient
        prediction_data=True
    )

    # ------------------------
    # 5. BERTopic initialization with all components
    # ------------------------
    print("\nInitializing BERTopic with seed topics...")
    vectorizer_model = CountVectorizer(
        stop_words=list(config.LYRIC_STOPWORDS),
        ngram_range=(1, 2),
        min_df=2,  # Must appear in at least 2 songs
        max_df=0.5,  # Ignore if in more than 50% of songs
        max_features=1000  # Limit vocabulary size
    )

    topic_model = BERTopic(
        embedding_model=embedding_model_name,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,  # ADD THIS LINE
        seed_topic_list=config.seed_topic_list,
        calculate_probabilities=True,
        verbose=True
    )

    # ------------------------
    # 6. Fit model
    # ------------------------
    print("\nFitting BERTopic model...")
    topics, probs = topic_model.fit_transform(df["clean_lyrics"], embeddings)

    print(f"\nInitial number of topics: {len(set(topics)) - (1 if -1 in topics else 0)}")

    # ------------------------
    # 7. Reduce topics to merge similar ones
    # ------------------------
    if len(set(topics)) > 2:  # Only reduce if we have multiple topics
        print(f"\nReducing topics with diversity threshold: {diversity_threshold}")
        topic_model.reduce_topics(
            df["clean_lyrics"],
            nr_topics="6"
        )
        topics = topic_model.topics_
        probs = topic_model.probabilities_
        print(f"Topics after reduction: {len(set(topics)) - (1 if -1 in topics else 0)}")

    # ------------------------
    # 8. Assign topics to dataframe
    # ------------------------
    df['id'] = topics
    if probs is not None:
        df['strength'] = probs.max(axis=1)
    else:
        df['strength'] = 0.0

    # ------------------------
    # 9. Topic labels
    # ------------------------
    topic_labels = []
    topic_info = topic_model.get_topic_info()
    print("\n" + "=" * 60)
    print("DISCOVERED TOPICS")
    print("=" * 60)

    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)
        if words:
            top_words = [word for word, _ in words[:5]]
            label = ", ".join(top_words)
            topic_labels.append(label)
            count = len([t for t in topics if t == topic_id])
            print(f"Topic {topic_id}: {label} (n={count})")

    # ------------------------
    # 10. Visualizations
    # ------------------------
    if visualize:
        os.makedirs("results", exist_ok=True)

        try:
            fig1 = topic_model.visualize_topics()
            fig1.write_html("results/bertopic_topics.html")
            print("\nSaved topic visualization to results/bertopic_topics.html")
        except Exception as e:
            print(f"\nCouldn't create topic visualization: {e}")

        try:
            fig2 = topic_model.visualize_heatmap()
            fig2.write_html("results/bertopic_heatmap.html")
            print("Saved heatmap to results/bertopic_heatmap.html")
        except Exception as e:
            print(f"Couldn't create heatmap: {e}")

        try:
            fig3 = topic_model.visualize_barchart(top_n_topics=8)
            fig3.write_html("results/bertopic_barchart.html")
            print("Saved barchart to results/bertopic_barchart.html")
        except Exception as e:
            print(f"Couldn't create barchart: {e}")

    # ------------------------
    # 11. Summary
    # ------------------------
    print("\n" + "=" * 60)
    print("BERTOPIC PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total topics (excluding outliers): {len(topic_labels)}")
    print(f"Outlier songs (topic -1): {len([t for t in topics if t == -1])}")

    return topic_model, topics, probs, df, topic_labels