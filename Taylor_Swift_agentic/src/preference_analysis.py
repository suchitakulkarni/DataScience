from src import *
def analyze_personal_preferences(df, rep_results, favorite_songs):
    """
    Analyze how your favorite songs align with Reputation's distinctive features.
    Includes normalization for plotting and a weighted alignment score.
    """
    print("\n" + "=" * 80)
    print("PERSONAL PREFERENCE ALIGNMENT (PPA)")
    print("=" * 80)

    # Clean song names for matching
    df['song_lower'] = df['Song_Name'].str.lower()
    fav_lower = [s.lower().strip() for s in favorite_songs]

    fav_df = df[df['song_lower'].isin(fav_lower)].copy()

    if fav_df.empty:
        print("No favorite songs matched. Check names in your list.")
        return None

    print(f"\nFound {len(fav_df)} of your favorites in dataset:")
    print(fav_df[['Song_Name', 'Album']])

    if rep_results is None or rep_results.empty:
        print("No Reputation results available for comparison.")
        return None

    # Focus on significant Reputation features
    sig_features = rep_results[rep_results['p_val'] < 0.05]['feature']
    aligned_results = []

    for feat in sig_features:
        if feat not in fav_df.columns:
            continue

        fav_mean = fav_df[feat].mean()
        rep_mean = rep_results.loc[rep_results['feature'] == feat, 'rep_mean'].values[0]
        others_mean = rep_results.loc[rep_results['feature'] == feat, 'others_mean'].values[0]

        # Compare closeness
        dist_to_rep = abs(fav_mean - rep_mean)
        dist_to_others = abs(fav_mean - others_mean)
        closer_to = "Reputation" if dist_to_rep < dist_to_others else "Others"

        # Alignment score for this feature (1 = fully aligned with Reputation, 0 = fully aligned with Others)
        if dist_to_rep + dist_to_others > 0:
            score = 1 - (dist_to_rep / (dist_to_rep + dist_to_others))
        else:
            score = 0.5  # if both distances zero, neutral

        aligned_results.append({
            'feature': feat,
            'fav_mean': fav_mean,
            'rep_mean': rep_mean,
            'others_mean': others_mean,
            'closer_to': closer_to,
            'alignment_score': score
        })

    aligned_df = pd.DataFrame(aligned_results)
    print("\nAlignment Results:")
    print(aligned_df)

    # Calculate overall weighted alignment score
    if not aligned_df.empty:
        overall_score = aligned_df['alignment_score'].mean()
        print(f"\nOverall Alignment Score (0=Others, 1=Reputation): {overall_score:.3f}")

    # Normalize for plotting
    plot_df = aligned_df.copy()
    all_vals = []
    for feat in plot_df['feature']:
        all_vals.extend([plot_df.loc[plot_df['feature'] == feat, 'rep_mean'].values[0],
                         plot_df.loc[plot_df['feature'] == feat, 'fav_mean'].values[0],
                         plot_df.loc[plot_df['feature'] == feat, 'others_mean'].values[0]])
    all_vals = np.array(all_vals)
    min_val, max_val = all_vals.min(), all_vals.max()

    def normalize(val):
        return (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in aligned_df.iterrows():
        rep_val = normalize(row['rep_mean'])
        fav_val = normalize(row['fav_mean'])
        oth_val = normalize(row['others_mean'])
        ax.plot(['Reputation', 'Favorites', 'Others'],
                [rep_val, fav_val, oth_val],
                marker='o', label=row['feature'])

    ax.set_title("Personal Preference Alignment with Reputation Features\n(Normalized Scale)")
    ax.set_ylabel("Normalized Feature Value (0–1)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/personal_alignment.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: results/personal_alignment.png")

    return aligned_df, overall_score


