"""Agentic recommendation system with explainability - Ollama version."""
"""Agentic recommendation system with explainability - Ollama version."""
import numpy as np
import pandas as pd
from typing import Dict
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient  # Relative import
from src import config


class RecommendationAgent:
    """Agent that provides and explains song recommendations using Ollama."""
    
    def __init__(self, similarity_results, df, model: str = config.MODEL):
        if config.USE_OPENAI == False:
            self.client = OllamaClient(model=model)
        else: self.client = OpenAIClient()
        self.similarity_results = similarity_results
        self.df = df
    
    def recommend_with_reasoning(self, song_name: str, n_recommendations: int = 5) -> Dict:
        """Get recommendations and explain the reasoning."""
        
        # Find song
        matches = self.df[self.df['Formatted_name'].str.lower() == song_name.lower()]
        if matches.empty:
            return {'error': f"Song '{song_name}' not found in dataset."}
        
        song_idx = matches.index[0]
        song_data = self.df.iloc[song_idx]
        
        # Get similar songs
        hybrid_sim = self.similarity_results['hybrid_similarity']
        similar_indices = np.argsort(hybrid_sim[song_idx])[-n_recommendations-1:-1][::-1]
        recommendations = self.df.iloc[similar_indices]
        
        # Build context for LLM
        context = f"""You are a music recommendation expert. Analyze why these songs are similar.
            USER'S SONG:
            - Title: {song_data['Song_Name']}
            - Album: {song_data['Album']}
            - Era: {song_data.get('era', 'Unknown')}
            - Audio: Danceability={song_data.get('danceability', 0):.2f}, Energy={song_data.get('energy', 0):.2f}, Valence={song_data.get('valence', 0):.2f}
            - Topic: {song_data.get('dominant_topic', 'Unknown')} (strength: {song_data.get('topic_strength', 0):.2f})
            
            TOP {n_recommendations} SIMILAR SONGS (by hybrid similarity):
            
            """
        for idx, (_, rec) in enumerate(recommendations.iterrows(), 1):
            sim_score = hybrid_sim[song_idx, similar_indices[idx-1]]
            context += f"{idx}. {rec['Song_Name']} ({rec['Album']}) - Similarity: {sim_score:.3f}\n"
            context += f"   Audio: Dance={rec.get('danceability', 0):.2f}, Energy={rec.get('energy', 0):.2f}, Valence={rec.get('valence', 0):.2f}\n"
            context += f"   Topic: {rec.get('dominant_topic', 'Unknown')}\n\n"
            context += """
            Provide:
            1. Why these songs are similar (2-3 sentences)
            2. 2-3 key patterns connecting them (musical or lyrical)
            3. Which recommendation to try first and why (1-2 sentences)
            
            Keep it concise and insightful.
            """
        
        print("  Generating explanation...")

        if config.USE_OPENAI == False:
            reasoning = self.client.generate(context, max_tokens=512)
        else:
            reasoning = self.client.chat_interactive(context)
        
        return {
            'song': song_data['Song_Name'],
            'recommendations': recommendations[['Song_Name', 'Album', 'era']].to_dict('records'),
            'reasoning': reasoning,
            'similarity_scores': [hybrid_sim[song_idx, idx] for idx in similar_indices]
        }
    
    def refine_recommendations(self, song_name: str, feedback: str, previous_reasoning: str) -> str:
        """Agent refines recommendations based on user feedback."""
        
        prompt = f"""You previously recommended songs similar to '{song_name}'.
                YOUR PREVIOUS REASONING:
                {previous_reasoning}
                
                USER FEEDBACK:
                {feedback}
                
                Based on this feedback, what should we adjust? Consider:
                - More upbeat or slower songs?
                - Different era focus?
                - Prioritize lyrics over sound (or vice versa)?
                
                Provide your recommendation in 2-3 sentences.
                """
        
        #return self.client.generate(prompt, max_tokens=256)
        if config.USE_OPENAI == False:
            return self.client.generate(prompt, max_tokens=768)
        else:
            return self.client.chat_interactive(prompt)


def interactive_recommendations():
    """Interactive recommendation session."""
    from data_loading import load_and_merge_data
    from similarity_analysis import create_hybrid_similarity_system
    
    print("="*80)
    print("AGENTIC RECOMMENDATION SYSTEM (Ollama)")
    print("="*80)
    print(f"Model: {config.MODEL}\n")
    
    print("Loading data...")
    df = load_and_merge_data(config.DATA_PATH, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    
    print("Creating similarity system (this may take a minute)...")
    sim_results = create_hybrid_similarity_system(df)
    
    print("Initializing recommendation agent...")
    agent = RecommendationAgent(sim_results, sim_results['df'])
    
    print("\n✓ Agent ready!")
    print("\nCommands:")
    print("  - Enter a Taylor Swift song name")
    print("  - After recommendations, provide feedback to refine")
    print("  - 'quit' to exit")
    print()
    
    last_reasoning = None
    last_song = None
    
    while True:
        try:
            song = input("\n🎵 Enter song (or 'quit'): ").strip()
            
            if song.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not song:
                continue
            
            print(f"\n🔍 Finding recommendations for '{song}'...")
            result = agent.recommend_with_reasoning(song, n_recommendations=5)
            
            if 'error' in result:
                print(f"\n✗ {result['error']}")
                print("\nAvailable songs (sample):")
                sample = sim_results['df']['Song_Name'].head(10).tolist()
                for s in sample:
                    print(f"  - {s}")
                continue
            
            print("\n" + "="*80)
            print(f"RECOMMENDATIONS FOR: {result['song']}")
            print("="*80)
            
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n{i}. {rec['Song_Name']}")
                print(f"   Album: {rec['Album']} | Era: {rec['era']}")
                print(f"   Similarity: {result['similarity_scores'][i-1]:.3f}")
            
            print("\n" + "-"*80)
            print("WHY THESE RECOMMENDATIONS:")
            print("-"*80)
            print(result['reasoning'])
            
            last_reasoning = result['reasoning']
            last_song = song
            
            # Get feedback
            feedback = input("\n💬 Feedback (or Enter to continue): ").strip()
            if feedback and last_reasoning:
                print("\n🤖 Agent: Analyzing feedback...")
                refinement = agent.refine_recommendations(last_song, feedback, last_reasoning)
                print(f"\n🤖 Refinement:\n{refinement}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    interactive_recommendations()
