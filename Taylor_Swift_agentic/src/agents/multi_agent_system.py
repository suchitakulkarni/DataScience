"""Multi-agent system for comprehensive song analysis - Ollama version."""
from typing import Dict
from agents.ollama_client import OllamaClient
from .openai_client import OpenAIClient
from src import config

class AnalystAgent:
    """Base class for specialist agents."""
    
    def __init__(self, role: str, model: str = config.MODEL):
        self.role = role
        if config.USE_OPENAI == False:
            self.client = OllamaClient(model=model)
        else:
            self.client = OpenAIClient()
    
    def analyze(self, context: str) -> str:
        """Analyze with role-specific prompt."""
        raise NotImplementedError


class LyricalAnalystAgent(AnalystAgent):
    """Specializes in lyrical content analysis."""
    
    def __init__(self, model: str = config.MODEL):
        super().__init__("Lyrical Analyst", model)

    def analyze(self, song_data: Dict) -> str:
        # Analyze based on computed features, NOT raw lyrics
        prompt = f"""You are a lyrical analyst specializing in songwriting patterns.

        Song: {song_data['Song_Name']}
        Album: {song_data['Album']}
        Era: {song_data.get('era', 'Unknown')}
    
        LYRICAL METRICS:
        - Average word length: {song_data.get('avg_word_length', 0):.2f}
        - Vocabulary diversity (unique word ratio): {song_data.get('unique_word_ratio', 0):.2f}
        - Total words: {song_data.get('total_words', 0)}
        - Sentiment polarity: {song_data.get('polarity', 0):.2f} (-1=negative, +1=positive)
        - Subjectivity: {song_data.get('subjectivity', 0):.2f} (0=objective, 1=subjective)
        - Dominant lyrical theme: {song_data.get('dominant_topic', 'Unknown')}
    
        Analyze (2-3 sentences per point):
        1. What these metrics suggest about the song's emotional tone and writing style
        2. How the vocabulary complexity and sentiment patterns compare to typical Taylor Swift songs
        3. What the thematic classification reveals about the song's subject matter
    
        Base your analysis strictly on these quantitative metrics.
        """
        
        if config.USE_OPENAI == False:
            return self.client.generate(prompt, max_tokens=400)
        else:
            return self.client.chat_interactive(prompt)


class MusicalAnalystAgent(AnalystAgent):
    """Specializes in musical/audio feature analysis."""
    
    def __init__(self, model: str = config.MODEL):
        super().__init__("Musical Analyst", model)


    def analyze(self, song_data: Dict) -> str:
        prompt = f"""You are a musical analyst specializing in audio production.

        Song: {song_data['Song_Name']}
        Album: {song_data['Album']}
    
        AUDIO FEATURES:
        - Danceability: {song_data.get('danceability', 'N/A'):.2f} (0=not danceable, 1=very danceable)
        - Energy: {song_data.get('energy', 'N/A'):.2f} (0=calm, 1=energetic)
        - Valence: {song_data.get('valence', 'N/A'):.2f} (0=sad, 1=happy)
        - Tempo: {song_data.get('tempo', 'N/A'):.0f} BPM
        - Acousticness: {song_data.get('acousticness', 'N/A'):.2f} (0=electronic, 1=acoustic)
        - Loudness: {song_data.get('loudness', 'N/A'):.1f} dB
    
        Analyze: musical mood/energy, production style, how it fits Taylor's sonic evolution. Max 4 sentences.
        """
        
        #return self.client.generate(prompt, max_tokens=400)
        if config.USE_OPENAI == False:
            return self.client.generate(prompt, max_tokens=400)
        else:
            return self.client.chat_interactive(prompt)


class ContextualAnalystAgent(AnalystAgent):
    """Specializes in era and contextual analysis."""
    
    def __init__(self, model: str = config.MODEL):
        super().__init__("Contextual Analyst", model)
    
    def analyze(self, song_data: Dict) -> str:
        prompt = f"""You are a music historian analyzing Taylor Swift's artistic evolution.
            S: {song_data['Song_Name']}
            A: {song_data['Album']}
            E: {song_data.get('era', 'Unknown')}
            
            Analyze: fit in artistic evolution, era-specificity, discography context. 
            """
        
        #return self.client.generate(prompt, max_tokens=400)
        if config.USE_OPENAI == False:
            return self.client.generate(prompt, max_tokens=400)
        else:
            return self.client.chat_interactive(prompt)


class OrchestratorAgent:
    """Orchestrates multiple analyst agents and synthesizes insights."""
    
    def __init__(self, model: str = config.REASONING_MODEL):
        if config.USE_OPENAI == False:
            self.client = OllamaClient(model=model)
        else:
            self.client = OpenAIClient()
        self.lyrical_agent = LyricalAnalystAgent()
        self.musical_agent = MusicalAnalystAgent()
        self.contextual_agent = ContextualAnalystAgent()
    
    def analyze_song(self, song_data: Dict) -> Dict[str, str]:
        """Coordinate analysis from multiple agents."""
        
        print(f"\n📊 Analyzing '{song_data['Song_Name']}'...")
        
        # Each agent analyzes independently
        print("  🎤 Lyrical analyst working...")
        lyrical = self.lyrical_agent.analyze(song_data)
        
        print("  🎵 Musical analyst working...")
        musical = self.musical_agent.analyze(song_data)
        
        print("  📚 Contextual analyst working...")
        contextual = self.contextual_agent.analyze(song_data)
        
        # Orchestrator synthesizes
        print("  🧠 Synthesizing insights...")
        synthesis_prompt = f"""You are the lead analyst synthesizing 
            insights from three specialists about "{song_data['Song_Name']}":
            LYRICAL ANALYST:
            {lyrical}
            
            MUSICAL ANALYST:
            {musical}
            
            CONTEXTUAL ANALYST:
            {contextual}
            
            Provide:
            1. Cohesive overall interpretation (2-3 sentences)
            2. Most interesting insight from combining all three perspectives (1-2 sentences)
            3. One unexpected connection or pattern (1-2 sentences)
            
            Be concise and insightful.
            """
        

        if config.USE_OPENAI == False:
            synthesis = self.client.generate(synthesis_prompt, max_tokens=512)
        else:
            synthesis = self.client.chat_interactive(synthesis_prompt)
        
        return {
            'lyrical': lyrical,
            'musical': musical,
            'contextual': contextual,
            'synthesis': synthesis
        }
    
    def compare_songs(self, song1_data: Dict, song2_data: Dict) -> str:
        """Agent-driven comparison of two songs."""
        
        prompt = f"""Compare these two Taylor Swift songs:
            SONG 1: {song1_data['Song_Name']} ({song1_data['Album']})
            - Era: {song1_data.get('era', 'Unknown')}
            - Energy: {song1_data.get('energy', 'N/A')}, Valence: {song1_data.get('valence', 'N/A')}
            - Topic: {song1_data.get('dominant_topic', 'Unknown')}
            
            SONG 2: {song2_data['Song_Name']} ({song2_data['Album']})
            - Era: {song2_data.get('era', 'Unknown')}
            - Energy: {song2_data.get('energy', 'N/A')}, Valence: {song2_data.get('valence', 'N/A')}
            - Topic: {song2_data.get('dominant_topic', 'Unknown')}
            
            Provide (2-3 sentences each):
            1. Key similarities and differences
            2. How they represent different phases of Taylor's artistry
            3. What fans of one might appreciate about the other
            """
        
        #return self.client.generate(prompt, max_tokens=768)

        if config.USE_OPENAI == False:
            return self.client.generate(prompt, max_tokens=768)
        else:
            return self.client.chat_interactive(prompt)


def interactive_multi_agent():
    """Interactive multi-agent analysis session."""
    from data_loading import load_and_merge_data
    
    print("="*80)
    print("MULTI-AGENT SONG ANALYSIS SYSTEM (Ollama)")
    print("="*80)
    print(f"Models: {config.MODEL} (specialists), {config.REASONING_MODEL} (orchestrator)")
    print()
    
    print("Loading data...")
    df = load_and_merge_data(config.DATA_PATH, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    
    print("Initializing multi-agent system...")
    orchestrator = OrchestratorAgent()
    
    print("\n✓ All agents ready!")
    print("\nCommands:")
    print("  analyze <song>        - Get multi-agent analysis")
    print("  compare <song1> vs <song2> - Compare two songs")
    print("  quit                  - Exit")
    print()
    
    while True:
        try:
            command = input("\n🎵 Command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if command.lower().startswith('analyze '):
                song_name = command[8:].strip()
                matches = df[df['Song_Name'].str.lower() == song_name.lower()]
                
                if matches.empty:
                    print(f"✗ Song '{song_name}' not found.")
                    continue
                
                song_data = matches.iloc[0].to_dict()
                result = orchestrator.analyze_song(song_data)
                
                print("\n" + "="*80)
                print("MULTI-AGENT ANALYSIS")
                print("="*80)
                
                print("\n[LYRICAL ANALYST]")
                print(result['lyrical'])
                
                print("\n[MUSICAL ANALYST]")
                print(result['musical'])
                
                print("\n[CONTEXTUAL ANALYST]")
                print(result['contextual'])
                
                print("\n[SYNTHESIS]")
                print(result['synthesis'])
            
            elif ' vs ' in command.lower() and command.lower().startswith('compare '):
                parts = command[8:].split(' vs ')
                if len(parts) != 2:
                    print("Usage: compare <song1> vs <song2>")
                    continue
                
                song1_name, song2_name = [s.strip() for s in parts]
                
                matches1 = df[df['Song_Name'].str.lower() == song1_name.lower()]
                matches2 = df[df['Song_Name'].str.lower() == song2_name.lower()]
                
                if matches1.empty or matches2.empty:
                    print("✗ One or both songs not found.")
                    continue
                
                print("\n🔄 Comparing songs...")
                comparison = orchestrator.compare_songs(
                    matches1.iloc[0].to_dict(),
                    matches2.iloc[0].to_dict()
                )
                
                print("\n" + "="*80)
                print("SONG COMPARISON")
                print("="*80)
                print(comparison)
            
            else:
                print("Unknown command. Try:")
                print("  analyze Blank Space")
                print("  compare Style vs Wildest Dreams")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    interactive_multi_agent()
