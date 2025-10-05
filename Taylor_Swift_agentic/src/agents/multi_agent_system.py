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
        lyrics_excerpt = song_data.get('lyrics', '')[:500]
        
        prompt = f"""You are a lyrical analyst specializing in songwriting.
            Song: {song_data['Song_Name']}
            Lyrics (excerpt): {lyrics_excerpt}...
            
            Analyze (2-3 sentences per point):
            1. Main themes and emotions
            2. Writing style and notable literary devices
            3. How it compares to typical Taylor Swift lyrics
            
            Be specific and insightful.
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
        #prompt = f"""You are a musical analyst specializing in audio production.
        #    Song: {song_data['Song_Name']}
        #    Audio Features:
        #    - Danceability: {song_data.get('danceability', 'N/A')}
        #    - Energy: {song_data.get('energy', 'N/A')}
        #    - Valence (happiness): {song_data.get('valence', 'N/A')}
        #    - Tempo: {song_data.get('tempo', 'N/A')} BPM
        #    - Acousticness: {song_data.get('acousticness', 'N/A')}
            
        #    Analyze (2-3 sentences per point):
        #    1. Musical mood and energy
        #    2. Production style characteristics
        #    3. How it fits Taylor Swift's sonic palette
        #
        #    Be specific about the audio features.
        #    """
        prompt = f"""{song_data['Song_Name']}
        D:{song_data.get('danceability', '?')} 
        E:{song_data.get('energy', '?')} 
        V:{song_data.get('valence', '?')} 
        T:{song_data.get('tempo', '?')} 
        A:{song_data.get('acousticness', '?')}

        Analyze: mood/energy, production style, fit in Taylor's discography. Max 50 words."""
        
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
