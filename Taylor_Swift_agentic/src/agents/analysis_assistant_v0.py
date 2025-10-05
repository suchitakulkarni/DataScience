"""Conversational agent for exploring Taylor Swift analysis results - Ollama version."""
import pandas as pd
from .ollama_client import OllamaClient  # Relative import within agents/
#from .. import config  # Up one level to src, then config
from src import config

class AnalysisAssistant:
    """Agent that answers questions about analysis results using Ollama."""
    
    def __init__(self, model: str = config.OLLAMA_MODEL):
        self.client = OllamaClient(model=model)
        self.context = self._load_analysis_context()
        self.system_prompt = self._build_system_prompt()
    
    def _load_analysis_context(self):
        """Load all analysis results into context."""
        context = {
            'era_stats': None,
            'topic_summary': None,
            'reputation_diff': None,
            'songs_with_topics': None
        }
        
        try:
            context['topic_summary'] = pd.read_csv(f'{config.RESULTS_DIR}/topic_era_summary.csv')
            context['reputation_diff'] = pd.read_csv(f'{config.RESULTS_DIR}/reputation_feature_differences.csv')
            context['songs_with_topics'] = pd.read_csv(f'{config.RESULTS_DIR}/songs_with_topics.csv')
            print("✓ Loaded analysis results")
        except FileNotFoundError as e:
            print(f"⚠ Warning: Could not load some results: {e}")
        
        return context
    
    def _build_system_prompt(self):
        """Build system prompt with analysis context."""
        prompt = """You are an expert music analyst who has just completed \
        a comprehensive analysis of Taylor Swift's discography. \
        You have access to the following analysis results:"""
        
        if self.context['topic_summary'] is not None:
            prompt += "TOPIC-ERA SUMMARY:\n"
            prompt += self.context['topic_summary'].to_string() + "\n\n"
        
        if self.context['reputation_diff'] is not None:
            prompt += "REPUTATION DISTINCTIVE FEATURES (top 5):\n"
            prompt += self.context['reputation_diff'].head(5).to_string() + "\n\n"
        
        if self.context['songs_with_topics'] is not None:
            prompt += "SONGS WITH TOPICS (sample of 10):\n"
            prompt += self.context['songs_with_topics'].head(10).to_string() + "\n\n"
        
        prompt += """Your role is to:
            1. Answer questions about Taylor Swift's music based on this data
            2. Provide insights about patterns in her discography
            3. Explain technical analysis results in accessible language
            4. Be specific and reference the data when possible
            
            Keep responses concise but informative (2-4 paragraphs max).
            """
        return prompt
    
    def ask(self, question: str) -> str:
        """Ask the agent a question about the analysis."""
        return self.client.chat_interactive(
            user_message=question,
            system_prompt=self.system_prompt
        )
    
    def suggest_insights(self) -> str:
        """Agent proactively suggests interesting insights."""
        prompt = """Based on the Taylor Swift analysis results you have access to, 
                suggest 3-5 interesting insights or patterns that a music analyst should investigate further. 
                Be specific and reference the actual data."""
        
        return self.ask(prompt)
    
    def reset(self):
        """Reset conversation history."""
        self.client.reset_conversation()


def interactive_session():
    """Start an interactive chat session with the agent."""
    
    print("="*80)
    print("TAYLOR SWIFT ANALYSIS ASSISTANT (Ollama)")
    print("="*80)
    print(f"Model: {config.OLLAMA_MODEL}")
    print("\nInitializing agent...")
    
    try:
        agent = AnalysisAssistant()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  Terminal 1: ollama serve")
        print("  Terminal 2: ollama pull llama3.1:8b")
        return
    
    print("✓ Agent ready!")
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'insights' for AI-suggested insights")
    print("  - 'reset' to clear conversation history")
    print("  - 'quit' to exit")
    print()
    
    while True:
        try:
            question = input("\n🎵 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'reset':
                agent.reset()
                print("\n✓ Conversation history cleared")
                continue
            
            if question.lower() == 'insights':
                print("\n🤖 Agent: Generating insights...\n")
                response = agent.suggest_insights()
            elif question:
                response = agent.ask(question)
            else:
                continue
            
            print(f"\n🤖 Assistant:\n{response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    interactive_session()
