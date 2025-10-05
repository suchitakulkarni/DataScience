"""Conversational agent for exploring Taylor Swift analysis results - Ollama version."""
import os
import pandas as pd

from .openai_client import OpenAIClient
from .ollama_client import OllamaClient  # Relative import within agents/
from src import config

# Suppress Ollama logging
os.environ['LLAMA_LOG_LEVEL'] = '0'

class AnalysisAssistant:
    def __init__(self, model: str = config.MODEL):
        if config.USE_OPENAI == False:
            self.client = OllamaClient(model=model)
        else: self.client = OpenAIClient()
        self.context = self._load_analysis_context()
        self.system_prompt = self._build_system_prompt()

    def _load_analysis_context(self):
        """Load all analysis results into context."""
        context = {
            'personal_alignment': None,
            'reputation_diff': None,
            'songs_with_topics': None
        }
        try:
            print(f"will use Ollama model = {config.MODEL}")
            context['personal_alignment'] = pd.read_csv(f'{config.RESULTS_DIR}/personal_alignment.csv')
            context['reputation_diff'] = pd.read_csv(f'{config.RESULTS_DIR}/reputation_feature_differences.csv')
            context['songs_with_topics'] = pd.read_csv(f'{config.RESULTS_DIR}/songs_with_topics.csv')
            print("Loaded analysis results")
        except FileNotFoundError as e:
            print(f"Warning: Could not load some results: {e}")

        return context

    def _build_system_prompt(self):
        """Build system prompt with analysis context - OPTIMIZED."""
        #prompt = """You are an expert in analyzing Taylor Swift's discography.
        #Answer questions concisely using only the provided data.\n\n"""
        prompt = """You are an analytical assistant for Taylor Swift's music data.

        Your task: 
        Answer questions *only* using the tabular data provided below. 
        If the data does not contain the answer, reply exactly with: 
        "I don’t have enough data to answer that."

        You must not use external knowledge, speculation, or general information about Taylor Swift.
        Use concise analytical language, citing specific features or metrics from the data.

        Data provided:\n\n
        """

        # Personal alignment - TOP 5 only
        if self.context['personal_alignment'] is not None:
            prompt += "PERSONAL ALIGNMENT (Top 5 features):\n"
            df = self.context['personal_alignment'][
                ['feature', 'closer_to', 'alignment_score']].head(5)  # Reduced columns
            prompt += df.to_string(index=False) + "\n\n"

        # Reputation distinctive features - TOP 3 only
        if self.context['reputation_diff'] is not None:
            prompt += "REPUTATION DISTINCTIVE FEATURES (Top 3):\n"
            df = self.context['reputation_diff'][['feature', 'difference']].head(3)  # Reduced
            prompt += df.to_string(index=False) + "\n\n"

        if self.context['songs_with_topics'] is not None:
            prompt += "SONG LYRICAL AND ACOUSTIC (Top 3):\n"
            df = self.context['songs_with_topics']
            prompt += df.to_string(index=False) + "\n\n"



        # Remove songs sample entirely - it's not being used effectively
        # The agent doesn't need example songs in the system prompt

        prompt += "\nEra definitions: Country (2006-2008), Fearless (2008-2010), Speak Now (2010-2012), Red (2012-2014), 1989 (2014-2016), Reputation (2017-2018), Lover (2019), Folklore/Evermore (2020-2021), Midnights (2022+)"
        print(f"System prompt size: {len(prompt)} chars")  # CHECK THIS
        return prompt

    def ask(self, question: str) -> str:
        restricted_query = (
            "Remember: Use only the provided data context. "
            "If the data doesn’t support an answer, say 'I don’t have enough data to answer that.'\n\n"
            f"Question: {question}"
        )
        return self.client.chat_interactive(
            user_message=restricted_query,
            system_prompt=self.system_prompt,
        )

    def suggest_insights(self) -> str:
        """Agent proactively suggests interesting insights."""
        prompt = """Based on the given Taylor Swift actual data analysis results, 
        suggest 1-3 insights or patterns that a music analyst should investigate further."""

        return self.ask(prompt)

    def reset(self):
        """Reset conversation history."""
        self.client.reset_conversation()

def interactive_session():
    """Start an interactive chat session with the agent."""

    print("=" * 80)
    print("TAYLOR SWIFT ANALYSIS ASSISTANT (Ollama)")
    print("=" * 80)
    print(f"Model: {config.MODEL}")
    print("\nInitializing agent...")

    try:
        agent = AnalysisAssistant()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Ollama is running:")
        print("  Terminal 1: ollama serve")
        print(f"  Terminal 2: ollama pull {config.MODEL}")
        return

    print("Agent ready!")
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'insights' for AI-suggested insights")
    print("  - 'reset' to clear conversation history")
    print("  - 'quit' to exit")
    print()

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if question.lower() == 'reset':
                agent.reset()
                print("\nConversation history cleared")
                continue

            if question.lower() == 'insights':
                print("\nAgent: Generating insights...\n")
                response = agent.suggest_insights()
            elif question:
                response = agent.ask(question)
            else:
                continue

            print(f"\nAssistant:\n{response}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    interactive_session()