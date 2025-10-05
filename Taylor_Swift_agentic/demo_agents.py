"""Demo script showcasing all agentic features."""
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src import config
from src.agents.ollama_client import test_ollama_connection
#from src.agents.openai_client import test_ollama_connection

import os

def main_menu():
    """Display main menu and handle selection."""
    
    print("\n" + "="*80)
    print("TAYLOR SWIFT AGENTIC ANALYSIS DEMO (Ollama)")
    print("="*80)
    print(f"\nUsing model: {config.MODEL}")
    
    # Test connection
    print("\nTesting Ollama connection...")
    if not test_ollama_connection():
        return
    
    while True:
        print("\n" + "-"*80)
        print("SELECT AGENT:")
        print("-"*80)
        print("1. Conversational Analysis Assistant")
        print("   - Ask questions about analysis results")
        print("   - Get AI-suggested insights")
        print()
        print("2. Agentic Recommendation System")
        print("   - Get song recommendations with explanations")
        print("   - Refine recommendations based on feedback")
        print()
        print("3. Multi-Agent Song Analysis")
        print("   - Lyrical, musical, and contextual analysis")
        print("   - Compare songs across dimensions")
        print()
        print("4. Tool-Using Agent")
        print("   - Agent uses tools to answer questions")
        print("   - Get song info, era stats, similarities")
        print()
        print("5. Memory-Enhanced Agent")
        print("   - Remembers previous conversations")
        print("   - Provides context-aware responses")
        print()
        print("0. Exit")
        print("-"*80)
        
        choice = input("\nSelect (0-5): ").strip()
        
        if choice == '0':
            print("\nGoodbye!")
            break
        elif choice == '1':
            from src.agents.analysis_assistant import interactive_session
            interactive_session()
        elif choice == '2':
            from src.agents.recommendation_agent import interactive_recommendations
            interactive_recommendations()
        elif choice == '3':
            from src.agents.multi_agent_system import interactive_multi_agent
            interactive_multi_agent()
        elif choice == '4':
            from src.agents.tool_agent import interactive_tool_agent
            interactive_tool_agent()
        elif choice == '5':
            from src.agents.memory_agent import interactive_memory_agent
            interactive_memory_agent()
        else:
            print("\n✗ Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
