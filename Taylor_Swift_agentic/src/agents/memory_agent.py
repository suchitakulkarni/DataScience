"""Agent with persistent conversation memory - Ollama version."""
import json
import os
from datetime import datetime
from typing import List, Dict
from agents.ollama_client import OllamaClient
from src import config


class MemoryAgent:
    """Agent with persistent conversation memory."""
    
    def __init__(self,  model: str = config.MODEL, memory_file: str = f'{config.RESULTS_DIR}/agent_memory.json'):
        self.client = OllamaClient(model=model)
        self.memory_file = memory_file
        self.short_term_memory = []  # Current session
        self.long_term_memory = []   # Persistent across sessions
        self._load_memory()
    
    def _load_memory(self):
        """Load long-term memory from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.long_term_memory = json.load(f)
                print(f"✓ Loaded {len(self.long_term_memory)} previous conversations")
            except Exception as e:
                print(f"⚠ Could not load memory: {e}")
                self.long_term_memory = []
    
    def _save_memory(self):
        """Save long-term memory to disk."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.long_term_memory, f, indent=2)
        except Exception as e:
            print(f"⚠ Could not save memory: {e}")
    
    def _get_relevant_memories(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve k most relevant past conversations."""
        if not self.long_term_memory:
            return []
        
        # Simple keyword matching (could use embeddings for better results)
        query_words = set(query.lower().split())
        
        scored_memories = []
        for memory in self.long_term_memory:
            question = memory.get('question', '').lower()
            answer = memory.get('answer', '').lower()
            
            # Score based on word overlap
            memory_words = set(question.split() + answer.split())
            overlap = len(query_words & memory_words)
            
            if overlap > 0:
                scored_memories.append((overlap, memory))
        
        # Return top k
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [m[1] for m in scored_memories[:k]]
    
    def ask(self, question: str, use_memory: bool = True) -> str:
        """
        Ask question with memory context.
        
        Args:
            question: User's question
            use_memory: Whether to use long-term memory
            
        Returns:
            Agent's answer
        """
        # Build context from memories
        context = ""
        if use_memory:
            relevant_memories = self._get_relevant_memories(question)
            if relevant_memories:
                context = "Previous relevant conversations:\n\n"
                for mem in relevant_memories:
                    context += f"Q: {mem['question']}\nA: {mem['answer'][:200]}...\n\n"
                context += "Current question:\n"
        
        # Get answer
        full_prompt = context + question
        answer = self.client.chat_interactive(full_prompt)
        
        # Store in short-term memory
        self.short_term_memory.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        return answer
    
    def save_session(self):
        """Save current session to long-term memory."""
        if self.short_term_memory:
            self.long_term_memory.extend(self.short_term_memory)
            self._save_memory()
            print(f"\n✓ Saved {len(self.short_term_memory)} interactions to memory")
            self.short_term_memory = []
    
    def get_session_summary(self) -> str:
        """Get summary of current session."""
        if not self.short_term_memory:
            return "No interactions in current session."
        
        summary_prompt = "Summarize the key topics discussed in this conversation:\n\n"
        for mem in self.short_term_memory:
            summary_prompt += f"Q: {mem['question']}\n"
        
        summary_prompt += "\nProvide a brief summary (2-3 sentences)."
        
        #return self.client.generate(summary_prompt, max_tokens=256)
        if config.USE_OPENAI == False:
            return self.client.generate(prompt, max_tokens=256)
        else:
            return self.client.chat_interactive(prompt)
    
    def clear_memory(self, clear_long_term: bool = False):
        """Clear memory."""
        self.short_term_memory = []
        if clear_long_term:
            self.long_term_memory = []
            self._save_memory()
            print("✓ All memory cleared")
        else:
            print("✓ Session memory cleared")


def interactive_memory_agent():
    """Interactive session with memory-enhanced agent."""
    
    print("="*80)
    print("MEMORY-ENHANCED AGENT (Ollama)")
    print("="*80)
    print(f"Model: {config.MODEL}\n")
    
    agent = MemoryAgent()
    
    print("✓ Agent ready!")
    print("\nCommands:")
    print("  - Ask questions (agent remembers context)")
    print("  - 'summary' to get session summary")
    print("  - 'save' to save session to long-term memory")
    print("  - 'clear' to clear session memory")
    print("  - 'quit' to exit (auto-saves)")
    print()
    
    try:
        while True:
            question = input("\n🎵 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                agent.save_session()
                print("\nGoodbye!")
                break
            
            if question.lower() == 'summary':
                print("\n📊 Session Summary:")
                print(agent.get_session_summary())
                continue
            
            if question.lower() == 'save':
                agent.save_session()
                continue
            
            if question.lower() == 'clear':
                agent.clear_memory()
                continue
            
            if not question:
                continue
            
            answer = agent.ask(question)
            print(f"\n🤖 Agent:\n{answer}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving session...")
        agent.save_session()
        print("Goodbye!")


if __name__ == "__main__":
    interactive_memory_agent()
