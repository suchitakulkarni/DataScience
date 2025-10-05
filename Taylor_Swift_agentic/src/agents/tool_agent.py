"""Agent that can use tools to answer questions - Ollama version."""
import json
import re
from typing import Dict, List, Any, Callable
from agents.ollama_client import OllamaClient
from src import config


class ToolAgent:
    """Agent that can use tools/functions to answer queries."""
    
    def __init__(self, model: str = config.OLLAMA_REASONING_MODEL):
        self.client = OllamaClient(model=model)
        self.tools = {}
        self.df = None
        self.similarity_results = None
    
    def register_tool(self, name: str, func: Callable, description: str):
        """Register a tool that the agent can use."""
        self.tools[name] = {
            'function': func,
            'description': description
        }
    
    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        if not self.tools:
            return "No tools available."
        
        desc = "Available tools:\n\n"
        for name, tool in self.tools.items():
            desc += f"- {name}: {tool['description']}\n"
        return desc
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        # Look for tool calls in format: TOOL[tool_name](arg1, arg2, ...)
        pattern = r'TOOL\[(\w+)\]\((.*?)\)'
        matches = re.findall(pattern, response)
        
        tool_calls = []
        for tool_name, args_str in matches:
            # Parse arguments (simple comma-separated for now)
            args = [arg.strip().strip('"\'') for arg in args_str.split(',') if arg.strip()]
            tool_calls.append({
                'tool': tool_name,
                'args': args
            })
        
        return tool_calls
    
    def execute_tool(self, tool_name: str, args: List[Any]) -> Any:
        """Execute a tool with given arguments."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            func = self.tools[tool_name]['function']
            result = func(*args)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def ask(self, question: str, max_iterations: int = 3) -> str:
        """
        Answer question using tools if needed.
        
        Args:
            question: User's question
            max_iterations: Maximum tool-use iterations
            
        Returns:
            Final answer
        """
        system_prompt = f"""You are a helpful assistant that can use tools to answer questions.
            {self._get_tools_description()}
            
            When you need to use a tool, respond with:
            TOOL[tool_name](arg1, arg2, ...)
            
            For example:
            TOOL[get_song_info](Blank Space)
            TOOL[get_era_stats](Pop Era)
            
            After using tools, provide your final answer starting with "ANSWER: "
            
            Be concise and specific in your responses.
            """
        
        conversation = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]
        
        for iteration in range(max_iterations):
            # Get response from model
            response = self.client.chat(conversation, max_tokens=768)
            
            # Check if response contains tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls - this is the final answer
                if 'ANSWER:' in response:
                    return response.split('ANSWER:')[1].strip()
                return response
            
            # Execute tool calls
            tool_results = []
            for call in tool_calls:
                print(f"  🔧 Using tool: {call['tool']}({', '.join(call['args'])})")
                result = self.execute_tool(call['tool'], call['args'])
                tool_results.append(f"Tool {call['tool']} result: {result}")
            
            # Add tool results to conversation
            conversation.append({'role': 'assistant', 'content': response})
            conversation.append({
                'role': 'user', 
                'content': '\n'.join(tool_results) + '\n\nNow provide your final answer.'
            })
        
        return "Could not answer question within iteration limit."
    
    def load_analysis_data(self, df, similarity_results):
        """Load analysis data for tools to use."""
        self.df = df
        self.similarity_results = similarity_results


# Define tool functions
def tool_get_song_info(agent, song_name: str) -> str:
    """Get information about a specific song."""
    if agent.df is None:
        return "Data not loaded"
    
    matches = agent.df[agent.df['Song_Name'].str.lower() == song_name.lower()]
    if matches.empty:
        return f"Song '{song_name}' not found"
    
    song = matches.iloc[0]
    info = f"Song: {song['Song_Name']}\n"
    info += f"Album: {song['Album']}\n"
    info += f"Era: {song.get('era', 'Unknown')}\n"
    info += f"Danceability: {song.get('danceability', 'N/A'):.2f}\n"
    info += f"Energy: {song.get('energy', 'N/A'):.2f}\n"
    info += f"Valence: {song.get('valence', 'N/A'):.2f}\n"
    info += f"Topic: {song.get('dominant_topic', 'Unknown')}"
    
    return info


def tool_get_era_stats(agent, era_name: str) -> str:
    """Get statistics for a specific era."""
    if agent.df is None:
        return "Data not loaded"
    
    era_df = agent.df[agent.df['era'].str.lower() == era_name.lower()]
    if era_df.empty:
        return f"Era '{era_name}' not found"
    
    stats = f"Era: {era_name}\n"
    stats += f"Number of songs: {len(era_df)}\n"
    stats += f"Avg Energy: {era_df['energy'].mean():.2f}\n"
    stats += f"Avg Valence: {era_df['valence'].mean():.2f}\n"
    stats += f"Avg Danceability: {era_df['danceability'].mean():.2f}"
    
    return stats


def tool_find_similar_songs(agent, song_name: str, n: int = 3) -> str:
    """Find similar songs to the given song."""
    if agent.df is None or agent.similarity_results is None:
        return "Data not loaded"
    
    matches = agent.df[agent.df['Song_Name'].str.lower() == song_name.lower()]
    if matches.empty:
        return f"Song '{song_name}' not found"
    
    song_idx = matches.index[0]
    sim_matrix = agent.similarity_results['hybrid_similarity']
    similar_indices = np.argsort(sim_matrix[song_idx])[-n-1:-1][::-1]
    
    similar_songs = agent.df.iloc[similar_indices]
    
    result = f"Similar songs to '{song_name}':\n"
    for i, (_, song) in enumerate(similar_songs.iterrows(), 1):
        result += f"{i}. {song['Song_Name']} ({song['Album']})\n"
    
    return result


def tool_compare_eras(agent, era1: str, era2: str) -> str:
    """Compare two eras."""
    if agent.df is None:
        return "Data not loaded"
    
    era1_df = agent.df[agent.df['era'].str.lower() == era1.lower()]
    era2_df = agent.df[agent.df['era'].str.lower() == era2.lower()]
    
    if era1_df.empty or era2_df.empty:
        return "One or both eras not found"
    
    comparison = f"Comparing {era1} vs {era2}:\n\n"
    comparison += f"{era1}: {len(era1_df)} songs, Avg Energy: {era1_df['energy'].mean():.2f}\n"
    comparison += f"{era2}: {len(era2_df)} songs, Avg Energy: {era2_df['energy'].mean():.2f}\n"
    
    energy_diff = era1_df['energy'].mean() - era2_df['energy'].mean()
    if abs(energy_diff) > 0.1:
        higher = era1 if energy_diff > 0 else era2
        comparison += f"\n{higher} is noticeably more energetic."
    
    return comparison


def interactive_tool_agent():
    """Interactive session with tool-using agent."""
    import numpy as np
    from data_loading import load_and_merge_data
    from similarity_analysis import create_hybrid_similarity_system
    
    print("="*80)
    print("TOOL-USING AGENT (Ollama)")
    print("="*80)
    print(f"Model: {config.OLLAMA_REASONING_MODEL}\n")
    
    print("Loading data...")
    df = load_and_merge_data(config.DATA_PATH, config.SPOTIFY_CSV, config.ALBUM_SONG_CSV)
    
    print("Creating similarity system...")
    sim_results = create_hybrid_similarity_system(df)
    
    print("Initializing tool agent...")
    agent = ToolAgent()
    agent.load_analysis_data(sim_results['df'], sim_results)
    
    # Register tools
    agent.register_tool(
        'get_song_info',
        lambda song: tool_get_song_info(agent, song),
        'Get detailed information about a specific song'
    )
    agent.register_tool(
        'get_era_stats',
        lambda era: tool_get_era_stats(agent, era),
        'Get statistics for a specific era'
    )
    agent.register_tool(
        'find_similar_songs',
        lambda song, n=3: tool_find_similar_songs(agent, song, int(n)),
        'Find similar songs to a given song'
    )
    agent.register_tool(
        'compare_eras',
        lambda era1, era2: tool_compare_eras(agent, era1, era2),
        'Compare statistics between two eras'
    )
    
    print("\n✓ Agent ready with tools!")
    print("\nThe agent can use these tools to answer your questions:")
    print("  - get_song_info: Get details about a song")
    print("  - get_era_stats: Get statistics for an era")
    print("  - find_similar_songs: Find similar songs")
    print("  - compare_eras: Compare two eras")
    print("\nExample questions:")
    print("  'Tell me about Blank Space'")
    print("  'What are the stats for the Pop Era?'")
    print("  'Find songs similar to Style'")
    print("  'Compare Country Era and Pop Era'")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            question = input("\n🎵 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\n🤖 Agent: Thinking...")
            answer = agent.ask(question)
            print(f"\n🤖 Agent:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    interactive_tool_agent()
