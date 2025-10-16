"""Conversational agent with Chain-of-Thought reasoning and dynamic data retrieval."""
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any

from .openai_client import OpenAIClient
from .ollama_client import OllamaClient
from src import config

os.environ['LLAMA_LOG_LEVEL'] = '0'


class DataDictionary:
    """Defines all available data columns and their meanings."""
    
    COLUMNS = {
        # Song identifiers
        'Song_Name': 'Name of the song',
        'Album': 'Album name the song belongs to',
        'album_clean': 'Cleaned album name',
        'era': 'Musical era (debut, fearless, speak_now, red, 1989, reputation, lover, folklore, evermore, midnights)',
        'Release_Date': 'Date when the song was released',
        
        # Audio features (0-1 scale)
        'danceability': 'How suitable the song is for dancing (0=not danceable, 1=very danceable)',
        'energy': 'Perceptual measure of intensity and activity (0=calm, 1=energetic)',
        'valence': 'Musical positivity/happiness (0=sad/negative, 1=happy/positive)',
        'acousticness': 'Confidence the track is acoustic (0=not acoustic, 1=acoustic)',
        'instrumentalness': 'Predicts whether track has no vocals (0=vocals, 1=instrumental)',
        'liveness': 'Detects presence of audience in recording (0=studio, 1=live)',
        'speechiness': 'Detects presence of spoken words (0=music, 1=speech)',
        
        # Other audio metrics
        'loudness': 'Overall loudness in decibels (typically -60 to 0)',
        'tempo': 'Estimated tempo in beats per minute (BPM)',
        'duration_ms': 'Duration of the song in milliseconds',
        'key': 'Musical key the song is in (0-11, corresponds to pitch classes)',
        'mode': 'Modality of track (0=minor, 1=major)',
        'time_signature': 'Time signature (beats per measure)',
        
        # Sentiment analysis
        'polarity': 'Sentiment polarity from lyrics (-1=negative, 1=positive)',
        'subjectivity': 'Subjectivity of lyrics (0=objective, 1=subjective)',
        
        # Topic modeling
        'dominant_topic': 'Primary topic/theme identified in the song',
        'topic_weight': 'Strength of the dominant topic assignment'
    }
    
    DATASETS = {
        'songs_with_topics': {
            'description': 'Main dataset with all songs and their features',
            'file': 'songs_with_topics.csv',
            'key_columns': ['Song_Name', 'Album', 'era', 'dominant_topic', 
                          'danceability', 'energy', 'valence', 'polarity']
        },
        'personal_alignment': {
            'description': 'Features showing alignment between 1989 and Reputation',
            'file': 'personal_alignment.csv',
            'key_columns': ['feature', 'closer_to', 'alignment_score']
        },
        'reputation_diff': {
            'description': 'Distinctive features of Reputation album',
            'file': 'reputation_feature_differences.csv',
            'key_columns': ['feature', 'difference']
        }
    }
    
    @classmethod
    def get_columns_for_question_type(cls, question_type: str) -> List[str]:
        """Suggest relevant columns based on question type."""
        mappings = {
            'emotion': ['valence', 'polarity', 'energy', 'mode'],
            'evolution': ['era', 'Release_Date', 'Album'],
            'musical': ['tempo', 'key', 'danceability', 'energy', 'acousticness'],
            'theme': ['dominant_topic', 'topic_weight', 'polarity', 'subjectivity'],
            'comparison': ['Album', 'era'],
            'statistics': ['Song_Name', 'Album', 'era']
        }
        return mappings.get(question_type, [])


class AnalysisAssistant:
    def __init__(self, model: str = config.MODEL):
        if not config.USE_OPENAI:
            self.client = OllamaClient(model=model)
        else:
            self.client = OpenAIClient()
        
        self.data_sources = self._load_data_sources()
        self.system_prompt = self._build_system_prompt()
        self.conversation_history = []
        print('Enhanced assistant initialized with CoT and dynamic data retrieval')

    def _load_data_sources(self) -> Dict[str, pd.DataFrame]:
        """Load all data sources into memory (not into prompt)."""
        sources = {}
        for name, info in DataDictionary.DATASETS.items():
            try:
                filepath = f"{config.RESULTS_DIR}/{info['file']}"
                sources[name] = pd.read_csv(filepath)
                print(f"Loaded {name}: {len(sources[name])} rows")
            except FileNotFoundError:
                print(f"Warning: Could not load {name}")
                sources[name] = None
        return sources

    def _build_system_prompt(self) -> str:
        """Build system prompt with data dictionary and reasoning instructions."""
        prompt = """You are an analytical assistant for Taylor Swift's music data with advanced reasoning capabilities.

YOUR CAPABILITIES:
1. Access to comprehensive music datasets (described below)
2. Chain-of-Thought reasoning for complex questions
3. Dynamic data retrieval - request specific columns as needed

AVAILABLE DATASETS:
"""
        # Add dataset descriptions
        for name, info in DataDictionary.DATASETS.items():
            if self.data_sources.get(name) is not None:
                prompt += f"\n{name}:\n"
                prompt += f"  Description: {info['description']}\n"
                prompt += f"  Rows: {len(self.data_sources[name])}\n"
                prompt += f"  Key columns: {', '.join(info['key_columns'])}\n"

        prompt += "\n\nAVAILABLE COLUMNS AND THEIR MEANINGS:\n"
        for col, meaning in DataDictionary.COLUMNS.items():
            prompt += f"  {col}: {meaning}\n"

        prompt += """

REASONING PROTOCOL (Chain-of-Thought):
For complex questions, follow this structure:

1. UNDERSTAND: Rephrase the question to confirm understanding
2. PLAN: Break down into steps:
   - What data/columns are needed?
   - What calculations or comparisons are required?
   - What insights should be derived?
3. REQUEST DATA: Specify exactly what data you need. The system uses pandas for efficient filtering and aggregation.
   
   DATA_REQUEST: {
     "dataset": "dataset_name",
     "columns": ["col1", "col2"],
     "filters": {"column": "value"},
     "aggregation": "describe aggregation needed"
   }
   
   FILTERING:
   - String filters: case-insensitive matching, falls back to partial match if exact fails
   - Numeric filters: exact value matching
   - Multiple filters: applied sequentially (AND logic)
   
   AGGREGATIONS:
   - "mean/average/avg grouped by era" → df.groupby('era').mean()
   - "sum/total grouped by album" → df.groupby('album').sum()
   - "count grouped by topic" → df.groupby('topic').size()
   - "max/min grouped by year" → df.groupby('year').max/min()
   - "summary/statistics/describe" → df.describe()
   
   IMPORTANT: 
   - Keep JSON on single line or properly formatted
   - Use double quotes for all strings
   - Filters are applied using pandas boolean indexing
4. ANALYZE: Once data is provided, analyze it step by step
5. ANSWER: Provide clear, evidence-based response

RESPONSE FORMAT:
- For simple questions: Answer directly with available context
- For complex questions: Show your reasoning steps
- Always cite specific metrics or data points
- If you need data not yet provided, output a DATA_REQUEST

EXAMPLE:
Question: "How has Taylor's music evolved from early to recent eras?"

UNDERSTAND: Comparing musical characteristics across time periods (early vs recent eras)

PLAN:
1. Define early eras (debut, fearless) and recent eras (folklore, evermore, midnights)
2. Compare key musical features: energy, valence, acousticness, tempo
3. Identify trends and differences

DATA_REQUEST: {
  "dataset": "songs_with_topics",
  "columns": ["era", "energy", "valence", "acousticness", "tempo"],
  "aggregation": "mean values grouped by era"
}

[After receiving data]
ANALYZE: 
- Early eras show higher energy (0.65 avg) vs recent (0.48 avg)
- Valence decreased from 0.55 to 0.42 (more melancholic)
- Acousticness increased dramatically (0.12 to 0.58)
- Tempo slightly decreased (125 to 118 BPM)

ANSWER: Taylor's music has evolved toward a more introspective, acoustic sound...

IMPORTANT RULES:
- Never invent data or statistics
- Base all claims on provided data
- If data is insufficient, explicitly state: "I need additional data to answer this"
- Never reference or quote song lyrics
- Use concise analytical language
"""
        return prompt

    def _extract_data_request(self, response: str) -> Optional[Dict]:
        """Extract DATA_REQUEST from agent response."""
        if "DATA_REQUEST:" not in response:
            return None
        
        try:
            # Find JSON block after DATA_REQUEST:
            start = response.find("DATA_REQUEST:") + len("DATA_REQUEST:")
            
            # Find the JSON object - need to handle nested braces
            json_start = response.find("{", start)
            if json_start == -1:
                return None
            
            # Count braces to find matching closing brace
            brace_count = 0
            json_end = json_start
            for i, char in enumerate(response[json_start:], json_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            json_str = response[json_start:json_end]
            
            # Clean up common issues
            json_str = json_str.replace('\n', ' ')  # Remove newlines
            json_str = json_str.replace('\\', '')   # Remove escapes
            
            # Try to parse
            parsed = json.loads(json_str)
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Could not parse DATA_REQUEST: {e}")
            print(f"Attempted to parse: {json_str[:200]}")
            
            # Fallback: try to extract manually
            try:
                return self._manual_parse_data_request(response)
            except Exception as e2:
                print(f"Manual parsing also failed: {e2}")
                return None
    
    def _manual_parse_data_request(self, response: str) -> Optional[Dict]:
        """Fallback manual parsing for DATA_REQUEST."""
        import re
        
        # Extract dataset
        dataset_match = re.search(r'"dataset"\s*:\s*"([^"]+)"', response)
        dataset = dataset_match.group(1) if dataset_match else "songs_with_topics"
        
        # Extract columns
        columns_match = re.search(r'"columns"\s*:\s*\[([^\]]+)\]', response)
        columns = []
        if columns_match:
            columns_str = columns_match.group(1)
            columns = [c.strip().strip('"\'') for c in columns_str.split(',')]
        
        # Extract filters
        filters = {}
        filters_match = re.search(r'"filters"\s*:\s*\{([^\}]+)\}', response)
        if filters_match:
            filters_str = filters_match.group(1)
            # Simple key-value extraction
            for pair in filters_str.split(','):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    filters[key] = value
        
        # Extract aggregation
        agg_match = re.search(r'"aggregation"\s*:\s*"([^"]+)"', response)
        aggregation = agg_match.group(1) if agg_match else ""
        
        result = {
            "dataset": dataset,
            "columns": columns,
            "filters": filters,
            "aggregation": aggregation
        }
        
        print(f"[MANUAL PARSE] Extracted: {result}")
        return result

    def _fulfill_data_request(self, request: Dict) -> str:
        """Retrieve requested data and format for agent using pandas operations."""
        dataset_name = request.get('dataset')
        columns = request.get('columns', [])
        filters = request.get('filters', {})
        aggregation = request.get('aggregation', '')
        
        if dataset_name not in self.data_sources or self.data_sources[dataset_name] is None:
            return f"ERROR: Dataset '{dataset_name}' not available"
        
        df = self.data_sources[dataset_name].copy()
        
        # Apply filters using pandas query or boolean indexing
        try:
            for col, value in filters.items():
                if col not in df.columns:
                    print(f"[WARNING] Column '{col}' not found in dataset, skipping filter")
                    continue
                
                # Handle different filter types
                if isinstance(value, str):
                    # String matching - try exact first, then case-insensitive
                    mask = df[col].astype(str).str.lower() == value.lower()
                    if not mask.any():
                        # Try partial match if exact fails
                        mask = df[col].astype(str).str.contains(value, case=False, na=False)
                    df = df[mask]
                else:
                    # Numeric filtering
                    df = df[df[col] == value]
                
                print(f"[FILTER] {col} = '{value}' → {len(df)} rows remaining")
            
            if len(df) == 0:
                return f"ERROR: No data matches filters: {filters}"
        
        except Exception as e:
            return f"ERROR: Filter failed: {str(e)}"
        
        # Select columns
        available_cols = [col for col in columns if col in df.columns]
        if not available_cols:
            return f"ERROR: None of the requested columns {columns} are available"
        
        # Warn about missing columns
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"[WARNING] Columns not found: {missing_cols}")
        
        df = df[available_cols]
        
        # Apply aggregation using pandas groupby
        if aggregation:
            agg_lower = aggregation.lower()
            
            # Detect groupby operations
            if 'grouped by' in agg_lower or 'group by' in agg_lower:
                # Find the grouping column
                group_col = None
                for col in available_cols:
                    if col.lower() in agg_lower:
                        group_col = col
                        break
                
                if group_col and group_col in df.columns:
                    # Get numeric columns for aggregation
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if numeric_cols:
                        # Determine aggregation function
                        if 'mean' in agg_lower or 'average' in agg_lower or 'avg' in agg_lower:
                            grouped = df.groupby(group_col)[numeric_cols].mean()
                        elif 'sum' in agg_lower or 'total' in agg_lower:
                            grouped = df.groupby(group_col)[numeric_cols].sum()
                        elif 'count' in agg_lower:
                            grouped = df.groupby(group_col).size().to_frame('count')
                        elif 'max' in agg_lower or 'maximum' in agg_lower:
                            grouped = df.groupby(group_col)[numeric_cols].max()
                        elif 'min' in agg_lower or 'minimum' in agg_lower:
                            grouped = df.groupby(group_col)[numeric_cols].min()
                        else:
                            # Default to mean
                            grouped = df.groupby(group_col)[numeric_cols].mean()
                        
                        print(f"[AGGREGATION] Grouped by {group_col}, computed {agg_lower}")
                        return f"DATA RETRIEVED (Aggregated):\n{grouped.to_string()}\n"
            
            # Summary statistics
            elif 'summary' in agg_lower or 'statistics' in agg_lower or 'describe' in agg_lower:
                return f"DATA RETRIEVED (Summary):\n{df.describe().to_string()}\n"
        
        # Return results based on size
        if len(df) == 0:
            return "DATA RETRIEVED: No rows match the criteria"
        elif len(df) == 1:
            # Single row - show as key-value pairs
            result = "DATA RETRIEVED (1 row):\n"
            for col in df.columns:
                result += f"  {col}: {df.iloc[0][col]}\n"
            return result
        elif len(df) <= 20:
            # Small dataset - show all rows
            return f"DATA RETRIEVED ({len(df)} rows):\n{df.to_string(index=False)}\n"
        else:
            # Large dataset - show summary + sample
            summary = df.describe().to_string()
            sample = df.sample(min(20, len(df))).to_string(index=False)
            return f"DATA RETRIEVED ({len(df)} total rows):\n\nSUMMARY STATISTICS:\n{summary}\n\nSAMPLE (20 rows):\n{sample}\n"

    def ask(self, question: str, max_iterations: int = 3) -> str:
        """
        Answer question with CoT reasoning and dynamic data retrieval.
        
        Args:
            question: User's question
            max_iterations: Maximum rounds of data retrieval (prevents infinite loops)
        """
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        # Add basic context for simple questions
        context = self._get_basic_context()
        conversation.append({
            "role": "system", 
            "content": f"BASIC CONTEXT:\n{context}\n\nYou may use this context for simple questions, or request specific data for complex analysis."
        })
        
        iteration = 0
        full_response = ""
        
        while iteration < max_iterations:
            # Get agent response
            if iteration == 0:
                response = self.client.chat_interactive(
                    user_message=conversation[-2]["content"],
                    system_prompt=self.system_prompt + "\n" + conversation[-1]["content"]
                )
            else:
                # Continue conversation with new data
                response = self.client.chat_interactive(
                    user_message=conversation[-1]["content"],
                    system_prompt=self.system_prompt
                )
            
            full_response += response + "\n\n"
            
            # Check if agent needs data
            data_request = self._extract_data_request(response)
            
            if data_request:
                print(f"\n[Agent requesting data: {data_request.get('dataset')} - {data_request.get('columns')}]")
                
                # Fulfill request
                data_response = self._fulfill_data_request(data_request)
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "user", "content": data_response})
                
                full_response += f"[Data retrieved]\n\n"
                iteration += 1
            else:
                # No more data needed, return final response
                return response
        
        return full_response + "\n[Reached maximum iteration limit]"

    def _get_basic_context(self) -> str:
        """Get basic dataset statistics for simple queries."""
        context = ""
        
        if self.data_sources['songs_with_topics'] is not None:
            df = self.data_sources['songs_with_topics']
            context += f"Dataset: {len(df)} songs across {df['era'].nunique()} eras\n"
            context += f"Eras: {', '.join(df['era'].unique())}\n"
            context += f"Albums: {df['album_clean'].nunique()} total\n\n"
            
            context += "Songs per era:\n"
            context += df['era'].value_counts().to_string() + "\n"
        
        return context

    def suggest_insights(self) -> str:
        """Agent proactively suggests insights using available data."""
        prompt = """Based on the available Taylor Swift datasets, suggest 2-3 interesting 
        analytical questions that could reveal insights about her musical evolution, 
        themes, or style changes. For each suggestion, briefly explain what data would 
        be needed to investigate it."""
        
        return self.ask(prompt)

    def reset(self):
        """Reset conversation history."""
        self.client.reset_conversation()
        self.conversation_history = []


def interactive_session():
    """Start an interactive chat session with enhanced agent."""
    print("=" * 80)
    print("ENHANCED TAYLOR SWIFT ANALYSIS ASSISTANT")
    print("Chain-of-Thought Reasoning + Dynamic Data Retrieval")
    print("=" * 80)
    print(f"Model: {config.MODEL}")
    print("\nInitializing agent...")

    try:
        agent = AnalysisAssistant()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure Ollama is running or OpenAI is configured")
        return

    print("Agent ready!")
    print("\nCommands:")
    print("  - Type your question (complex questions will trigger CoT reasoning)")
    print("  - 'insights' for AI-suggested research questions")
    print("  - 'columns' to see available data columns")
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

            if question.lower() == 'columns':
                print("\nAVAILABLE DATA COLUMNS:")
                for col, desc in DataDictionary.COLUMNS.items():
                    print(f"  {col}: {desc}")
                continue

            if question.lower() == 'insights':
                print("\nAgent: Generating insight suggestions...\n")
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
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    interactive_session()
