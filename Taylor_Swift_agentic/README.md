# Taylor Swift Agentic

**Taylor Swift Agentic** is an experimental conversational AI that blends modern *agentic LLM architectures* with a creative twist — reasoning and expression inspired by the cultural and linguistic universe of Taylor Swift. It explores prompt engineering, tool use, and memory design for building lightweight, semi-autonomous agents.

## Concept

The project began as a sandbox for testing **agentic behavior** — systems that can plan, reason, and act using a large language model (LLM) as their reasoning core.  
By framing the agent in a familiar creative domain, it becomes easier to observe how different prompting strategies, memory mechanisms, and evaluation loops affect tone, accuracy, and persistence.

The goal is not to mimic Taylor Swift as a persona, but to use this aesthetic domain as a **testbed** for agent-like control structures.

## Features

- **Agentic control loop** – an orchestrator that plans actions and interprets model outputs  
- **Prompt templates** – modular, editable prompt components for different reasoning tasks  
- **Contextual memory** – conversation or state memory for continuity between turns  
- **LLM backend wrapper** – isolates API calls and allows future backends beyond OpenAI  
- **Experiment tracking hooks** – optional saving of runs, prompts, and results  
- **Configurable personality** – easy to change the style or domain of the agent  

## Repository Structure

```
Taylor_Swift_agentic/
│
├── README.md
├── main.py                     # Entry point for running the agentic analysis
├── demo_agents.py              # Example agents or demo scripts
│
├── setup_agents.sh              # Shell script for environment/agent initialization
│
├── src/                         # Core source code package
│   ├── __init__.py
│   ├── berttopic.py             # BERTopic model integration
│   ├── classification.py        # ML classification logic
│   ├── common_imports.py        # Standardized imports and settings
│   ├── config.py                # Configuration parameters
│   ├── data_loading.py          # Data ingestion and preprocessing
│   ├── era_analysis.py          # Temporal/era-based statistical analysis
│   ├── feature_extraction.py    # Embedding and feature generation
│   ├── preference_analysis.py   # Listener preference modeling
│   ├── preprocessing.py         # Text and metadata preprocessing
│   ├── reputation_analysis.py   # Thematic study of Reputation era
│   ├── similarity_analysis.py   # Similarity metrics and clustering
│   ├── topic_modeling.py        # LDA / BERTopic modeling
│   ├── utils.py                 # General utility functions
│   └── visualization.py         # Plotting and visualization routines
│
└── src/agents/                  # (Planned) Modular LLM agents or reasoning tools

```

## Getting Started

### 1. Clone and set up
```bash
git clone https://github.com/suchitakulkarni/DataScience.git
cd DataScience/Taylor_Swift_agentic
pip install -r requirements.txt
```

### 2. Set environment variables
Create a `.env` file or export variables in your shell:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 3. Run a session
```bash
python main.py 
```
OR
```bash
python demo_agents.py 
```
You’ll enter an interactive mode (or load a predefined prompt chain from `examples/`).  
Outputs and logs will be saved in `results/`.

## Configuration

Edit `config.json` (if present) or parameters at the top of `agent_core.py` to modify:
- Model and temperature settings  
- Personality style / tone templates  
- Memory length and summarization threshold  
- Logging and caching options  

## Development Notes

- The codebase is experimental. Expect incomplete features and evolving APIs.  
- Major work ahead:  
  - Robust testing (`pytest`)  
  - Retry / backoff for API calls  
  - Modular tool plugin system  
  - Improved evaluation of agent responses  
  - CI pipeline  

## Philosophy

This project lives at the playful intersection of **artistic language** and **structured reasoning**.  
By studying how creative tone interacts with procedural reasoning, we learn about both — how LLMs handle identity, and how structured control can be layered over generative fluency.

## Contributing

Contributions are welcome.  
To propose a feature or improvement:
1. Fork the repository  
2. Create a new branch  
3. Submit a pull request with a clear description  

Bug reports, design discussions, and new experiment ideas are equally valuable.

## License

MIT License.  
See `LICENSE` file for details.
