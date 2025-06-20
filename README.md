# Multi-Agent Stock Analysis

This project implements a **multi-agent system for stock analysis** using Large Language Models (LLMs), time series forecasting, and web scraping. The system leverages multiple specialized AI agents to analyze news, predict stock trends, and assess various business, technical, and ethical factors affecting publicly traded companies.

## Features

- **Multi-Agent Evaluation:** Agents act as financial analysts, marketing experts, innovation analysts, risk assessors, technology experts, consumer behavior analysts, environmental impact assessors, ethical implications analysts, and global market analysts.
- **Automated News Scraping:** Fetches and processes recent news articles relevant to a given stock ticker and time window.
- **LLM-Based Scoring:** Each agent uses a prompt template and an LLM (e.g., Gemini, OpenAI) to score the impact of news on the company from its perspective.
- **Time Series Forecasting:** Integrates models like LSTM, SARIMA, AutoTS, and TIME-MOE for price/returns prediction.
- **Extensible Tooling:** Modular toolkit for scraping, PDF reading, code execution, and more.
- **Configurable & Modular:** Easily add new agent types, models, or tools.

## Directory Structure

```
.
├── customAgents/           # Core agent, tool, and runtime implementations
├── Modeling/               # Time series models and data
├── test_agent.py           # Example: multi-agent news analysis for a ticker
├── prepare_ts.py           # Example: time series forecasting and agent scoring
├── prompt.py               # Agent prompt templates
├── lstm.py                 # LSTM model implementation
├── models.py               # Additional model utilities
├── config.json             # API keys and model config
├── requirements.txt        # Python dependencies
└── README.md               # (You are here)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Multi-agent-stock-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   - Edit `config.json` and add your [Google Generative AI](https://ai.google.dev/) and [SerpAPI](https://serpapi.com/) keys:
     ```json
     {
       "google_api_key": "YOUR_GOOGLE_API_KEY",
       "model": "gemini-1.5-flash",
       "serpapi_key": "YOUR_SERPAPI_KEY"
     }
     ```

## Usage

### 1. Multi-Agent News Analysis

Run the example to analyze the impact of news on a stock (e.g., NVDA):

```bash
python test_agent.py
```

- The script will:
  - Search for recent news about the ticker.
  - Scrape and clean the articles.
  - Run each agent's LLM prompt on the articles.
  - Output scores for each agent and a final aggregated score.

### 2. Time Series Forecasting + Agent Scoring

Run the time series preparation and forecasting pipeline:

```bash
python prepare_ts.py
```

- The script will:
  - Download historical stock data.
  - Run multiple forecasting models (LSTM, SARIMA, AutoTS, TIME-MOE).
  - For each time step, fetch news, run agent scoring, and save results.

### 3. Custom Agents and Tools

- Add new agent prompts in `prompt.py`.
- Implement new tools in `customAgents/agent_tools/`.
- Extend models in `Modeling/` or `lstm.py`.

## Example Output

```
Document 1
Scoring...
financial_analyst: 0.85
marketing_expert: 0.70
...
final_score: 0.78
```

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Project Structure

- **customAgents/**: Core agent logic, toolkits, LLM wrappers, and runtime environments.
- **Modeling/**: Time series models and data.
- **prompt.py**: Prompt templates for each agent type.
- **test_agent.py**: Example script for multi-agent news analysis.
- **prepare_ts.py**: Example script for time series forecasting and agent scoring.

## Extending the System

- **Add a new agent:** Define a new prompt in `prompt.py` and update the agent list.
- **Add a new tool:** Implement a new class in `customAgents/agent_tools/` inheriting from `BaseTool`.
- **Add a new model:** Place your model code in `Modeling/` and update the pipeline in `prepare_ts.py`.

## License

[Specify your license here]

## Acknowledgements

- [Google Generative AI](https://ai.google.dev/)
- [SerpAPI](https://serpapi.com/)
- [LangChain](https://python.langchain.com/)
- [PyTorch](https://pytorch.org/)
- [Yahoo Finance](https://finance.yahoo.com/)

---

Let me know if you want to tailor this further, add usage screenshots, or include more technical details!