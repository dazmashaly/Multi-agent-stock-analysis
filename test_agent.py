from customAgents.agent_llm import SimpleInvokeLLM
from customAgents.agent_prompt import BasePrompt
from customAgents.runtime import SimpleRuntime
from prompt import agent_prompts
import json
import numpy as np

def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

config = load_config()
llm = SimpleInvokeLLM(model=config['model'], api_key=config['api_key'], temperature=0.0)

def get_agent_rate(article_description, tickerSymbol, start_date, pred_date):
    agent_types = list(agent_prompts.keys())
    scores = {}

    for agent_type in agent_types:
        prompt = BasePrompt(text=agent_prompts[agent_type])
        prompt.construct_prompt({"article_description": article_description, "tickerSymbol": tickerSymbol, "start_date": start_date, "pred_date": pred_date})
        agent = SimpleRuntime(llm, prompt)
        agent_rate = float(agent.loop())
        scores[agent_type] = agent_rate
        print(f"{agent_type}: {agent_rate}")

    return np.mean(list(scores.values()))

article_description = """
Breaking News: Tech Giant XYZ Corp Unveils Revolutionary AI-Powered Product

XYZ Corp, a leading technology company, has just announced the launch of its groundbreaking AI-powered product, 'IntelliSense Pro.' This innovative solution combines advanced machine learning algorithms with real-time data processing to revolutionize decision-making across various industries.

Key features of IntelliSense Pro include:
1. Predictive analytics for business forecasting
2. Natural language processing for enhanced customer interactions
3. Computer vision capabilities for quality control in manufacturing
4. Adaptive learning systems for personalized user experiences

Industry experts predict that IntelliSense Pro could potentially disrupt multiple sectors, including finance, healthcare, and retail. The product's ability to process vast amounts of data and provide actionable insights in real-time is expected to significantly improve operational efficiency and drive innovation.

XYZ Corp's CEO, Jane Smith, stated, "IntelliSense Pro represents a major leap forward in AI technology. We believe this product will empower businesses to make smarter decisions, optimize their processes, and stay ahead in an increasingly competitive global market."

The company has also announced partnerships with several Fortune 500 companies for early adoption and implementation of IntelliSense Pro. These collaborations are expected to provide valuable case studies and further refine the product's capabilities.

As XYZ Corp prepares for a wide-scale rollout in the coming months, industry analysts are closely watching the potential impact on the AI market and the broader implications for digital transformation across various sectors.

"""
tickerSymbol = "NVDA"
start_date = "2020-01-01"
pred_date = "2024-05-01"

print(get_agent_rate(article_description, tickerSymbol, start_date, pred_date))