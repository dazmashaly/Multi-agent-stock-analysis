from customAgents.agent_llm import SimpleInvokeLLM
from customAgents.agent_prompt import BasePrompt
from customAgents.runtime import SimpleRuntime
from prompt import agent_prompts
import json
import numpy as np
from serpapi import GoogleSearch
import datetime
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import re
import time

def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

config = load_config()
llm = SimpleInvokeLLM(model=config['model'], api_key=config['google_api_key'], temperature=0.0)

def clean_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Replace multiple line breaks with a single line break
    text = re.sub(r'\n+', '\n', text)
    # Strip leading and trailing whitespace
    return text



def google_search(query,start_date,end_date, num_results=5):
    # change date format to MM/DD/YYYY
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    start_date = start_date.strftime("%m/%d/%Y")
    end_date = end_date.strftime("%m/%d/%Y")
    params = {
        "engine": "google",
        "q": query,
        "api_key": config["serpapi_key"],  # Using the API key from https://serpapi.com/manage-api-key
        "num": num_results,
        "tbs": f"cdr:1,cd_min:{start_date},cd_max:{end_date}"
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    if "organic_results" in results:
        return results["organic_results"]
    else:
        return []

def scrape_urls(urls):
    
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    docs_transformed = [clean_text(doc.page_content) for doc in docs_transformed if doc is not None]
    return docs_transformed



def get_agent_rate(tickerSymbol, start_date, pred_date):
    agent_types = list(agent_prompts.keys())
    scores = {}
    final_socres = []
    query = f"{tickerSymbol} news"
    search_results = google_search(query, start_date, pred_date)
    links = [result["link"] for result in search_results]
    docs = scrape_urls(links)
    
    for i,doc in enumerate(docs,1):
        
        print(f"Document {i}")
        time.sleep(30)
        if len(doc) < 250:
            print("Document too short")
            continue
        print("Scoring...")
        try:
            for agent_type in agent_types:
                time.sleep(3)
                prompt = BasePrompt(text=agent_prompts[agent_type])
                prompt.construct_prompt({"article_description": doc, "tickerSymbol": tickerSymbol, "start_date": start_date, "pred_date": pred_date})
                agent = SimpleRuntime(llm, prompt)
                agent_rate = float(agent.loop()) 
                scores[agent_type] = agent_rate
                print(f"{agent_type}: {agent_rate}")
                final_socres.append(np.mean(list(scores.values())))
            scores["final_score"] = np.mean(list(scores.values()))
            with open(f"scraped\\{tickerSymbol}_doc_{i}_{start_date}_{pred_date}.txt", "w", encoding="utf-8") as f:
                f.write(doc)
                f.write("\n\n")
                f.write(f"Scores: {scores}")
                f.write("\n\n")
        except Exception as e:
            print(f"failed to score document{i}")
            print(f"Error: {e}")
            continue


    return np.mean(final_socres)

tickerSymbol = "NVDA"
start_date = "2020-01-01"
pred_date = "2024-01-08"

print(get_agent_rate(tickerSymbol, start_date, pred_date))