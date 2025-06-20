import random
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *
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
import torch
from lstm import LSTMModelHandler
import os
import logging
import warnings


def clear_terminal():
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For Linux and macOS
    else:
        os.system('clear')

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


def google_search(query,start_date,end_date, num_results=10):
    # change date format to MM/DD/YYYY
    # start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    # end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
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
    html2text = Html2TextTransformer()
    scraped = 0
    docs_transformed = []
    for url in urls:
        loader = AsyncHtmlLoader(url)
        try:
            doc = loader.load()

            doc_transformed = html2text.transform_documents(doc)
            doc_transformed = doc_transformed[0] if doc_transformed else None
            doc_transformed = clean_text(doc_transformed.page_content)
            if len(doc_transformed) > 1500:
                docs_transformed.append(doc_transformed)
                scraped += 1
            else:
                print(f"Document too short: {len(doc_transformed)} characters")
                continue
            if scraped >= 5:
                break
        except Exception as e:
            print(f"Failed to scrape url: {url}")
            print(f"Error: {e}")
            continue
    return docs_transformed


def get_agent_rate(tickerSymbol, start_date, pred_date):
    agent_types = list(agent_prompts.keys())
    scores = {}
    final_socres = []
    query = f"{tickerSymbol} stocks news, stock market"
    search_results = google_search(query, start_date, pred_date)
    links = [result["link"] for result in search_results]
    docs = scrape_urls(links)
    if isinstance(start_date,datetime.datetime):
        start_date = start_date.strftime("%Y-%m-%d")
        pred_date = pred_date.strftime("%Y-%m-%d") 
    for i,doc in enumerate(docs,1):
        
        print(f"Document {i}")
        time.sleep(30)
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
                final_socres.append(agent_rate)
            scores["final_score"] = np.mean(final_socres)

            # Define the directory for storing scraped data
            scraped_dir = "scraped"

            # Ensure the directory exists
            if not os.path.exists(scraped_dir):
                os.makedirs(scraped_dir)
            # Assuming `tickerSymbol`, `i`, `start_date`, `pred_date`, `doc`, and `scores` are already defined
            file_name = f"{tickerSymbol}_doc_{i}_{start_date}_{pred_date}.txt"
            file_path = os.path.join(scraped_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc)
                f.write("\n\n")
                f.write(f"Scores: {scores}")
                f.write("\n\n")
        except Exception as e:
            print(f"failed to score document {i}")
            print(f"Error: {e}")
            continue


    return np.mean(final_socres)


def get_data(tickerSymbol, start_date, end_date):
    data = yf.Ticker(tickerSymbol)
    data = data.history(start=start_date, end=end_date).dropna()
    data = data.resample("W").mean()
    df = pd.DataFrame(data.index, columns=["date"])
    df["price"] = data["Close"].values
    df["returns"] = df["price"].pct_change()
    df["date"] = pd.to_datetime(data.index)
    df["series_id"] = tickerSymbol
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.dropna()
    return df




def prepare_data(tickerSymbol, start_date,end_date,model ,price=False, plot=False):
    tsa_preds = []
    rates = []
    final_result = []
    data = get_data(tickerSymbol,start_date,end_date)
    print(f"Found: {len(data)} Samples in the data")
    value_column = "price" if price else "returns"
    if not os.path.exists("scraped"):
        os.makedirs("scraped")
    av = os.listdir("scraped")
    av = [(i.split("_")[0],i.split("_")[3]) for i in av]
    for idx, r in tqdm(data.iloc[25:].iterrows(), total=len(data.iloc[25:])):
        try:
            if model == "SARIMA":
                
                curr_data = data[[value_column,"date"]]
                curr_data.set_index("date",inplace=True)
                curr_data = curr_data[value_column][:idx-1]
                order,seasonal_order = get_best_params_for_SARIMA(curr_data,3)
                results = fit_sarimax_model(curr_data, order,seasonal_order)
                pred = results.forecast()
                pred_date = f"{pred.index[0].year}-{pred.index[0].month}-{pred.index[0].day}"
                pred_date = datetime.datetime.strptime(pred_date, "%Y-%m-%d")
                pred = pred.values[0]
                if pred == 0:
                    pred = curr_data[-1]


            elif model == "AutoTS":
                curr_data = data[:idx-1]
                AutoTS = get_AUTO_TS_model(curr_data, value_column)

                prediction = AutoTS.predict()
                clear_terminal()
                # Get forecast
                pred = prediction.forecast.values[1][0]
                pred_date = f"{prediction.forecast.index[1].year}-{prediction.forecast.index[1].month}-{prediction.forecast.index[1].day}"
                pred_date = datetime.datetime.strptime(pred_date, "%Y-%m-%d")


            elif model == "TIME-MOE":
                curr_data = data[:idx-1]
                curr_data.dropna(inplace=True)
                seq = torch.tensor(curr_data[[value_column]].astype(np.float32).to_numpy())
                seq = seq.transpose(1,0)
                if value_column == "price":
                    normed_seqs, mean, std = normalize_data(seq)
                else:
                    normed_seqs = seq
                    mean = 0
                    std = 1
                MOE = get_Time_MOE_model()
                prediction_length = 2
                output = MOE.generate(normed_seqs, max_new_tokens=prediction_length) 
                normed_predictions = output[:, -prediction_length:] 

                # inverse normalize
                predictions = normed_predictions * std + mean
                pred = predictions[0][0].numpy() + 0
                pred_date = data.iloc[idx-1]['date']  #timestamp           

            elif model == "LSTM":
                curr_data = data[:idx-1]
                curr_data.dropna(inplace=True)
                curr_data = curr_data[["date",value_column]]
                LSTM = LSTMModelHandler("lstm.json")
                pred = LSTM.fit_predict(curr_data, num_epochs=10, batch_size=16, logging=False, input_seq_len=10) 
                pred_date = data.iloc[idx-1]['date']

            else:
                raise Exception(f"Invalid Model {model} plase use SARIMA, AutoTS, LSTM or TIME-MOE")
            
            # Get Agent rate
            pred_date = pd.to_datetime(pred_date)
            end_date = pred_date - pd.Timedelta(weeks=1)
            start_date = end_date - pd.Timedelta(weeks=1)
            # continue if date is before date
            if (tickerSymbol, start_date.strftime("%Y-%m-%d")) in av:
                continue
            print(f"Getting agent rate for {tickerSymbol} from {start_date} to {end_date}")
            agent_rate = get_agent_rate(tickerSymbol, start_date, pred_date)
            #pdb.set_trace()
            rates.append(agent_rate)            
            tsa_preds.append(pred)
            agent_rate =  get_agent_rate(tickerSymbol, start_date, pred_date)
            
            rates.append(agent_rate)
            final_result.append({
                "tsa": pred,
                "agent": agent_rate,
                "date": pred_date.strftime('%Y-%m-%d'),
                "label": data.iloc[idx-1][value_column],
                "error": pred - data.iloc[idx-1][value_column]
            })
            
        except Exception as e:
            print(f"Error message: {str(e)}")
            print(f"Error at date {data.iloc[idx-1]['date']}:")
            print("Skipping this iteration...")
            continue
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data.values, label='Actual Returns', color='blue')
        
        pred_dates = [pd.to_datetime(item['date']) for item in final_result]
        tsa_preds = [item['tsa'] for item in final_result]
        agent_rates = [item['agent'] for item in final_result]
        
        plt.plot(pred_dates, tsa_preds, label='TSA Predictions', color='red', linestyle='--')
        # plt.plot(pred_dates, agent_rates, label='Agent Rates', color='green', linestyle='--')
        plt.title('Weekly Returns: Actual vs Predictions')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return tsa_preds, rates, final_result

if __name__ == "__main__":

    tickers = ["AAPL", ]
    final_data = pd.DataFrame()
    model = "LSTM"
    for ticker in tickers:
        final_result = prepare_data(ticker,"2023-01-01","2024-10-01",model,True,False)
        new_data = pd.DataFrame(final_result[-1])
        new_data["model"] = model
        new_data["ticker"] = ticker
        new_data.to_csv(f"{ticker}_{model}.csv",index=False)
        final_data = pd.concat([final_data,new_data])
        final_data.to_csv("final_data.csv",index=False)
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.ERROR)
    # crete csv directory
    os.makedirs("csvs",exist_ok=True)
    # ["AAPL", "NFLX","ADBE","AMZN","GOOGL", "MSFT", "TSLA", "META", "NVDA", "PYPL"]
    tickers = ["AAPL", "NFLX","ADBE","AMZN","GOOGL", "MSFT", "TSLA", "META", "NVDA", "PYPL"]
    if os.path.exists("final_data.csv"):
        final_data = pd.read_csv("final_data.csv")
    else:
        final_data = pd.DataFrame()
    models = ["TIME-MOE","AutoTS","SARIMA"]
    for model in models:
        for ticker in tickers:
            final_result = prepare_data(ticker,"2023-01-01","2024-10-01",model,True,False)
            new_data = pd.DataFrame(final_result[-1])
            new_data["model"] = model
            new_data["ticker"] = ticker
            csv_path = os.path.join("csvs",f"{ticker}_{model}.csv")
            new_data.to_csv(csv_path,index=False)
            final_data = pd.concat([final_data,new_data])
            final_data.to_csv(f"final_data_{model}.csv",index=False)
