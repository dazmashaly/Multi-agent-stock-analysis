import random
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *


def get_agent_rate(tickerSymbol, start_date, pred_date):
    return random.random()

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
    return df




def prepare_data(tickerSymbol, start_date,end_date,model ,price=False, plot=False):
    tsa_preds = []
    rates = []
    final_result = []
    
    data = get_data(tickerSymbol,start_date,end_date)
    value_column = "price" if price else "returns"

    for idx, r in tqdm(data.iloc[25:].iterrows(), total=len(data.iloc[25:])):
        try:
            if model == "SARIMA":
                
                curr_data = data[[value_column,"date"]]
                curr_data.set_index("date",inplace=True)
                curr_data = curr_data[value_column][:idx]
                order,seasonal_order = get_best_params_for_SARIMA(curr_data)
                results = fit_sarimax_model(curr_data, order,seasonal_order)
                pred = results.forecast()
                pred_date = f"{pred.index[0].year}-{pred.index[0].month}-{pred.index[0].day}"
                pred_date = datetime.datetime.strptime(pred_date, "%Y-%m-%d")
                pred = pred.values[0]
                if pred == 0:
                    pred = curr_data[-1]


            elif model == "AutoTS":
                curr_data = data[:idx]
                model = get_AUTO_TS_model(curr_data, value_column)

                prediction = model.predict()

                # Get forecast
                pred = prediction.forecast.values[1][0]
                pred_date = f"{prediction.forecast.index[1].year}-{prediction.forecast.index[1].month}-{prediction.forecast.index[1].day}"
                pred_date = datetime.datetime.strptime(pred_date, "%Y-%m-%d")


            elif model == "TIME-MOE":
                curr_data = data[:idx]
                curr_data.dropna(inplace=True)
                seq = torch.tensor(curr_data[[value_column]].astype(np.float32).to_numpy())
                seq = seq.transpose(1,0)
                if value_column == "price":
                    normed_seqs, mean, std = normalize_data(seq)
                else:
                    normed_seqs = seq
                    mean = 0
                    std = 1
                model = get_Time_MOE_model()
                prediction_length = 2
                output = model.generate(normed_seqs, max_new_tokens=prediction_length) 
                normed_predictions = output[:, -prediction_length:] 

                # inverse normalize
                predictions = normed_predictions * std + mean
                pred = predictions[0][0].numpy() + 0
                pred_date = data.iloc[idx]['date']                

            else:
                raise Exception("Invalid Model plase use SARIMA, AutoTS or TIME-MOE")

                
            
            tsa_preds.append(pred)
            
            # Get Agent rate
            pred_date = pd.to_datetime(pred_date)
            end_date = pred_date - pd.Timedelta(weeks=1)
            start_date = end_date - pd.Timedelta(weeks=1)
            
            agent_rate = get_agent_rate(tickerSymbol, start_date, pred_date)
            rates.append(agent_rate)
            
            final_result.append({
                "tsa": pred,
                "agent": agent_rate,
                "date": pred_date.strftime('%Y-%m-%d'),
                "label": data.iloc[idx][value_column],
                "error": pred - data.iloc[idx][value_column]
            })
            
        except Exception as e:
            print(f"Error at date {data[idx]}:")
            print(f"Error message: {str(e)}")
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

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "PYPL", "INTC", "CSCO"]
    final_data = pd.DataFrame()
    model = "TIME-MOE"
    for ticker in tickers:
        final_result = prepare_data(ticker,"2023-01-01","2024-10-01",model,False,False)
        new_data = pd.DataFrame(final_result[-1])
        new_data["model"] = model
        new_data["ticker"] = ticker
        new_data.to_csv(f"{ticker}_{model}.csv",index=False)
        final_data = pd.concat([final_data,new_data])
        final_data.to_csv("final_data.csv",index=False)