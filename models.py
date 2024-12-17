
from statsmodels.tsa.statespace.sarimax import SARIMAX
from autots import AutoTS
from transformers import AutoModelForCausalLM
import random
import warnings
import itertools
import numpy as np
warnings.filterwarnings('ignore')
# SARIMA Model functions
def fit_sarimax_model(curr_data, order, seasonal_order):
    """Helper function to fit SARIMAX model with various fallback options"""
    try:
        # First attempt with original parameters but more stable settings
        model = SARIMAX(curr_data,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=True,  # Changed to False for better stability
                       enforce_invertibility=True,)
        
        results = model.fit(disp=True)
        # print("GOT MODEL 1")
        return results
    
    except:
        
            # Second attempt with simplified model
            simplified_order = (1, 1, 1)
            simplified_seasonal = (0, 1, 1, 12)
            
            model = SARIMAX(curr_data,
                          order=simplified_order,
                          seasonal_order=simplified_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False,
                          initialization='approximate_diffuse')
            
            results = model.fit(disp=False,
                              method='lbfgs',
                              optim_score='harvey',
                              maxiter=1000)
            # print("GOT MODEL 2  ")
            return results
    
def get_best_params_for_SARIMA(y, n_samples=30):
    d = range(1, 5)
    p = range(0, 3)
    q = range(1, 5)
    P = D = Q = range(1, 5)
    
    # Generate all possible parameter combinations
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in itertools.product(P, D, Q)]
    
    # Randomly sample n combinations from the parameter grid
    random_pdq = random.sample(pdq, min(n_samples, len(pdq)))
    random_seasonal_pdq = random.sample(seasonal_pdq, min(n_samples, len(seasonal_pdq)))

    best_aic = np.inf
    best_param = None
    best_param_seasonal = None

    # Perform random search
    for param in random_pdq:
        for param_seasonal in random_seasonal_pdq:
            try:
                model = SARIMAX(y,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=True,
                                enforce_invertibility=True)

                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_param = param
                    best_param_seasonal = param_seasonal
            except Exception as e:
                # If there's an error, continue to the next parameter combination
                continue

    return best_param, best_param_seasonal

# auto_ts model functions
def get_AUTO_TS_model(data, value_column):
    model = AutoTS(
                forecast_length=2,
                frequency='infer',
                prediction_interval=0.9,
                ensemble='auto',
                model_list="fast",  # "superfast", "default", "fast_parallel"
                transformer_list="fast",  # "superfast",
                drop_most_recent=1,
                max_generations=4,
                num_validations=2,
                validation_method="backwards",
                verbose=0,
            )
    model = model.fit(
                data,
                date_col='date',
                value_col=value_column,
                id_col='series_id' ,
            )
    return model

# Time-MOE model functions
def get_Time_MOE_model():       
    model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,)
    return model

def normalize_data(data):
    mean, std = data.mean(dim=-1, keepdim=True), data.std(dim=-1, keepdim=True)
    normed_seqs = (data - mean) / std
    return normed_seqs, mean, std