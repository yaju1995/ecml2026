from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


# -----------------------------
# LOAD PARQUET FILE
# -----------------------------
region = "Luxembourg"
path = Path(__file__).parent.parent.parent / "data" / "processed" / "Irradiance_2021_2025_Elia_Consumption_Forecast" / f"processed_irradiance_forecast_production_{region}_2021_2025.parquet"
df = pd.read_parquet(path)

# Ensure datetime format
df["datetime"] = pd.to_datetime(df["datetime"])

# Remove timezone for consistency
df["datetime"] = df["datetime"].dt.tz_localize(None)

# Sort chronologically
df = df.sort_values("datetime")

# -----------------------------
# GROUP BY DAY
# -----------------------------
df["date"] = df["datetime"].dt.date
grouped = df.groupby("date")
n_days = len(grouped)

# Initialize arrays dynamically
solar_measured = np.zeros((n_days, 96))       # 96 steps per day (15-min intervals)
solar_forecast_6pm = np.zeros((n_days, 96))
solar_forecast_11am = np.zeros((n_days, 96))

# -----------------------------
# FILL ARRAYS WITHOUT RESAMPLING
# -----------------------------
day_index = 0

for idx, day_df in grouped:

    # Value Correction due to timezone (night so NaN should be 0 in any case)
    if np.isnan(day_df["measured_irradiance"].values[91]):
        day_df["measured_irradiance"].values[91] = 0.0
    if np.isnan(day_df["measured_irradiance"].values[87]):
        day_df["measured_irradiance"].values[87] = 0.0

    solar_measured_val = day_df["measured_irradiance"].values
    solar_forecast_6pm_val = day_df["dayahead6pmforecast_irradiance"].values
    solar_forecast_11am_val = day_df["dayahead11amforecast_irradiance"].values

    s = len(day_df)
    if s != 96:
        if s > 96:
            solar_measured_val = solar_measured_val[:96]
            solar_forecast_6pm_val = solar_forecast_6pm_val[:96]
            solar_forecast_11am = solar_forecast_11am[:96]
        else:
            solar_measured_val = np.concatenate([solar_measured_val,np.array([0 for i in range (96-s)])], axis = 0 )
            solar_forecast_6pm_val = np.concatenate([solar_forecast_6pm_val,np.array([0 for i in range (96-s)])], axis = 0 )
            solar_forecast_11am_val= np.concatenate([solar_forecast_11am_val,np.array([0 for i in range (96-s)])], axis = 0 )


    solar_measured[day_index] = solar_measured_val
    solar_forecast_6pm[day_index] = solar_forecast_6pm_val
    solar_forecast_11am[day_index] = solar_forecast_6pm_val
    day_index += 1
# Trim unused rows (if incomplete days were skipped)
solar_measured = solar_measured[:day_index]
solar_forecast_6pm = solar_forecast_6pm[:day_index]
solar_forecast_11am = solar_forecast_11am[:day_index]

# -----------------------------
# OPTIONAL: convert W/m² → kW/m² and apply efficiency
# -----------------------------
solar_measured *= 0.001 * 0.22
solar_forecast_6pm *= 0.001 * 0.22
solar_forecast_11am *= 0.001 * 0.22

### NON FLEXIBLE DEMAND FROM DATA ###
# Open avg_consumption.csv file to get the average consumption per 15 minutes

avg_consumption_path = Path(__file__).parent.parent.parent / "data" / "processed" / "non_flexible_load_SRD_2022" / "avg_consumption.csv"

avg_consumption = pd.read_csv(avg_consumption_path, low_memory=False)
 
avg_consumption = avg_consumption.iloc[:, 1:].values  # Remove the first column (time)
avg_consumption = avg_consumption.flatten()
 

# Ensure datetime format
df["datetime"] = pd.to_datetime(df["datetime"])
df_filtered = df.copy()

# Filter year 2022
df_2022 = df_filtered[df_filtered["datetime"].dt.year == 2024].copy()

# Sort just in case
df_2022 = df_2022.sort_values("datetime")


def generate_nf_consumption(num_nf_cons:int, num_episodes:int, T:int, avg_cons = avg_consumption):
    """Generate the total nf_consumption of num_nf_cons non-flexible consumer"""
    avg_cons_len = len(avg_cons)
    num_of_ep_in_data = len(avg_cons) // T
    consumption = np.zeros((num_of_ep_in_data, T))

    for i in range(num_of_ep_in_data):
        start = T*i
        end = start + T
        consumption[i] = avg_cons[start:end]
    
    extension_factor = int(np.ceil(num_episodes / num_of_ep_in_data)) #Number of duplication of avg_cons needed to cover num_episodes
    consumption = np.array(extension_factor * list(num_nf_cons*consumption)) 

    return consumption
 
 
## GENERATE PRICE SIGNALS FROM NET ENERGY ##
def generate_price_from_net_energy(num_episodes: int, T: int, seed: int = None, n_conso=20, PV_area = 1200.0) -> np.ndarray:
    """
    Generate price signal per episode using 1 - (energy_available / max_daily_energy_available),
    accounting for unit conversion between kW (solar) and W (demand).
 
    Args:
        num_episodes (int): Number of episodes (days)
        T (int): Time steps per episode
        seed (int): Optional random seed for reproducibility
 
    Returns:
        np.ndarray: Price signal in [0, 1], shape [num_episodes, T]
    """
 
    solar_measured_data_copy = PV_area*solar_measured.copy()
    solar_forecast_11am_copy = PV_area*solar_forecast_11am.copy()*PV_area
    solar_forecast_6pm_copy = PV_area*solar_forecast_6pm.copy()
    
    if solar_measured_data_copy is None or solar_forecast_11am_copy is None or solar_forecast_6pm_copy is None:
        raise ValueError("solar_measured / forecast must be defined")
 
    n_days = solar_measured_data_copy.shape[0]
    price_signals = np.zeros((num_episodes, T))
    forecast_6pm_price_signals = np.zeros((num_episodes, T))
    forecast_11am_price_signals = np.zeros((num_episodes,T))
 
    solar_pow = np.zeros((num_episodes, T))
    forecast_6pm_solar_pow = np.zeros((num_episodes, T))
    forecast_11am_solar_pow = np.zeros((num_episodes, T))

 
    # Generate fixed non-flexible demand
    # demand_profiles = generate_fixed_demand_scenarios(num_episodes, T, seed=seed)
    
    # True non-flexible demand profiles
    demand_profiles = generate_nf_consumption(num_nf_cons=n_conso, num_episodes=num_episodes, T=T)  # Scale to match the number of non-flexible consumption profiles
 
 
    # price(i,t) = 1 - net_solar_energy(i,t)/max_i{(net_solar_energy(i,t))}
    for i in range(num_episodes):

        price_signals[i], solar_pow[i] = compute_daily_price(solar_measured_data_copy, demand_profiles, i, n_days, T)
        forecast_6pm_price_signals[i], forecast_6pm_solar_pow[i] = compute_daily_price(solar_forecast_6pm_copy, demand_profiles, i, n_days, T)
        forecast_11am_price_signals[i], forecast_11am_solar_pow[i] = compute_daily_price(solar_forecast_11am_copy, demand_profiles, i, n_days, T)

    # nan_indices = np.where(np.isnan(forecast_11am_solar_pow)) 
    # print("Number of NaNs:", len(nan_indices[0]))
    # print("Episode indices with NaNs:", np.unique(nan_indices[0]))
    # print("Time steps with NaNs:", np.unique(nan_indices[1]))
    # nan_dict = {ep : [i for i in np.where(np.isnan(solar_pow[ep]))] for ep in np.unique(nan_indices[0])}
    # print(nan_dict)
    return price_signals, solar_pow, forecast_6pm_price_signals, forecast_6pm_solar_pow, forecast_11am_price_signals, forecast_11am_solar_pow, demand_profiles

def compute_daily_price(solar_data, demand_profiles, episode, n_days, T):

    day_index = episode % n_days
    day_solar = solar_data[day_index]*1000  # Convert from kW to W
    net_energy = day_solar - demand_profiles[episode] # Subtract demand from solar energy to obtain net available solar power
    net_energy = np.clip(net_energy, 0, None)

    max_energy = np.max(net_energy)
    if max_energy > 0:
        normalized_energy = net_energy / max_energy
    else:
        normalized_energy = np.zeros_like(net_energy)

    price = np.clip(1 - normalized_energy, 0 , 1)
    solar_power = day_solar
    return price, solar_power