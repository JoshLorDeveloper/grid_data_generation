
# Create batch data from existing raw data
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from src.data_generation.environment import EnvironmentDataDescriptor, MockEnvironment
from src.data_generation.utils.constants import BATTERY_NUMS, DAY_LENGTH, DAY_START, NUM_PROSUMERS, YEAR_LENGTH
from src.data_generation.convert_batch import BatchWriter
from src.data_generation.simulate import get_observation


def create_batch(dfs : list[pd.DataFrame],  mock_environment : MockEnvironment, batch_writer : BatchWriter):
    
    
    # Extract and normalize the buy, sell, and prosumer cols
    sentinel_df = dfs[0]
    sentinel_cols = sentinel_df.columns.to_list()
    
    buy_price_cols = [x for x in sentinel_cols if "agent_buy" in x]
    sell_price_cols = [x for x in sentinel_cols if "agent_sell" in x]
    prosumer_demand_cols = [x for x in sentinel_cols if "prosumer_response" in x]
    for simulation_step_idx in range(sentinel_df.shape[0]):
        
        if simulation_step_idx != sentinel_df.loc[simulation_step_idx, "step"]:
            print("misaligned batch data generation")
        
        simulate_day = sentinel_df.loc[simulation_step_idx, "day"]
        
        microgrid_buy_prices = sentinel_df.loc[simulation_step_idx, buy_price_cols].values
        microgrid_sell_prices = sentinel_df.loc[simulation_step_idx, sell_price_cols].values

        total_demand = np.zeros(DAY_LENGTH, dtype=np.float64)
        for df in dfs:
            prosumer_demand = np.array(df.loc[simulation_step_idx, prosumer_demand_cols], dtype=np.float64)
            total_demand+=prosumer_demand
        
        step_reward = sentinel_df.loc[simulation_step_idx, "reward"]
        
        batch_writer.write_batch(
            simulation_step_idx, 
            np.concatenate([microgrid_buy_prices, microgrid_sell_prices]),
            get_observation(
                total_demand,
                mock_environment.hourly_solar_constants.loc[simulate_day, :].values,
                mock_environment.utility_hourly_buy_prices.loc[simulate_day, :].values,  
            ),
            step_reward,
        )

def setup():
    
    # build environment
    environment_data_descriptor = EnvironmentDataDescriptor(
        time_col_idx=1,
        day_of_week_col_idx=None,
        price_col_idx=3,
        solar_gen_col_idx=2,
        temp_col_idx=None,
        prosumer_col_idx_list=list(range(4, 4 + NUM_PROSUMERS)),
        battery_nums=BATTERY_NUMS,
        pv_sizes=None,
        prosumer_noise_scale=None,
        generation_noise_scale=None,
    )
    building_data_df = pd.read_csv("./building_data/building_demand_2016.csv").interpolate().fillna(0)
    building_metadata_df =  pd.read_csv("./building_data/building_metadata.csv", index_col="building_id")
    mock_environment = MockEnvironment(
        building_data_df=building_data_df,
        building_metadata_df=building_metadata_df,
        environment_data_descriptor=environment_data_descriptor,
    )
    return mock_environment

def get_dataframes(folder_name, run_folder_name):
    # Get all dataframes
    folder_path = Path("simulated_data").joinpath(folder_name).joinpath(run_folder_name)
    data_files = [p for p in folder_path.iterdir() if p.is_file()]
    dfs = [pd.read_csv(data_file) for data_file in data_files]
    return dfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--run_folder_name", type=str)

    args = parser.parse_args()
    
    dfs = get_dataframes(args.folder_name, args.run_folder_name)
    
    mock_environment = setup()
    
    batch_writer = BatchWriter(
        f"./batch_data/{args.folder_name}"
    )
    
    create_batch(dfs, mock_environment, batch_writer)
