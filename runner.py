
import argparse
import pandas as pd
from typing import Callable
from src.data_generation.environment import EnvironmentDataDescriptor, MockEnvironment
from src.data_generation.simulate import SimulationConfig, simulate
from src.data_generation import price_generation_functions
from src.data_generation.convert_batch import BatchWriter

from os.path import exists

from typing import Dict
from pathlib import Path
import time


def get_save_simulation_data_function(no_save=False, folder_name=None):
    timestr = time.strftime("%Y-%m-%d %Hh %Mm %Ss")
    folder_path = Path(f"./simulated_data/{folder_name or timestr}")
    if not no_save:
        folder_path.mkdir(parents=True, exist_ok=True)
    
    def save_simulation_data(simulation_row: Dict, prosumer_name: str, simulation_step_idx: int):
        
        simulation_row_df = pd.DataFrame([simulation_row], index=[simulation_step_idx])

        file_path = folder_path.joinpath(f"{prosumer_name}.csv")
        
        include_header = (not file_path.is_file())
        
        if not no_save:
            simulation_row_df.to_csv(file_path, header=include_header, mode="a")
    return save_simulation_data
        

def run(folder_name: str, price_generation_function: Callable, generate_batch_data=False, prosumer_noise_scale=0.1, generation_noise_scale=0.1,):
    # build environment
    num_prosumers = 10
    environment_data_descriptor = EnvironmentDataDescriptor(
        time_col_idx=0,
        day_of_week_col_idx=1,
        price_col_idx=2,
        solar_gen_col_idx=3,
        temp_col_idx=4,
        prosumer_col_idx_list=list(range(5, 5 + num_prosumers)),
        battery_nums=[50]*num_prosumers,
        pv_sizes=[100]*num_prosumers,
        prosumer_noise_scale=prosumer_noise_scale,
        generation_noise_scale=generation_noise_scale,
    )
    building_data_df = pd.read_csv("./building_data.csv").interpolate().fillna(0)
    mock_environment = MockEnvironment(
        building_data_df=building_data_df,
        environment_data_descriptor=environment_data_descriptor,
    )
    
    # build simulation
    simulation_config = SimulationConfig(
        num_simulation_steps=12000,
        day_start=13,
        year_start=2012,
        prices_generation_function=price_generation_function,
    )
    
    if generate_batch_data:
        batch_writer = BatchWriter(
            f"./batch_data/{folder_name}"
        )
    else:
        batch_writer = None
    
    simulate(
        mock_environment=mock_environment,
        simulation_config=simulation_config,
        write_data=get_save_simulation_data_function(folder_name=folder_name),
        batch_writer=batch_writer,
    )
    
    
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--price_generation_function", type=str, default="constant_prices_generation_function")
    parser.add_argument("--offset_multiplier", type=float, default=0.1)
    parser.add_argument("--scale_multiplier", type=float, default=0.1)
    parser.add_argument("--generate_batch_data", default=False, action='store_true')
    parser.add_argument("--prosumer_noise_scale", type=float, default=0.1)
    parser.add_argument("--generation_noise_scale", type=float, default=0.1)
    args = parser.parse_args()
    run(
        args.folder_name, 
        getattr(price_generation_functions, f"get_{args.price_generation_function}")(**vars(args)), 
        args.generate_batch_data,
        args.prosumer_noise_scale,
    )
