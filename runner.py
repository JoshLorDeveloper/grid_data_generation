import wandb
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
    specific_folder_path = f"{folder_name}/{timestr}" if folder_name else timestr
    folder_path = Path(f"./simulated_data/{specific_folder_path}")
    if not no_save:
        folder_path.mkdir(parents=True, exist_ok=True)
    
    def save_simulation_data(simulation_row: Dict, prosumer_name: str, simulation_step_idx: int):
        
        simulation_row_df = pd.DataFrame([simulation_row], index=[simulation_step_idx])

        file_path = folder_path.joinpath(f"{prosumer_name}.csv")
        
        include_header = (not file_path.is_file())
        
        if not no_save:
            simulation_row_df.to_csv(file_path, header=include_header, mode="a")
    return save_simulation_data
        

def run(folder_name: str, price_generation_function: Callable, no_save=False, generate_batch_data=False, prosumer_noise_scale=0.1, generation_noise_scale=0.1, num_simulation_steps=1000):
    # build environment
    num_prosumers = 49
    environment_data_descriptor = EnvironmentDataDescriptor(
        time_col_idx=1,
        day_of_week_col_idx=None,
        price_col_idx=3,
        solar_gen_col_idx=2,
        temp_col_idx=None,
        prosumer_col_idx_list=list(range(4, 4 + num_prosumers)),
        battery_nums=[50]*num_prosumers,
        pv_sizes=None,
        prosumer_noise_scale=prosumer_noise_scale,
        generation_noise_scale=generation_noise_scale,
    )
    building_data_df = pd.read_csv("./building_data/building_demand_2016.csv").interpolate().fillna(0)
    building_metadata_df =  pd.read_csv("./building_data/building_metadata.csv", index_col="building_id")
    mock_environment = MockEnvironment(
        building_data_df=building_data_df,
        building_metadata_df=building_metadata_df,
        environment_data_descriptor=environment_data_descriptor,
    )
    
    # build simulation
    simulation_config = SimulationConfig(
        num_simulation_steps=num_simulation_steps,
        day_start=1,
        year_start=2016,
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
        write_data=get_save_simulation_data_function(folder_name=folder_name, no_save=no_save),
        batch_writer=batch_writer,
    )

def explicit_bool(parser, arg, nonable=False):
    if arg == "None" and nonable:
        return None
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        parser.error(
            "Boolean value expected, instead got {}. Are None values allowed: {}".format(
                arg, nonable
            )
        )
    
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--price_generation_function", type=str, default="constant_prices_generation_function")
    parser.add_argument("--offset_multiplier", type=float, default=0.1)
    parser.add_argument("--off_peak_offset_multiplier", type=float, default=0)
    parser.add_argument("--scale_multiplier", type=float, default=0.1)
    parser.add_argument("--no_save", type=lambda bool_arg: explicit_bool(parser, bool_arg, nonable=False), default=False)
    parser.add_argument("--generate_batch_data", type=lambda bool_arg: explicit_bool(parser, bool_arg, nonable=False), default=False)
    parser.add_argument("--prosumer_noise_scale", type=float, default=0.1)
    parser.add_argument("--generation_noise_scale", type=float, default=0.1)
    parser.add_argument("--num_simulation_steps", type=int, default=1000)
    # Logging Arguments
    parser.add_argument(
        "-w",
        "--wandb",
        help="Whether to upload results to wandb. must have wandb key.",
        type=lambda bool_arg: explicit_bool(parser, bool_arg, nonable=False),
        default=False,
    )
    args = parser.parse_args()
    # Uploading logs to wandb
    if args.wandb:
        wandb.init(project="data-generation", entity="market-maker")
        wandb.run.name = f"({args.offset_multiplier},{args.num_simulation_steps}){args.price_generation_function}-{wandb.run.name}--{args.folder_name}"
        wandb.config.update(args)
        
    run(
        args.folder_name, 
        getattr(price_generation_functions, f"get_{args.price_generation_function}")(**vars(args)),
        args.no_save,
        args.generate_batch_data,
        args.prosumer_noise_scale,
        args.generation_noise_scale,
        args.num_simulation_steps,
    )
