import wandb
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict

from .convert_batch import BatchWriter

from .utils.constants import DAY_LENGTH, YEAR_LENGTH
from .environment import MockEnvironment


@dataclass
class SimulationConfig:
    num_simulation_steps: int
    day_start: int
    year_start: int
    prices_generation_function: Callable[[int, int, np.ndarray, np.ndarray], np.ndarray]

def get_observation(daily_energy_consumption, daily_generation, daily_buy_prices):
        """Get today's observation."""

        return np.concatenate(
            (daily_energy_consumption, daily_generation, daily_buy_prices)
        ).astype(np.float32)

def simulate(mock_environment: MockEnvironment, simulation_config: SimulationConfig, write_data: Callable, batch_writer: BatchWriter = None) -> Dict[str, pd.DataFrame]:
    """
    Simulate mock environment with simulation config
    
    :param mock_environment: Environement to simulate
    :param simulation_config: Config to use in simulation
    :return: dataframe containing simulation data
    """
    
    simulation_data_by_prosumers : Dict[str, List[Dict]] = {}
    
    for simulation_step_idx in range(simulation_config.num_simulation_steps):
        
        simulation_row_by_prosumers : Dict[str, Dict] = {prosumer.name : [] for prosumer in mock_environment.prosumer_list}

        simulate_day = (((simulation_config.day_start - 1) + simulation_step_idx) % YEAR_LENGTH) + 1
        simulate_year = simulation_config.year_start + (simulation_config.day_start + simulation_step_idx - 1) // YEAR_LENGTH

        utility_day_buy_prices = mock_environment.utility_day_buy_prices_list[simulate_day] 
        utility_day_sell_prices = mock_environment.utility_day_sell_prices_list[simulate_day]
        
        microgrid_buy_prices, microgrid_sell_prices = simulation_config.prices_generation_function(
            simulate_day, 
            simulate_year, 
            utility_day_buy_prices, 
            utility_day_sell_prices,
        )
                
        general_step_data = {
            "step": simulation_step_idx,
            "year" : simulate_year,
            "day" : simulate_day,
        }
        
        buy_price_step_data = {}
        sell_price_step_data = {}
        for hour in range(DAY_LENGTH):
            buy_price_step_data[f"agent_buy_{hour}"] = microgrid_buy_prices[hour]
            sell_price_step_data[f"agent_sell_{hour}"] = microgrid_sell_prices[hour]
        
        # Calculate prosumer demand
        prosumer_demand_dict = {"Total": np.zeros(DAY_LENGTH)}
        for prosumer in mock_environment.prosumer_list:
            prosumer_name = prosumer.name
            
            simulated_demand = prosumer.get_real_response_twoprices(simulate_day, microgrid_buy_prices, microgrid_sell_prices, simulate_year)
            prosumer_demand_dict[prosumer_name] = simulated_demand
            
            # record step data for reporting
            prosumer_step_data = {}
            for hour in range(DAY_LENGTH):
                prosumer_step_data[f"prosumer_response_{hour}"] = simulated_demand[hour]
            simulation_row_by_prosumers[prosumer_name] = {
                **buy_price_step_data,
                **sell_price_step_data,
                **prosumer_step_data,
                "prosumer_name": prosumer_name,
                **general_step_data,
                "battery_num": prosumer.battery_num,
                "pv_size": prosumer.pv_size,
            }
            
            prosumer_demand_dict["Total"] = prosumer_demand_dict["Total"] + simulated_demand
                    
        # Calculate step reward
        step_reward = mock_environment.get_reward_twoprices(
            prosumer_demand_dict,
            simulate_day,
            microgrid_buy_prices,
            microgrid_sell_prices,
        )
        if step_reward is np.nan or step_reward is None:
            print(f"reward calculation failed on day {simulate_day}")
        elif wandb.run is not None:
            wandb.log({"step_reward": step_reward})
        for prosumer_name, simulation_row in simulation_row_by_prosumers.items():
            simulation_row["reward"] = step_reward
            write_data(simulation_row, prosumer_name, simulation_step_idx)
            
            # add row to cumulative dataframe
            simulation_data_by_prosumers.setdefault(prosumer_name, []).append(simulation_row)
        
        if batch_writer is not None and simulation_step_idx > 1:
            batch_writer.write_batch(
                simulation_step_idx, 
                np.concatenate([microgrid_buy_prices, microgrid_sell_prices]),
                get_observation(
                    prosumer_demand_dict["Total"],
                    mock_environment.hourly_solar_constants[
                        (simulate_day * DAY_LENGTH) : (simulate_day * DAY_LENGTH) + DAY_LENGTH
                    ],
                    utility_day_buy_prices,  
                ),
                step_reward,
            )
            
    
    simulation_data_df_by_prosumers = {}
    for prosumer_name, prosumer_data in simulation_data_by_prosumers.items():
            simulation_data_df_by_prosumers[prosumer_name] = pd.DataFrame(prosumer_data)
    return simulation_data_df_by_prosumers

    