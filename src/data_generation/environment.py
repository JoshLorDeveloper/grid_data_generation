import numpy as np
import pandas as pd
from dataclasses import dataclass
from .real_prosumer import RealProsumer
from .utils.constants import DAY_LENGTH, YEAR_LENGTH, SOLAR_CONSTANT_INSTALLMENT_AREA
from typing import Callable, List, Tuple, Optional

@dataclass
class EnvironmentDataDescriptor:
    time_col_idx: int
    day_of_week_col_idx: int
    price_col_idx: int
    solar_gen_col_idx: int
    temp_col_idx: int
    prosumer_col_idx_list: List[int]
    pv_sizes: Optional[List[int]] # if None generate pv_sizes from buidling size
    battery_nums: List[int]
    prosumer_noise_scale: float
    generation_noise_scale: float
    
    sell_price_function : Callable[[float, int, int], float] = lambda buy_price, day, year : buy_price * 0.6
    
class MockEnvironment:
    
    def __init__(
        self,
        building_data_df: pd.DataFrame,
        building_metadata_df: pd.DataFrame,
        environment_data_descriptor: EnvironmentDataDescriptor, 
    ):
        building_data_df = MockEnvironment.add_time_info(building_data_df, environment_data_descriptor)
        self.prosumer_list, self.hourly_solar_constants = MockEnvironment.create_prosumers(building_data_df, building_metadata_df, environment_data_descriptor)
        self.utility_hourly_buy_prices, self.utility_hourly_sell_prices, self.weekday_dict = MockEnvironment.get_environment_constants(building_data_df, environment_data_descriptor)
    
    def add_time_info(
        building_data_df: pd.DataFrame,
        environment_data_descriptor: EnvironmentDataDescriptor,
    ):
        datetime_col = pd.to_datetime(building_data_df.iloc[:,environment_data_descriptor.time_col_idx]).dt
        building_data_df['hour'] = datetime_col.hour
        building_data_df['day'] = datetime_col.day_of_year
        building_data_df['is_weekday'] = datetime_col.weekday <= 4
         # correct for leap year, as data does not include leap days
        if len(building_data_df[building_data_df['day'] == 59]) == 0:
            building_data_df.loc[building_data_df['day'] > 59, 'day'] -= 1
        return building_data_df
    
    def get_environment_constants(
        building_data_df: pd.DataFrame,
        environment_data_descriptor: EnvironmentDataDescriptor,
    ):

        utility_hourly_buy_prices = building_data_df.pivot(index='day', columns='hour', values=building_data_df.columns[environment_data_descriptor.price_col_idx])
        utility_hourly_sell_prices = utility_hourly_buy_prices * 0.6
        
        weekday_dict = dict(zip(building_data_df['day'], building_data_df['is_weekday']))
        
        return utility_hourly_buy_prices, utility_hourly_sell_prices, weekday_dict

        
    def create_prosumers(
        building_data_df: pd.DataFrame,
        building_metadata_df: pd.DataFrame,
        environment_data_descriptor: EnvironmentDataDescriptor, 
    ) -> Tuple[List[RealProsumer], np.ndarray]:
        prosumer_list : List[RealProsumer] = []
        
        column_labels = building_data_df.columns

        hourly_solar_constants = building_data_df.pivot(index='day', columns='hour', values=building_data_df.columns[environment_data_descriptor.solar_gen_col_idx]).interpolate()
        for prosumer_idx, prosumer_col_idx in enumerate(environment_data_descriptor.prosumer_col_idx_list):
            prosumer_name = column_labels[prosumer_col_idx]
            prosumer_demand = building_data_df.pivot(index='day', columns='hour', values=building_data_df.columns[prosumer_col_idx]).interpolate()
            
            # calculate pv_size
            if environment_data_descriptor.pv_sizes is None:
                prosumer_building_sqm = building_metadata_df.loc[prosumer_name, "sqm"]
                pv_size = prosumer_building_sqm / SOLAR_CONSTANT_INSTALLMENT_AREA * 1/2
            else:
                pv_size = environment_data_descriptor.pv_sizes[prosumer_idx]
            
            
            prosumer = RealProsumer(
                name=prosumer_name,
                yearlongdemand=prosumer_demand,
                yearlonggeneration= hourly_solar_constants,
                battery_num=environment_data_descriptor.battery_nums[prosumer_idx],
                pv_size=pv_size,
                noise_scale=environment_data_descriptor.prosumer_noise_scale,
                generation_noise_scale=environment_data_descriptor.generation_noise_scale,
            )
            prosumer_list.append(prosumer)
        return prosumer_list, hourly_solar_constants

    def get_reward_twoprices(self, for_energy_consumptions, for_day, buy_prices, sell_prices):
        """
        Purpose: Compute reward given grid prices, transactive price set ahead of time, and energy consumption of the participants

        Returns:
            Reward for RL agent (- |net money flow|): in order to get close to market equilibrium
            Reward for profit maximization is amount of money it generates (prices dot demand)
        """
        # external prices to buy from the grid
        buyprice_grid = self.utility_hourly_buy_prices.loc[for_day, :]
        # external prices to sell to the grid
        sellprice_grid = self.utility_hourly_sell_prices.loc[for_day, :]
        
        total_consumption = for_energy_consumptions["Total"]

        # entry true if purchase from microgrid, false if purchase from utility
        # buyprice_grid is the external grid, self.buyprice is prices for "agent"
        # ! forcing agent to strictly be different than the utility price
        # Bool vector containing when prosumers buy from microgrid
        test_buy_from_grid = buy_prices < buyprice_grid
        # Bool vector containing when prosumers sell to microgrid
        test_sell_to_grid = sell_prices > sellprice_grid
        # If buyprice from grid and buyprice from market equal (at all times?, why not only at any time) splits seal of electricity between the market and grid evenly
        # if np.all(buy_prices == buyprice_grid):
        #     # CONSTANT
        #     test_buy_from_grid = np.repeat(0.5, buyprice_grid.shape)
        # if np.all(sell_prices == sellprice_grid):
        #     # CONSTANT
        #     test_sell_to_grid = np.repeat(0.5, sellprice_grid.shape)

        # cost associated with net consumption of entire microgrid (from the perspective of the microgrid)
        # when prosumers are purchasing power from microgrid, if more power is consumed than is being produced on the grid, the grid must purchase from utility
        # alternately if more power is produced than is being consumed on the microgridgrid, the microgrid must sell back to utility
        # ! money_to_utility bug resolved
        money_to_utility = np.dot(
            np.maximum(0, total_consumption * test_buy_from_grid), buyprice_grid
        ) + np.dot(np.minimum(0, total_consumption * test_sell_to_grid), sellprice_grid)

        money_from_prosumers = 0
        grid_money_from_prosumers = 0
        for prosumerName in for_energy_consumptions:
            if prosumerName != "Total":
                # Net money to microgrid from prosumers
                money_from_prosumers += (
                    # money agent gains, selling E energy to prosumer
                    np.dot(
                        np.maximum(0, for_energy_consumptions[prosumerName])
                        * test_buy_from_grid,
                        buy_prices,
                    )
                    + np.dot(
                        np.minimum(0, for_energy_consumptions[prosumerName])
                        * test_sell_to_grid,
                        sell_prices,
                    )
                )  # money agent loses, to sell E energy to prosumer
                # Net money to external grid from prosumers (not including microgrid transactions w utility)
                grid_money_from_prosumers += np.dot(
                    np.maximum(0, for_energy_consumptions[prosumerName])
                    * np.logical_not(test_buy_from_grid),
                    buyprice_grid,
                ) + np.dot(
                    np.minimum(0, for_energy_consumptions[prosumerName])
                    * np.logical_not(test_sell_to_grid),
                    sellprice_grid,
                )

        total_prosumer_cost = (
            grid_money_from_prosumers + money_from_prosumers
        )  # money leaving the prosumers

        # profit maximizing
        total_reward = money_from_prosumers - money_to_utility

        return total_reward #, money_from_prosumers, money_to_utility, total_prosumer_cost