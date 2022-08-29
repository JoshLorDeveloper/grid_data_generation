import numpy as np
from.utils.constants import DAY_LENGTH

def get_constant_prices_generation_function(
    offset_multiplier=0.1,
    **args,
):
    def constant_prices_generation_function(
        day:int, 
        year:int, 
        utility_day_buy_prices:np.ndarray, 
        utility_day_sell_prices:np.ndarray,
    ):
        price_diff = utility_day_buy_prices - utility_day_sell_prices
        
        microgrid_day_buy_prices = utility_day_buy_prices - offset_multiplier*price_diff
        microgrid_day_sell_prices = utility_day_sell_prices + offset_multiplier*price_diff
        return microgrid_day_buy_prices, microgrid_day_sell_prices
    return constant_prices_generation_function


def get_random_prices_generation_function(
    offset_multiplier=0.1,
    scale_multiplier=0.1,
    **args,
):
    def random_prices_generation_function(
        day:int, 
        year:int, 
        utility_day_buy_prices:np.ndarray, 
        utility_day_sell_prices:np.ndarray,
        
    ):
        price_diff = utility_day_buy_prices - utility_day_sell_prices
        offset = offset_multiplier*price_diff
        scale = scale_multiplier*price_diff
        
        mean_microgrid_day_buy_prices = utility_day_buy_prices - offset
        microgrid_day_buy_prices_noise = np.random.normal(0, scale, DAY_LENGTH)
        microgrid_day_buy_prices = mean_microgrid_day_buy_prices + microgrid_day_buy_prices_noise
        
        mean_microgrid_day_sell_prices = utility_day_sell_prices + offset
        microgrid_day_sell_prices_noise = np.random.normal(0, scale, DAY_LENGTH)
        microgrid_day_sell_prices = mean_microgrid_day_sell_prices + microgrid_day_sell_prices_noise
        
        
        return microgrid_day_buy_prices, microgrid_day_sell_prices
    
    return random_prices_generation_function


# randomly 
def get_grouped_random_prices_generation_function(
    offset_multiplier=0.1,
    **args,
):
    def grouped_random_prices_generation_function(
        day:int, 
        year:int, 
        utility_day_buy_prices:np.ndarray, 
        utility_day_sell_prices:np.ndarray,
        
    ):
        price_diff = utility_day_buy_prices - utility_day_sell_prices
        offset = offset_multiplier*price_diff
        direction = np.random.choice([-1, 0, 1])
        
        offset_day_buy_prices = utility_day_buy_prices + offset * (-1 + direction)
        
        offset_day_sell_prices = utility_day_sell_prices + offset * (1 + direction)
        
        
        return offset_day_buy_prices, offset_day_sell_prices
    
    return grouped_random_prices_generation_function
    