import numpy as np
import datetime
from.utils.constants import DAY_LENGTH

def get_baseline_prices_generation_function(
    **args,
):
    def baseline_prices_generation_function(
        day:int, 
        year:int, 
        utility_hourly_buy_price:np.ndarray, 
        utility_hourly_sell_price:np.ndarray,
    ):
        return utility_hourly_buy_price, utility_hourly_sell_price
        
    return baseline_prices_generation_function

# always uses offset
def get_constant_prices_generation_function(
    offset_multiplier=0.1,
    **args,
):
    def constant_prices_generation_function(
        day:int, 
        year:int, 
        utility_hourly_buy_price:np.ndarray, 
        utility_hourly_sell_price:np.ndarray,
    ):
        price_diff = utility_hourly_buy_price - utility_hourly_sell_price
        
        microgrid_day_buy_prices = utility_hourly_buy_price - offset_multiplier*price_diff
        microgrid_day_sell_prices = utility_hourly_sell_price + offset_multiplier*price_diff
        return microgrid_day_buy_prices, microgrid_day_sell_prices
    return constant_prices_generation_function

# uses offset on weekdays
def get_constant_peak_day_prices_generation_function(
    offset_multiplier=0.1,
    off_peak_offset_multiplier=0,
    **args,
):
    def constant_peak_day_prices_generation_function(
        day:int, 
        year:int, 
        utility_hourly_buy_price:np.ndarray, 
        utility_hourly_sell_price:np.ndarray,
    ):
        date_object = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
        if year%4 == 0 and day >= 59:
            date_object+= datetime.timedelta(1)
        weekday = date_object.weekday()
        if weekday > 4:
            # weekend
            price_diff = utility_hourly_buy_price - utility_hourly_sell_price
        
            microgrid_day_buy_prices = utility_hourly_buy_price - off_peak_offset_multiplier*price_diff
            microgrid_day_sell_prices = utility_hourly_sell_price + off_peak_offset_multiplier*price_diff
        else:
            # weekday
            price_diff = utility_hourly_buy_price - utility_hourly_sell_price
            
            microgrid_day_buy_prices = utility_hourly_buy_price - offset_multiplier*price_diff
            microgrid_day_sell_prices = utility_hourly_sell_price + offset_multiplier*price_diff
        return microgrid_day_buy_prices, microgrid_day_sell_prices
    return constant_peak_day_prices_generation_function

# uses offset on peak hours
def get_constant_peak_hour_prices_generation_function(
    offset_multiplier=0.1,
    off_peak_offset_multiplier=0,
    **args,
):
    def constant_peak_hour_prices_generation_function(
        day:int, 
        year:int, 
        utility_hourly_buy_price:np.ndarray, 
        utility_hourly_sell_price:np.ndarray,
    ):
        date_object = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
        if year%4 == 0 and day >= 59:
            date_object+= datetime.timedelta(1)
        weekday = date_object.weekday()
        if weekday > 4:
            # weekend
            price_diff = utility_hourly_buy_price - utility_hourly_sell_price
        
            microgrid_day_buy_prices = utility_hourly_buy_price - off_peak_offset_multiplier*price_diff
            microgrid_day_sell_prices = utility_hourly_sell_price + off_peak_offset_multiplier*price_diff
        else:
            # weekday
            price_diff = utility_hourly_buy_price - utility_hourly_sell_price

            # peak hour mask
            PEAK_HOURS = [15, 16, 17, 18, 19]
            mask = np.isin(np.arange(24), PEAK_HOURS)
            
            microgrid_day_buy_prices = utility_hourly_buy_price - price_diff*(offset_multiplier*mask + off_peak_offset_multiplier*(~mask))
            microgrid_day_sell_prices = utility_hourly_sell_price + price_diff*(offset_multiplier*mask + off_peak_offset_multiplier*(~mask))

        return microgrid_day_buy_prices, microgrid_day_sell_prices
    return constant_peak_hour_prices_generation_function




def get_random_prices_generation_function(
    offset_multiplier=0.1,
    scale_multiplier=0.1,
    **args,
):
    def random_prices_generation_function(
        day:int, 
        year:int, 
        utility_hourly_buy_price:np.ndarray, 
        utility_hourly_sell_price:np.ndarray,
        
    ):
        price_diff = utility_hourly_buy_price - utility_hourly_sell_price
        offset = offset_multiplier*price_diff
        scale = scale_multiplier*price_diff
        
        mean_microgrid_day_buy_prices = utility_hourly_buy_price - offset
        microgrid_day_buy_prices_noise = np.random.normal(0, scale, DAY_LENGTH)
        microgrid_day_buy_prices = mean_microgrid_day_buy_prices + microgrid_day_buy_prices_noise
        
        mean_microgrid_day_sell_prices = utility_hourly_sell_price + offset
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
        utility_hourly_buy_price:np.ndarray, 
        utility_hourly_sell_price:np.ndarray,
        
    ):
        price_diff = utility_hourly_buy_price - utility_hourly_sell_price
        offset = offset_multiplier*price_diff
        direction = np.random.choice([-1, 0, 1])
        
        offset_day_buy_prices = utility_hourly_buy_price + offset * (-1 + direction)
        
        offset_day_sell_prices = utility_hourly_sell_price + offset * (1 + direction)
        
        
        return offset_day_buy_prices, offset_day_sell_prices
    
    return grouped_random_prices_generation_function
    