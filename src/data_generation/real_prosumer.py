import contextlib
import numpy as np
from .utils.constants import DAY_LENGTH, YEAR_LENGTH
from scipy.optimize import minimize


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
        
class RealProsumer:
    
    def __init__(
        self,
        name,
        yearlongdemand,
        yearlonggeneration,
        battery_num=0,
        pv_size=0,
        noise_scale=0.1,
        generation_noise_scale=0.1
    ):
        self.name = name.replace(" (kWh)", "")
        self.yearlongdemand = yearlongdemand
        self.yearlonggeneration = yearlonggeneration
        max_gen_by_hour = np.zeros(DAY_LENGTH)
        for day_number in range(0, YEAR_LENGTH):
            if day_number in self.yearlonggeneration.index:
                gen = pv_size * self.yearlonggeneration.loc[day_number, :]
                max_gen_by_hour = np.maximum(max_gen_by_hour, gen.fillna(0).values)
        self.maxgeneration = max_gen_by_hour
        self.battery_num = battery_num
        self.pv_size = pv_size
        self.capacity = 13.5  # kW-hour
        self.batterycyclecost = 273 / 2800  # per unit capacity
        self.eta = 0.95  # battery one way efficiency
        self.c_rate = 0.35
        self.battery_discharged_capacity = 121
        self.battery_discharged_times = 5656
        self.noise_scale=noise_scale
        self.generation_noise_scale=generation_noise_scale
        
    def get_real_response_twoprices(self, day, buyprices, sellprices, year = None, num_optim_steps=10000):
        """
        Determines the net load of the prosumer on a specific day, in response to energy prices

        Args:
                day: day of the year. Allowed values: [0,365)
                buyprices: DAY_LENGTH hour price vector, supplied as an np.array
                sellprices: DAY_LENGTH hour price vector, supplied as an np.array
        """

        load = self.yearlongdemand.loc[day, :]
        gen = self.pv_size * self.yearlonggeneration.loc[day, :]
        
        eta = self.eta
        capacity = self.capacity
        battery_num = self.battery_num
        c_rate = self.c_rate
        Ltri = np.tril(np.ones((DAY_LENGTH, DAY_LENGTH)))

        def dailyobjective(x):
            net = load - gen + (-eta + 1 / eta) * abs(x) / 2 + (eta + 1 / eta) * x / 2
            return np.sum(np.maximum(net, 0) * buyprices) + np.sum(
                np.minimum(net, 0) * sellprices
            )

        def hourly_con_charge_max(x):
            # Shouldn't charge or discharge too fast
            return c_rate * capacity * battery_num - x

        def hourly_con_charge_min(x):
            # Shouldn't charge or discharge too fast
            return c_rate * capacity * battery_num + x

        def hourly_con_cap_max(x):
            # x should respect the initial state of charge
            return capacity * battery_num - np.matmul(Ltri, x)

        def hourly_con_cap_min(x):
            # x should respect the initial state of charge
            return np.matmul(Ltri, x)

        con1_hourly = {"type": "ineq", "fun": hourly_con_charge_min}
        con2_hourly = {"type": "ineq", "fun": hourly_con_charge_max}
        con3_hourly = {"type": "ineq", "fun": hourly_con_cap_min}
        con4_hourly = {"type": "ineq", "fun": hourly_con_cap_max}
        cons_hourly = (con1_hourly, con2_hourly, con3_hourly, con4_hourly)

        x0 = [battery_num * capacity] * DAY_LENGTH
        # x0 = [0]*DAY_LENGTH

        sol = minimize(
            dailyobjective,
            x0,
            constraints=cons_hourly,
            method="SLSQP",
            options={"maxiter": num_optim_steps},
        )

        if not sol.success:
            net = load - gen
            # import pdb; pdb.set_trace()
            x = sol["x"]
            battery_discharged_capacity = np.sum(np.abs(x))
            battery_discharged_times = np.sum(np.abs(x) > 0.1)
            # v1: changing this so that it takes the same behavior if the solution is reached or not -- still dependent on the battery's behavior.
            net = load - gen + (-eta + 1 / eta) * abs(x) / 2 + (eta + 1 / eta) * x / 2
            upper_bound = (
                load - gen + np.ones(DAY_LENGTH) * capacity * battery_num * c_rate
            )
            lower_bound = (
                load - gen - np.ones(DAY_LENGTH) * capacity * battery_num * c_rate
            )
            net = np.minimum(net, upper_bound)  # upper bound
            net = np.maximum(net, lower_bound)  # lower bound
        else:
            x = sol["x"]
            # import pdb; pdb.set_trace()
            battery_discharged_capacity = np.sum(np.abs(x))
            battery_discharged_times = np.sum(np.abs(x) > 0.1)
            net = load - gen + (-eta + 1 / eta) * abs(x) / 2 + (eta + 1 / eta) * x / 2
            upper_bound = (
                load - gen + np.ones(DAY_LENGTH) * capacity * battery_num * c_rate
            )
            lower_bound = (
                load - gen - np.ones(DAY_LENGTH) * capacity * battery_num * c_rate
            )
            net = np.minimum(net, upper_bound)  # upper bound
            net = np.maximum(net, lower_bound)  # lower bound
        # sol['x'] = x
        # sol['fun'] = dailyobjective(x)

        calculated_demand = np.array(net)
        noise = np.random.normal(loc = 0, scale = np.abs(calculated_demand * self.noise_scale), size = DAY_LENGTH)
        
        with temp_seed(int(f"{day}{year}")):
            generation_noise = np.random.normal(loc = 0, scale = np.abs(self.maxgeneration * self.generation_noise_scale), size = DAY_LENGTH) 
        
        simulated_demand = calculated_demand + noise + generation_noise

        return simulated_demand