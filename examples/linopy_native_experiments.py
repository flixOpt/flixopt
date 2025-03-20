from typing import Dict, List, Literal, Optional, Tuple

import linopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr


class SystemModel(linopy.Model):
    def __init__(
            self,
            timesteps: pd.DatetimeIndex,
            hours_of_last_step: Optional[float] = None,
            periods: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        timesteps : pd.DatetimeIndex
            The timesteps of the model.
        hours_of_last_step : Optional[float], optional
            The duration of the last time step. Uses the last time interval if not specified
        periods : Optional[List[int]], optional
            The periods of the model. Every period has the same timesteps.
            Usually years are used as periods.
        """
        super().__init__(force_dim_names=True)
        self.timesteps = timesteps
        self.timesteps.name = 'time'
        self.periods = pd.Index(periods, name='period') if periods is not None else None

        if hours_of_last_step:
            last_date = pd.DatetimeIndex([self.timesteps[-1] + pd.to_timedelta(hours_of_last_step, 'h')])
        else:
            last_date = pd.DatetimeIndex([self.timesteps[-1] + (self.timesteps[-1] - self.timesteps[-2])])
        self.timesteps_extra = self.timesteps.append(last_date)
        self.timesteps_extra.name = 'time'
        hours_per_step = self.timesteps_extra.to_series().diff()[1:].values / pd.to_timedelta(1, 'h')
        self.hours_per_step = xr.DataArray(
            data=np.tile(hours_per_step, (len(self.periods), 1)) if self.periods is not None else hours_per_step,
            coords=self.coords,
            name='hours_per_step'
        )

    @property
    def snapshots(self):
        return xr.Dataset(
            coords={'period': list(self.periods), 'time': list(self.timesteps)} if self.periods is not None else {'time': list(self.timesteps)},
        )

    @property
    def coords(self):
        return self.snapshots.coords

    @property
    def time_variables(self, filter_by: Optional[Literal['binary', 'continous', 'integer']] = None):
        if filter_by is None:
            all_variables = super().variables
        elif filter_by == 'binary':
            all_variables = super().binaries
        elif filter_by == 'integer':
            all_variables = super().integers
        elif filter_by == 'continous':
            all_variables = super().continuous
        else:
            raise ValueError(f'Invalid filter_by "{filter_by}", must be one of "binary", "continous", "integer"')
        return all_variables[[name for name in all_variables if 'time' in all_variables[name].dims]]

    @property
    def index_shape(self) -> Tuple[int, int]:
        return len(self.periods) if self.periods is not None else 1, len(self.timesteps)


m = SystemModel(pd.date_range(start='2025-01-01', end='2025-01-08', freq='h', name='time'), periods=[2025, 2030])

rng = np.random.default_rng(seed=42)
random_array = rng.random(m.index_shape)

total = pd.Index(range(1), name='total')

x = m.add_variables(lower=0, coords=m.coords, name="x")  # x is a variable for every timestep and period
y = m.add_variables(lower=0, coords=m.coords, name="y")  # y is a variable for every timestep
z = m.add_variables(lower=0, name="z")  # z is a scalar variable

factor = xr.DataArray(random_array * 10, coords=m.coords)

con1 = m.add_constraints(3 * x + 7 * y >= 10 * factor, name="con1")
con2 = m.add_constraints(5 * x + 2 * y >= 3 * factor, name="con2")
con3 = m.add_constraints(z >= 3, name="con3")

# Complex constraint
con_weekly = m.add_constraints(
    (3 * y).where(m.snapshots['period'] == 2025).sum() <= 20, name="con_per_period")

# Size constraint, using a scalar variable
s = m.add_variables(lower=0, name="s")
con_size = m.add_constraints(x <= s, name="con_size")

# Size constraint, using a period variable
s_per_period = m.add_variables(lower=0, coords=(m.periods,), name="s_per_period")
con_size_per_period = m.add_constraints(x <= s_per_period, name="con_size_per_period")

# Constraint the total over the month of April to be 11000
total_of_kw1 = m.add_variables(upper=11000, name="total_of_KW1")
con_per_month = m.add_constraints(
    (m.hours_per_step * x).where(x.coords['time'].dt.week == 1).sum() <= total_of_kw1,
    name="con_total_per_month"
)


##### Storage #####
# Add a variable thats one step longer (charge state)
charge_state = m.add_variables(lower=100, coords=(m.periods, m.timesteps_extra), name="charge_state")
flow_storage = m.add_variables(lower=-100, upper=100, coords=m.coords, name="flow_netto_charging")

con_storage = m.add_constraints(
    charge_state.isel(time=slice(1, None)) == charge_state.isel(time=slice(None, -1)) * 0.99 + flow_storage,
    name="con_storage"
)

# Start every period with 1000 kWh
con_storage_start = m.add_constraints(
    charge_state.isel(time=0) == xr.DataArray([1000, 2000], coords=(m.periods,)),
    name="con_storage_start"
)
# Start = End for every period
start_is_end = False
if start_is_end:
    con_storage_start_end = m.add_constraints(
        charge_state.isel(time=0) == charge_state.isel(time=-1),
        name="con_storage_start_end"
    )
m.add_constraints(charge_state.isel(period=0, time=40) == 6*charge_state.isel(period=1, time=40), name="couple_periods")
m.add_objective((x + 2 * y).sum() + z)

m.solve()

# --- Plotting ---
# plot all results directly
m.time_variables.solution.to_dataframe().plot(grid=True, ylabel="Optimal Value", title="All Time variables",)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Order the dataframe by period and time
df = m.time_variables.solution.to_dataframe()

# Plotting per period is easy
fig = px.line(charge_state.solution.to_dataframe().reset_index(), x="time", y="solution", color="period", title="Charge State in MWh")
fig.show()

# Plotting the whole is even easier
fig= charge_state.solution.plot()
fig.figure.show()


