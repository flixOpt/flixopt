"""
Example showing how cumulative tracking could work (conceptual - not yet implemented)

This demonstrates the power of cumulative variables vs. simple totals.
"""

import numpy as np
import pandas as pd

import flixopt as fx

print('=' * 80)
print('CUMULATIVE TRACKING - Conceptual Example')
print('=' * 80)

# Create time index for one year with hourly resolution
time = pd.date_range('2025-01-01', '2025-12-31 23:00', freq='h')

# ============================================================================
# CURRENT BEHAVIOR (Total only)
# ============================================================================
print('\n1. CURRENT BEHAVIOR - Only total constraints')
print('-' * 80)

# Current API only allows limiting the TOTAL over entire horizon
current_params = fx.StatusParameters(
    effects_per_startup={'cost': 1000},
    min_uptime=4,
    startup_limit=50,  # Maximum 50 startups over ENTIRE year
    active_hours_max=4000,  # Maximum 4000 hours total
)

print('✓ Can limit total startups: 50 per year')
print('✓ Can limit total active hours: 4000 per year')
print('✗ CANNOT limit startups per month/quarter')
print('✗ CANNOT ensure progressive usage (e.g., 1000h by Q1, 2500h by Q2)')
print('✗ CANNOT limit startup rate in specific periods')

# ============================================================================
# PROPOSED BEHAVIOR (Cumulative tracking)
# ============================================================================
print('\n' + '=' * 80)
print('2. PROPOSED BEHAVIOR - Cumulative constraints over time')
print('-' * 80)

# Create progressive limits for different use cases
quarterly_ends = pd.DatetimeIndex(['2025-03-31', '2025-06-30', '2025-09-30', '2025-12-31'])
monthly_checkpoints = pd.date_range('2025-01-31', '2025-12-31', freq='ME')

# Example 1: Progressive startup limits (for warranty/maintenance)
# "Can do 10 startups in Q1, 25 cumulative by Q2, 40 by Q3, 50 by Q4"
cumulative_startup_limits = pd.Series([10, 25, 40, 50], index=quarterly_ends, name='cumulative_startup_limit')

# Example 2: Minimum energy delivery milestones (for contracts)
# "Must deliver at least X MWh by end of each quarter"
cumulative_delivery_mins = pd.Series(
    [1000, 2500, 4000, 6000],  # MWh
    index=quarterly_ends,
    name='cumulative_delivery_min',
)

# Example 3: Monthly CO2 budget tracking
# Progressively increasing CO2 budget (e.g., 1000 tons/month)
cumulative_co2_budget = pd.Series(
    np.arange(1, 13) * 1000,  # 1000, 2000, ..., 12000 tons
    index=monthly_checkpoints,
    name='cumulative_CO2_max',
)

print('\n📊 Example Cumulative Constraints:')
print('\nStartup Limits (Progressive):')
for date, limit in cumulative_startup_limits.items():
    print(f'  By {date.strftime("%Y-%m-%d")}: max {limit:2d} startups cumulative')

print('\nMinimum Energy Delivery:')
for date, min_mwh in cumulative_delivery_mins.items():
    print(f'  By {date.strftime("%Y-%m-%d")}: min {min_mwh:5.0f} MWh delivered')

print('\nCO2 Budget (Monthly checkpoints):')
for i, budget in enumerate(cumulative_co2_budget.values(), 1):
    print(f'  By end of month {i:2d}: max {budget:6.0f} tons CO2')

# ============================================================================
# CONCEPTUAL API (How it could look)
# ============================================================================
print('\n' + '=' * 80)
print('3. CONCEPTUAL API - How this could be used')
print('-' * 80)

print("""
# Proposed API extension to StatusParameters:
proposed_params = fx.StatusParameters(
    effects_per_startup={'cost': 1000},
    min_uptime=4,

    # OLD: Only total limit
    startup_limit=50,  # Total over entire period

    # NEW: Cumulative limits over time
    cumulative_startup_limit=cumulative_startup_limits,  # Progressive limits
    # Creates variable: cumulative_startups[t] = sum(startup[0:t+1])
    # Adds constraint: cumulative_startups[checkpoint_t] <= limit[checkpoint_t]
)

# For Flows - energy delivery contracts:
contract_flow = fx.Flow(
    label='contracted_supply',
    bus='electricity',
    size=100,

    # OLD: Only total flow hours per period
    flow_hours_min=6000,  # At least 6000 MWh total

    # NEW: Cumulative milestones
    cumulative_flow_hours_min=cumulative_delivery_mins,  # Progressive minimums
    # Creates variable: cumulative_flow_hours[t] = sum(flow_rate[0:t+1] * dt)
    # Adds constraint: cumulative_flow_hours[milestone_t] >= target[milestone_t]
)

# For Effects - budget and emissions tracking:
CO2_effect = fx.Effect(
    'CO2',
    unit='tons',

    # NEW: Cumulative maximum over time
    cumulative_maximum=cumulative_co2_budget,  # Monthly checkpoints
    # Creates variable: cumulative_CO2[t] = sum(CO2_emissions[0:t+1])
    # Adds constraint: cumulative_CO2[checkpoint_t] <= budget[checkpoint_t]
)
""")

# ============================================================================
# BENEFITS
# ============================================================================
print('\n' + '=' * 80)
print('4. BENEFITS OF CUMULATIVE TRACKING')
print('-' * 80)

print("""
✓ Progressive Limits
  - Warranty compliance (max X starts in Y period)
  - Maintenance scheduling (limit cycling before maintenance)

✓ Contract Compliance
  - Staged delivery requirements
  - Take-or-pay minimum deliveries
  - Progressive capacity obligations

✓ Budget Management
  - Monthly/quarterly spending limits
  - Emissions allowance tracking
  - Fuel quota management

✓ Rolling Window Constraints
  - "Max 10 starts per 30 days" (at any point)
  - "Max 1000 tons CO2 per month" (rolling)

✓ Rate Limiting
  - Control speed of resource consumption
  - Prevent front-loading or back-loading

✓ Flexible Modeling
  - User defines checkpoints and limits
  - Works with any time granularity
  - Compatible with periods and scenarios
""")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print('\n' + '=' * 80)
print('5. CURRENT vs. PROPOSED')
print('-' * 80)

print(f'\n{"Capability":<45} {"Current":<15} {"Proposed":<15}')
print('-' * 80)
print(f'{"Limit total startups per period":<45} {"✓":<15} {"✓":<15}')
print(f'{"Limit startups in specific sub-periods":<45} {"✗":<15} {"✓":<15}')
print(f'{"Progressive startup limits (Q1, Q2, ...)":<45} {"✗":<15} {"✓":<15}')
print(f'{"Rolling window constraints":<45} {"✗":<15} {"✓":<15}')
print(f'{"Staged delivery milestones":<45} {"✗":<15} {"✓":<15}')
print(f'{"Monthly budget tracking":<45} {"✗":<15} {"✓":<15}')
print(f'{"Rate limiting":<45} {"✗":<15} {"✓":<15}')
print(f'{"Contract compliance (min delivery by date)":<45} {"✗":<15} {"✓":<15}')

print('\n' + '=' * 80)
print('CONCLUSION')
print('-' * 80)
print("""
Cumulative tracking (similar to storage charge state) would enable a much richer
set of optimization problems:
- Real-world maintenance schedules
- Contract compliance
- Budget and emissions tracking
- Progressive constraints

The pattern already exists in storage - extending it to other variables would be
natural and powerful!
""")
print('=' * 80)
