import linopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def create_gradient_model(
    flow_demand,
    max_increasing_gradient,
    max_decreasing_gradient,
    size=None,
    relative_minimum=0.0,
    relax_on_startup_full=False,
    relax_on_shutdown_full=False,
    relax_on_startup_to_min_rate=False,
    relax_on_shutdown_to_min_rate=False,
    previous_flow_rate=0.0,
    dt=1.0
):
    """
    Erstellt ein linopy Modell mit Gradientenbeschränkungen.
    
    Mathematische Formulierungen basieren auf den bereitgestellten Gleichungen (3) bis (16).
    Unterstützt sowohl feste Anlagengrößen als auch Investitionsoptimierung.
    """
    m = linopy.Model()
    
    n_timesteps = len(flow_demand)
    time = xr.IndexVariable('time', np.arange(n_timesteps))

    # --- 1. Variablen ---
    is_invest = (size is None)
    
    if is_invest:
        # Investitionsfall: Kapazität ist eine Optimierungsvariable
        size_var = m.add_variables(lower=0, upper=100, name='size')

    else:
        # Feste Größe
        size_var = size
        
    flow = m.add_variables(lower=0, coords=[time], name='flow')
    
    # Statusvariablen (On/Off) und Flankenerkennung (Startup/Shutdown)
    on = m.add_variables(binary=True, coords=[time], name='on')
    startup = m.add_variables(binary=True, coords=[time], name='startup')
    shutdown = m.add_variables(binary=True, coords=[time], name='shutdown')
    
    # Zeitliche Verschiebung für Differenzenbildung

    # 1. Zeitschritt
    if previous_flow_rate>0: previous_on = 1
    else: previous_on = 0

    flow_prev = flow.shift(time=1,fill_value=previous_flow_rate)
    on_prev = on.shift(time=1,fill_value=previous_on)

    # --- 2. Allgemeine Nebenbedingungen ---
    
    # Erkennung für Startup und Shutdown

    m.add_constraints(on - on_prev == startup - shutdown, name='startup1_self')
    m.add_constraints(startup + shutdown <= 1, name='startup2_self')

    # Big-M für Linearisierungen
    M = 10000
    if not is_invest:
        M = size_var * 2
    
    # Leistungsgrenzen (Min/Max Last)
    if is_invest:
        # Linearisierte Bedingungen für den Investitionsfall
        m.add_constraints(flow <= size_var, name='max_flow_invest_limit')
        m.add_constraints(flow <= M * on, name='max_flow_on_off')
        if relative_minimum > 0:
            m.add_constraints(flow >= relative_minimum * size_var - M * (1 - on), name='min_flow_invest_limit')
    else:
        # Einfache Bedingungen für feste Größe
        m.add_constraints(flow <= size_var * on, name='max_flow_fixed')
        m.add_constraints(flow >= relative_minimum * size_var * on, name='min_flow_fixed')

    # --- 3. Gradientenbeschränkungen ---

    # -- STEIGENDER GRADIENT --
    if relax_on_startup_full:
        # Volle Relaxation beim Einschalten (Gradient wird ignoriert)
        m.add_constraints(flow - flow_prev <= max_increasing_gradient * dt + M * startup, name='grad_up_full_relax')
    
    elif relax_on_startup_to_min_rate:
        if not is_invest:
            # Feste Größe: Gleichung (5)
            q_min = relative_minimum * size_var
            val = max(0, q_min - max_increasing_gradient * dt)
            m.add_constraints(flow - flow_prev <= val * startup + max_increasing_gradient * dt, name='grad_up_min_relax_fixed')
        else:
            m.add_constraints(flow - flow_prev <=  max_increasing_gradient * dt + M * startup, name='grad_up_no_startup')
            # Investition: Gleichungen (6) bis (9) mit Hilfsvariable j
            j = m.add_variables(binary=True,  coords=[time], name='j_up')
            # (6) & (7)
            m.add_constraints(flow - flow_prev <= max_increasing_gradient * dt + M * (1 - startup) + M * j, name='eq6')
            m.add_constraints(flow - flow_prev <= relative_minimum * size_var + M * (1 - startup) + M * (1 - j), name='eq7')
            # (8) & (9)
            m.add_constraints(relative_minimum * size_var <= max_increasing_gradient * dt + M * j, name='eq8')
            m.add_constraints(max_increasing_gradient * dt * j <= relative_minimum * size_var, name='eq9')
    else:
        # Standard Gradient ohne Relaxation: Gleichung (3)
        m.add_constraints(flow - flow_prev <= max_increasing_gradient * dt, name='grad_up_standard')

    # -- FALLENDER GRADIENT --
    if relax_on_shutdown_full:
        # Volle Relaxation beim Ausschalten: Gleichung (11)
        m.add_constraints(flow_prev - flow <= max_decreasing_gradient * dt + M * shutdown, name='grad_down_full_relax')

    elif relax_on_shutdown_to_min_rate:
        if not is_invest:
            # Feste Größe: Gleichung (12)
            q_min = relative_minimum * size_var
            val = max(0, q_min - max_decreasing_gradient * dt)
            m.add_constraints(flow_prev - flow <= val * shutdown + max_decreasing_gradient * dt, name='grad_down_min_relax_fixed')
        else:
            m.add_constraints(flow_prev - flow <= max_decreasing_gradient * dt + M * shutdown, name='grad_down_no_shutdown')
            # Investition: Gleichungen (13) bis (16) mit Hilfsvariable k
            k = m.add_variables(binary=True, coords=[time], name='k_down')
            # (13) & (14)
            m.add_constraints(flow_prev - flow <= max_decreasing_gradient * dt + M * (1 - shutdown) + M * k, name='eq13')
            m.add_constraints(flow_prev - flow <= relative_minimum * size_var + M * (1 - shutdown) + M * (1 - k), name='eq14')
            # (15) & (16)
            m.add_constraints(relative_minimum * size_var <= max_decreasing_gradient * dt + M * k, name='eq15')
            m.add_constraints(max_decreasing_gradient * dt * k <= relative_minimum * size_var, name='eq16')
    else:
        # Standard Gradient ohne Relaxation: Gleichung (10)
        m.add_constraints(flow_prev - flow <= max_decreasing_gradient * dt, name='grad_down_standard')

    # --- 4. Bedarfsdeckung & Zielfunktion ---
    
    # Slack-Variable für ungedeckten Bedarf zur Vermeidung von Infeasibility
    p_external = m.add_variables(lower=0, coords=[time], name='p_external')
    m.add_constraints(flow + p_external == xr.DataArray(flow_demand, coords=[time]), name='demand_coverage')

    # Zielfunktion: Priorisierung der Bedarfsdeckung, dann Minimierung von Betrieb und Investition
    obj = (p_external * 1000).sum()        # Sehr hohe Strafe für ungedeckten Bedarf
    obj += flow.sum()             # Kosten für Erzeugung

    if is_invest:
        obj += size_var*0.1

    m.add_objective(obj)
    return m

def plot_results(demand, flow_sol, on_sol, startup_sol, shutdown_sol, title="Optimierungsergebnisse"):
    """Visualisiert die Ergebnisse der Optimierung."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    time = np.arange(len(demand))
    
    # Linke Achse: Leistungen
    ax1.step(time, demand, where='post', label='Bedarf', color='red', linestyle='--', alpha=0.7)
    ax1.step(time, flow_sol, where='post', label='Flow (optimiert)', color='blue', linewidth=2)
    ax1.set_xlabel('Zeitschritt')
    ax1.set_ylabel('Leistung / Flow')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Rechte Achse: Statusvariablen
    ax2 = ax1.twinx()
    ax2.step(time, on_sol, where='post', label='Status (On)', color='green', alpha=0.2)
    ax2.scatter(time[startup_sol > 0.5], startup_sol[startup_sol > 0.5], color='green', marker='^', label='Startup')
    ax2.scatter(time[shutdown_sol > 0.5], shutdown_sol[shutdown_sol > 0.5], color='red', marker='v', label='Shutdown')
    ax2.set_ylabel('Status (Binär)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_test_case(name, demand, **kwargs):
    """Hilfsfunktion zum Ausführen und Anzeigen eines Testfalls."""
    print(f"\n--- Starte {name} ---")
    m = create_gradient_model(demand, **kwargs)

    m.solve(solver_name='gurobi')

    if m.status == 'ok':
        flow_vals = m.variables['flow'].solution.values
        on_vals = m.variables['on'].solution.values
        startup_vals = m.variables['startup'].solution.values
        shutdown_vals = m.variables['shutdown'].solution.values
        
        if 'size' in m.variables:
            print(f"Investierte Größe: {m.variables['size'].solution.values:.2f}")

        print(f"Flow-Werte: {np.round(flow_vals, 2)}")
        plot_results(demand, flow_vals, on_vals, startup_vals, shutdown_vals, title=name)
    else:
        print(f"Fehler: Optimierung beendet mit Status {m.status}")

def test_gradient_scenarios():
    """Definiert und führt verschiedene Test-Szenarien aus."""
    common_demand = [0, 0, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50]
    
    # Test 1: Standard Gradient (langsame Laständerung)
    run_test_case(
        "Test 1: Standard Gradient (keine Relaxation)",
        common_demand,
        max_increasing_gradient=10,
        max_decreasing_gradient=10,
        size=50,
        relative_minimum=0.2
    )
    
    # Test 2a: Relaxation auf Minimallast
    run_test_case(
        "Test 2a: Startup Relaxation auf Min-Rate",
        common_demand,
        max_increasing_gradient=10,
        max_decreasing_gradient=10,
        size=50,
        relative_minimum=0.3,
        relax_on_startup_to_min_rate=True,
        relax_on_shutdown_to_min_rate=True
    )
    
    # Test 2b: Volle Relaxation
    run_test_case(
        "Test 2b: Volle Relaxation (Startup/Shutdown)",
        common_demand,
        max_increasing_gradient=10,
        max_decreasing_gradient=10,
        size=50,
        relative_minimum=0.3,
        relax_on_startup_full=True,
        relax_on_shutdown_full=True
    )
    
    # Test 3: Investitionsoptimierung
    run_test_case(
        "Test 3: Investitionsoptimierung mit Startup-Min-Relaxation",
        common_demand,
        max_increasing_gradient=10,
        max_decreasing_gradient=10,
        size=None,
        relative_minimum=0.3,
        relax_on_startup_to_min_rate=True,
        relax_on_shutdown_full=True,
        previous_flow_rate=20
    )

if __name__ == '__main__':
    test_gradient_scenarios()
