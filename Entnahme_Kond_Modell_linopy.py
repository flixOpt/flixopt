import linopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def plot_pq_diagram(P_points, Q_points, p_demand, q_demand, P_sol=None, Q_sol=None):
    plt.figure(figsize=(10, 7))

    # 1. Polygon plotten (geschlossen)
    x_poly = list(P_points) + [P_points[0]]
    y_poly = list(Q_points) + [Q_points[0]]
    plt.plot(x_poly, y_poly, 'b-o', label='PQ-Diagramm (Turbine)', linewidth=2)
    plt.fill(x_poly, y_poly, alpha=0.2, color='blue')

    # 2. Bedarfspunkte plotten
    plt.scatter(p_demand, q_demand, color='red', marker='x', s=100, label='Bedarf (P, Q)')

    # Zeitschritt-Labels an die Bedarfspunkte
    for t, (p, q) in enumerate(zip(p_demand, q_demand)):
        plt.annotate(
            str(t),
            xy=(p, q),
            xytext=(6, 6),
            textcoords="offset points",
            color="red",
            fontsize=9,
            fontweight="bold",
        )

    # 3. Optimierte Betriebspunkte (falls vorhanden)
    if P_sol is not None and Q_sol is not None:
        plt.scatter(P_sol, Q_sol, color='green', marker='o', s=50, label='Optimierte Betriebspunkte', zorder=5)
        # Verbindungslinien für zeitlichen Verlauf (optional)
        plt.plot(P_sol, Q_sol, 'g--', alpha=0.5)

        # Zeitschritt-Labels an die Arbeitspunkte + Zuordnungslinien Bedarf <-> Arbeitspunkt
        for t, (p_op, q_op) in enumerate(zip(P_sol, Q_sol)):
            plt.annotate(
                str(t),
                xy=(p_op, q_op),
                xytext=(6, -10),
                textcoords="offset points",
                color="green",
                fontsize=9,
                fontweight="bold",
            )
            plt.plot(
                [p_demand[t], p_op],
                [q_demand[t], q_op],
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.6,
                zorder=1,
            )

    plt.xlabel('Elektrische Leistung P (x)')
    plt.ylabel('Thermische Leistung Q (y)')
    plt.title('P,Q-Diagramm der Entnahmekondensationsturbine')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def create_polygon_model(P_points, Q_points, p_demand, q_demand, costs=1.0, a=0.0, b=0.0, c=0.0, costs_import=10.0, costs_export=10):
    # 1. Modell initialisieren
    m = linopy.Model()
    
    # Dimensionen
    n_points = len(P_points)
    n_timesteps = len(p_demand)
    
    point_indices = np.arange(n_points)
    time_indices = np.arange(n_timesteps)

    time = xr.IndexVariable('time', time_indices)
    point = xr.IndexVariable('point', point_indices)


    # 2. Variablen definieren
    P = m.add_variables(lower=0, coords=[time], name='x')
    Q = m.add_variables(lower=0, coords=[time], name='y')
    is_on = m.add_variables(binary=True, coords=[time], name='is_on')
    p_external = m.add_variables(lower=0, upper=100, coords=[time], name='p_external')
    p_export = m.add_variables(lower=0, upper=100, coords=[time], name='p_export')
    
    lambdas = m.add_variables(lower=0, upper=1, 
                               coords=[time, point],
                               name='lambda')
    
    # 3. Konstanten/Parameter als DataArrays
    P_i = xr.DataArray(P_points, coords=[point])
    Q_i = xr.DataArray(Q_points, coords=[point])

    # 4. Bedingungen hinzufügen
    # P und Q als gewichtete Summe der Eckpunkte
    m.add_constraints((P_i * lambdas).sum('point') == P, name='eq_x')
    m.add_constraints((Q_i * lambdas).sum('point') == Q, name='eq_y')
    
    # Die Summe der Lambdas muss dem Einschaltzustand entsprechen
    m.add_constraints(lambdas.sum('point') == is_on, name='sum_lambda')

    # BEMERKUNG: Modellierung des Zustands "AUS" durch Binärvariable is_on

    # ZUR BEISPIELHAFTEN ANWENDUNG
    # Bedarfswerte
    p_req = xr.DataArray(p_demand, coords=[time])
    q_req = xr.DataArray(q_demand, coords=[time])
    # EXAKTE WÄRMEERZEUGUNG
    # y = q_req
    m.add_constraints(Q == q_req, name='demand_q')
    # STROMBEDARF
    # x + p_external - p_export == p_req
    m.add_constraints(P + p_external - p_export == p_req, name='demand_p')
    # Zielfunktion (Fixkosten a fallen nur an, wenn die Anlage AN ist)
    objective = (costs * (a * is_on + b * P + c * Q)).sum() + (costs_import * p_external).sum() + (costs_export * p_export).sum()
    m.add_objective(objective, sense='min')
    
    return m

if __name__ == '__main__':
    # Eckpunkte Turbine (P, Q) - Diagramm
    # Beachte: (35,0) ist der kleinste P-Wert bei Q=0.
    P_turbine = [30, 50, 90, 95, 70]
    Q_turbine = [10, 100, 100, 50, 10]

    # gegebene Testwerte zum Betrieb
    p_demand = [35, 60, 60, 80, 90, 70, 20, 50, 95, 95, 0 ]
    q_demand = [40, 50, 80, 90, 100, 100, 80, 10, 80, 20, 0]
    
    costs_fuel = 1.0

    # Frischdampfbedarf:
    # m_FD = a1 + b1 * P + c1 * Q
    # beliebige Annahme:
    a1 = 50
    b1 = 2
    c1 = 1

    # Brennstoffbedarf
    # Q_BS = a2 + b2 * m_FD
    # beliebige Annahme:
    a2 = 10
    b2 = 1

    # Gesamt
    # Q_BS = a2+a1*b2 + b1*b2 * P + c1*b2 * Q
    a_val = a2 + a1 * b2
    b_val = b1 * b2
    c_val = c1 * b2


    model = create_polygon_model(P_turbine, Q_turbine, p_demand, q_demand, 
                                 costs=costs_fuel, a=a_val, b=b_val, c=c_val, 
                                 costs_import=10, costs_export=10)
    
    model.solve(solver_name='gurobi')
    
    if model.status == 'ok':
        n_t = len(p_demand)
        total_costs = model.objective.value
        print(f"Optimierung erfolgreich. Gesamtkosten: {total_costs:.2f}")
        
        P_sol = model.variables['x'].solution
        Q_sol = model.variables['y'].solution
        is_on_sol = model.variables['is_on'].solution
        p_ext_sol = model.variables['p_external'].solution
        p_exp_sol = model.variables['p_export'].solution

        print("t | P_Bedarf | P_Turbine | P_Extern | P_Export | Q_Turbine | Q_Bedarf | Status")
        for t in range(n_t):
            status = "AN" if is_on_sol[t] > 0.5 else "AUS"
            print(
                f"{t} | {p_demand[t]:8.2f} | {P_sol[t]:9.2f} | {p_ext_sol[t]:8.2f} | {p_exp_sol[t]:8.2f} | {Q_sol[t]:9.2f} | {q_demand[t]:8.2f} | {status}"
            )

        # Grafik anzeigen
        plot_pq_diagram(P_turbine, Q_turbine, p_demand, q_demand, P_sol.values, Q_sol.values)
    else:
        print(f"Optimierung fehlgeschlagen mit Status: {model.status}")
        # Grafik trotzdem anzeigen (ohne Lösung)
        plot_pq_diagram(P_turbine, Q_turbine, p_demand, q_demand)
