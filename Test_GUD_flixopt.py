import numpy as np
import pandas as pd
import flixopt as fx

if __name__ == '__main__':
    # --- 1. Zeitreihen und Basisdaten definieren ---
    # Beispielhafter Strombedarf über 6 Stunden
    elec_demand_per_h = np.array([10, 20, 30, 40, 0, 50])
    # Erstellung eines Zeitindex für die Simulation
    timesteps = pd.date_range('2020-01-01', periods=len(elec_demand_per_h), freq='h', name='time')

    # Initialisierung des Energiesystems (FlowSystem)
    flow_system = fx.FlowSystem(timesteps=timesteps)

    # --- 2. Effekte (z.B. Kosten, CO2) definieren ---
    # Standard-Effekt 'costs' für die Zielfunktion des Modells
    costs = fx.Effect(
        label='costs',
        unit='€',
        description='Kosten',
        is_standard=True,
        is_objective=True,
    )

    # --- 3. Busse (Knotenpunkte für Energieflüsse) ---
    strom_bus = fx.Bus(label='Strom')
    waerme_bus = fx.Bus(label='Wärme')
    gas_bus = fx.Bus(label='Gas')

    # --- 4. Flüsse (Flows) definieren ---
    # Gastarif: Bezug von Gas mit Kosten (2 € pro Flusseinheit/Stunde)
    gas_supply_flow = fx.Flow(label='Gastarif', bus='Gas', effects_per_flow_hour=2)

    # Eingangsstrom zum GUD-Kraftwerk (Gas_to_GUD)
    gas_flow = fx.Flow(label='Gas_to_GUD', bus='Gas', size=300, relative_minimum=0.1)

    # Elektrischer Ausgangsstrom des GUD
    strom_flow = fx.Flow(label='Strom_Output', bus='Strom', size=50)

    # Thermischer Ausgangsstrom des GUD
    waerme_flow = fx.Flow(label='Waerme_Output', bus='Wärme')

    # --- 5. GUD (Gas- und Dampfturbinenanlage) konfigurieren ---
    # Die GUD-Instanz verknüpft Brennstoff (Gas), Strom und Wärme über lineare Gleichungen
    chp_gud = fx.linear_converters.GUD(
        label='mein_gud',
        # Kennlinien-Parameter (Intercept und Steigung für Brennstoffverbrauch und Wärmeauskopplung)
        power_fuel_intercept=1, power_fuel_slope=0.3,
        power_heat_intercept=1, power_heat_slope=0.6,
        fuel_flow=gas_flow,
        electrical_flow=strom_flow,
        thermal_flow=waerme_flow,
        defining_flow=gas_flow,  # die definierende Größe
        status_parameters=fx.StatusParameters(force_startup_tracking=True) # Tracking von Starts/Stopps
    )

    # --- 6. Quellen und Senken ---
    # Gasquelle, die den Gastarif bereitstellt
    gas_source = fx.Source(label='Gasquelle', outputs=[gas_supply_flow])

    # Stromsenke mit fixem Lastprofil basierend auf den Testdaten
    strom_sink = fx.Sink(
        label='Strombedarf',
        inputs=[fx.Flow(label='P_el', bus='Strom', size=1, fixed_relative_profile=elec_demand_per_h)]
    )
    # Wärmesenke mit einem konstanten Bedarf (hier überdimensioniert für den Test)
    waerme_sink = fx.Sink(
        label='Wärmebedarf',
        inputs=[fx.Flow(label='Q', bus='Wärme', size=1000)]
    )

    # Alle Elemente dem System hinzufügen
    flow_system.add_elements(costs, strom_bus, waerme_bus, gas_bus, gas_source, chp_gud, strom_sink, waerme_sink)

    # --- 7. Modellierung und Lösung ---
    # Mathematisches Modell aufbauen
    flow_system.build_model()

    # Modell lösen (Hinweis: Gurobi muss installiert und lizenziert sein, sonst anderen Solver nutzen)
    flow_system.solve(fx.solvers.GurobiSolver())

    # --- 8. Ergebnisse visualisieren ---
    # Bilanzplot für die GUD-Anlage anzeigen
    flow_system.statistics.plot.balance('mein_gud', show=True)
    # Sankey-Diagramm der Energieflüsse erstellen
    flow_system.statistics.plot.sankey.flows()
    chp_gud.piecewise_conversion.plot()