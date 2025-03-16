::: flixOpt.components.LinearConverter

### Example Usage

```python
import flixOpt as fx

# Create a heat pump with COP = 3
heat_pump = fx.LinearConverter(
    label="HeatPump",
    inputs=[fx.Flow(label="power_in", bus='Heat')],
    outputs=[fx.Flow(label="heat_out", bus='Heat')],
    conversion_factors=[{"power_in": 3, "heat_out": 1}]
)
```
