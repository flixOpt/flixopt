# Two-stage Optimization
**Note:** This example relies on time series data. You can find it in the `examples` folder of the FlixOpt repository.

This example demonstrates a two-stage optimization approach using downsampling to reduce computational time:

- **Stage 1 (Sizing)**: Optimize investment decisions (equipment sizes) using a downsampled model with lower temporal resolution
- **Stage 2 (Dispatch)**: Fix the optimal sizes from stage 1 and optimize the detailed dispatch using the full temporal resolution
- **Performance comparison**: Compare computation time and solution quality between two-stage and combined optimization

This approach is particularly useful for:
- Large-scale models with high temporal resolution
- Development and debugging phases where faster iterations are needed
- Investment planning where detailed dispatch is less critical during sizing decisions

```python
{! ../examples/05_Two-stage-optimization/two_stage_optimization.py !}
```
