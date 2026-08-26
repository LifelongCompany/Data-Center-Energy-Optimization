# UTCAS: Throughput- and Carbon-Aware Scheduling for Data Centers

## What it does
Data centers consume a growing share of electricity and emit carbon. Common energy-aware schedulers shift load heuristically and often hurt latency, causing SLA violations. UTCAS (Unifying Throughput and Carbon-Aware Scheduling) is a reinforcement-learning scheduler that shifts flexible jobs to hours with more renewable energy or lower grid carbon intensity, while keeping queue length and SLA deadlines under control.

UTCAS uses Proximal Policy Optimization (PPO) with a decoupled action space, plus a Lyapunov-drift term added to the reward. The Lyapunov term penalises growing queue backlog, so the agent keeps clearing the queue instead of delaying tasks indefinitely to save energy.

## Method
### 1. Decoupled PPO
The action space mixes discrete and continuous decisions:
- **Discrete:** accept or suspend each arriving task.
- **Continuous:** how much server capacity to allocate to it.

### 2. Lyapunov drift stability
A Lyapunov term based on the squared queue length is added to the reward. As backlog grows, the penalty grows, pushing the agent to drain the queue. This keeps energy savings from turning into unbounded delays.

## Customization
`main.py` is the entry point. Configure it through CLI arguments:
| Argument            |  Type  |    Default   | Description                                                |
| :------------------ | :----: | :----------: | :--------------------------------------------------------- |
| `--mode`            | String |   `run_all`  | `verification`, `diagnostic`, `ablation`, or `run_all`     |
| `--baseline`        | String | `rule_based` | Agent for comparison: `fifo`, `rule_based`, `max_capacity` |
| `--use_solar`       |  Flag  |    `False`   | Enables solar energy generation profiling                  |
| `--use_wind`        |  Flag  |    `False`   | Enables wind energy generation profiling                   |
| `--use_battery`     |  Flag  |    `False`   | Enables energy storage systems (ESS) logic                 |
| `--lyapunov_weight` |  Float |    `0.01`    | Scalar weight for queue stability urgency                  |
| `--sla_penalty`     |  Float |     `5.0`    | Cost multiplier for missing SLA deadlines                  |
| `--energy_weight`   |  Float |    `10.0`    | Scalar weight for electricity & carbon emission costs      |
| `--days`            |   Int  |     `30`     | Duration (in days) to synthesise the data environment      |
| `--epochs`          |   Int  |     `10`     | Number of PPO training epochs during diagnostic            |

### Running examples
**Example 1: Baseline comparison against FIFO**
```bash
python main.py --mode diagnostic --baseline fifo --days 15 --use_solar
```
**Example 2: Testing strong SLA punishment**
```bash
python main.py --mode diagnostic --sla_penalty 100.0 --lyapunov_weight 0.5
```
**Example 3: Full verification and simulation suite**
```bash
python main.py --mode run_all
```

## Reading the results
The `diagnostic` and `run_all` modes write four charts to `outputs/experiment_results`:
1. **`comparison_response.png`** — carbon intensity (background) against power draw of UTCAS (red) vs baseline (black dashed). Expect UTCAS to dip during high carbon-intensity periods.
2. **`stability_dynamics_diagnostic.png`** — backlog (area) and Lyapunov penalty over time. Look for the penalty spiking when backlog nears its threshold and pulling it back down; if backlog grows without bound, raise `lyapunov_weight`.
3. **`cumulative_carbon.png`** — accumulated carbon cost; a flatter UTCAS line means lower emissions.
4. **`sla_cdf.png`** — CDF of task wait times; a curve that reaches 1.0 sooner means lower latency.
