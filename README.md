# UTCAS: Unifying Throughput and Carbon-Aware Scheduling in Data Centers

## Project Overview
Data centers are responsible for a rapidly growing share of global electricity consumption and carbon emissions. However, existing energy-aware scheduling mechanisms often rely on heuristic load shifting that penalizes computational latency, inadvertently causing high instances of Service Level Agreement (SLA) violations.

This project introduces **UTCAS** (Unifying Throughput and Carbon-Aware Scheduling), an advanced Reinforcement Learning framework powered by Decoupled Proximal Policy Optimization (PPO) and Lyapunov Drift optimization. By applying Lyapunov drift constraints directly into the reward signal, the model achieves dynamic stability, ensuring that energy cost minimization does not cause unsustainable backlog accumulation or system queue failure. 

Through this environment, UTCAS strategically shifts flexible computational loads to periods of high renewable energy availability or low grid carbon intensity, while simultaneously adhering to strict SLA constraints.

## Methodology

### 1. Decoupled Proximal Policy Optimization (PPO)
UTCAS relies on a custom-designed Decoupled PPO architecture to solve the mixed continuous-discrete action space of data center operations. 
* **Discrete Actions:** Represents binary admissions (accept/suspend) for individual tasks arriving in real-time.
* **Continuous Actions:** Represents proportional resource allocations representing physical server capacity.

### 2. Lyapunov Drift Stability
A core theoretical contribution is the integration of Lyapunov Optimization to bind the Deep Reinforcement Learning (DRL) agent.
Traditional RL agents focused on energy minimization tend to perpetually delay tasks, causing systemic collapse. To counter this, UTCAS measures a mathematical "penalty" based on the quadratic drift of the queue length. 
* As the backlog grows, the Lyapunov pressure increases exponentially. 
* The PPO agent perceives this pressure in the unified reward signal, effectively forcing it to clear the queue (accelerate throughput) when the system approaches instability, naturally balancing energy savings with reliable service times.

---

## Customization Guide

The system features a robust, unified `main.py` orchestrator. All configurations and hyperparameters can be modified directly via CLI arguments, allowing researchers and operators to seamlessly configure ablation studies, baselines, and custom energy portfolios.

### Available Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--mode` | String | `run_all` | `verification`, `diagnostic`, `ablation`, or `run_all` |
| `--baseline` | String | `rule_based` | Agent for comparison: `fifo`, `rule_based`, `max_capacity` |
| `--use_solar` | Flag | `False` | Enables Solar Energy generation profiling |
| `--use_wind` | Flag | `False` | Enables Wind Energy generation profiling |
| `--use_battery` | Flag | `False` | Enables Energy Storage Systems (ESS) logic |
| `--lyapunov_weight` | Float | `0.01` | Scalar weight for queue stability urgency |
| `--sla_penalty` | Float | `5.0` | Cost multiplier for missing SLA deadlines |
| `--energy_weight` | Float | `10.0` | Scalar weight for electricity & carbon emission costs |
| `--days` | Int | `30` | Duration (in days) to synthesize data environment |
| `--epochs` | Int | `10` | Number of PPO training epochs during diagnostic |

### Running Examples

**Example 1: Baseline Comparison against FIFO**
Run the diagnostic protocol comparing the RL agent against a basic FIFO baseline for 15 days of data, utilizing solar energy.

```bash
python main.py --mode diagnostic --baseline fifo --days 15 --use_solar
```

**Example 2: Testing Extreme SLA Punishments**
Evaluate how the agent behaves when SLA violations are severely punished, essentially forcing high throughput regardless of the carbon cost.

```bash
python main.py --mode diagnostic --sla_penalty 100.0 --lyapunov_weight 0.5
```

**Example 3: Full Verification and Simulation Suite**
Run all phases (Environment tests, RL Training Diagnostics, and Ablation Study).

```bash
python main.py --mode run_all
```

---

## Result Interpretation

Upon running the `diagnostic` or `run_all` modules, the engine generates five core visual analytical charts located in `outputs/experiment_results`. 

1. **Response Dynamics (Power Comparison): `comparison_response.png`**
   * *What it is:* A dual-axis plot showing carbon intensity overlaid with the total power consumption of both the UTCAS agent and the Baseline.
   * *How to read it:* Verify that during spikes in the gray background area (Carbon Intensity), the red line (UTCAS PPO) dips substantially lower than the black dashed line (Baseline). This proves the agent is actively "shedding load" during dirty-grid hours.

2. **Lyapunov Stability Dynamics: `stability_dynamics_diagnostic.png`**
   * *What it is:* A line-and-area chart displaying the direct correlation between physical system backlog and mathematical Lyapunov penalty over time.
   * *How to read it:* Look for periodic "breathing" patterns. As the blue area (Backlog) hits an upper threshold, the red dotted line (Penalty) should spike, forcing the backlog immediately back down. If the backlog constantly grows unbounded, the `lyapunov_weight` may be too low.

3. **Cumulative Carbon Footprint: `cumulative_carbon.png`**
   * *What it is:* A straightforward comparative line chart showing the accumulated total carbon proxy cost.
   * *How to read it:* A flatter slope implies better environmental friendliness. The UTCAS line should visibly sit below baselines at the end of the simulation.

4. **SLA Compliance (Wait Time CDF): `sla_cdf.png`**
   * *What it is:* A Cumulative Distribution Function (CDF) showing the probability distribution of task wait times.
   * *How to read it:* The faster the curve reaches `1.0` (100% probability) on the Y-axis, the more responsive the system is. If a curve stretches far to the right, it indicates severe latency trailing.

5. **Physical Compliance Landscape: `physical_compliance.png`**
   * *What it is:* A jittered strip-plot highlighting the precision of physical load requests by each agent.
   * *How to read it:* Look at the density of points. The UTCAS agent should theoretically cluster heavily in the "Golden Zone" [0.85 - 1.0], meaning it is operating near maximum efficient capacity without crossing the physical constraint barrier (1.0). Data points exceeding the red line represent physical impossibilities or over-allocations.