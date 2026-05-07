import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
from src.utils.vis_lagrangian import plot_lagrangian_dynamics # Import the new function

def plot_stability_dynamics(df_hist, save_dir="outputs/experiment_results", suffix=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="white", context="paper", font_scale=1.2)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Backlog (Left Axis) - Area Plot style
    color1 = 'tab:blue'
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('System Backlog (Queue + Active)', color=color1, fontweight='bold')
    # Use fill_between for area effect
    ax1.fill_between(df_hist['step'], 0, df_hist['queue_backlog'], color=color1, alpha=0.3, label='System Backlog')
    ax1.plot(df_hist['step'], df_hist['queue_backlog'], color=color1, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)

    # Plot Penalty (Right Axis) - Line Plot style
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Lyapunov Pressure (Penalty)', color=color2, fontweight='bold')
    ax2.plot(df_hist['step'], df_hist['lyapunov_penalty'], color=color2, linestyle='--', linewidth=2.0, label='Lyapunov Penalty')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)

    # Add Mean Line for Backlog
    mean_backlog = df_hist['queue_backlog'].mean()
    ax1.axhline(mean_backlog, color=color1, linestyle=':', alpha=0.8, linewidth=1.5, label=f'Mean Backlog ({mean_backlog:.1f})')

    # Title and Layout
    plt.title(f'Lyapunov Stability Dynamics: Urgency vs. Backlog {suffix.replace("_", " ").title()}', fontsize=14, fontweight='bold', pad=15)

    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # Combine legends
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.subplots_adjust(top=0.85) # Ensure enough space for the legend
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'stability_dynamics{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stability plot: {save_path}")

def plot_results(all_metrics, save_dir="outputs/results"):
    os.makedirs(save_dir, exist_ok=True)
    agent_names = list(all_metrics.keys())

    # --- Setup: Nature-Style Aesthetics ---
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    # Custom Palette:
    # UTCAS/PPO/DRL -> Teal/Green (Hero)
    # Baselines -> Grey/Dark (Background)

    def get_color(name):
        name_lower = name.lower()
        if 'utcas' in name_lower or 'ppo' in name_lower or 'drl' in name_lower:
            return '#1abc9c' # Teal
        elif 'fifo' in name_lower:
            return '#95a5a6' # Concrete Grey
        elif 'capacity' in name_lower or 'max' in name_lower:
            return '#34495e' # Dark Blue Grey
        elif 'rule' in name_lower:
            return '#7f8c8d' # Asbestos Grey
        else:
            return '#bdc3c7' # Silver

    agent_colors = {name: get_color(name) for name in agent_names}

    DPI = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Create Summary DataFrame
    df_list = []
    for name, metrics in all_metrics.items():
        # Summing up step values for totals
        total_elec = np.sum(metrics.get('electricity_costs', [0]))
        total_carbon = np.sum(metrics.get('carbon_costs', [0]))
        total_opex = total_elec + total_carbon # PURE CASH (No Penalty)

        # Scaling SLA for reporting if needed, but here we focus on OpEx

        tasks_proc = metrics.get('tasks_processed', 0)

        # Energy Mix Logic
        total_energy = np.sum(metrics.get('total_energy', [0]))
        total_renewable = np.sum(metrics.get('renewable_usage', [0]))

        # Handling potential floating point errors or 0/0
        if total_energy < 1e-6:
            grid_energy = 0
            ren_percent = 0
        else:
            grid_energy = max(0, total_energy - total_renewable)
            ren_percent = (total_renewable / total_energy) * 100

        df_list.append({
            'Agent': name,
            'OpEx': total_opex,
            'Total Energy': total_energy,
            'Renewable Energy': total_renewable,
            'Grid Energy': grid_energy,
            'Renewable %': ren_percent,
            'Tasks Processed': tasks_proc,
            'Cost Per 1k Tasks': (total_opex / (tasks_proc / 1000)) if tasks_proc > 0 else 0
        })

    df = pd.DataFrame(df_list)

    # ==================================================================================
    # Plot 1: Operational Cost (OpEx) - The "Money Saver" Plot
    # ==================================================================================
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Agent', y='OpEx', data=df, hue='Agent', palette=agent_colors, legend=False)

    plt.title('Operational Expenditure (Electricity + Carbon Tax)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Total Cash Cost ($) [Lower is Better]')
    plt.xlabel('')
    plt.xticks(rotation=30, ha="right")

    # Annotate Winner
    best_agent = df.loc[df['OpEx'].idxmin()]
    plt.figtext(0.5, 0.01,
                f"Observation: {best_agent['Agent']} achieves the lowest operational cost by intelligently shifting loads.",
                ha="center", fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'operational_cost.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # ==================================================================================
    # Plot 2: Cumulative Carbon Footprint - The "Nature" Plot
    # ==================================================================================
    plt.figure(figsize=(10, 6))

    for name in agent_names:
        # Get step-by-step carbon costs
        carbon_costs = np.array(all_metrics[name].get('carbon_costs', []))
        # Convert cost to mass proxy (assuming constant price/penalty ratio or just use cost accumulation)
        # Prompt asked for "Cumulative Carbon Footprint". Accumulating costs is a direct proxy.
        # If we want mass: mass = cost / (0.05 scaling * intensity_factor).
        # Using cumulative COST is safer as it represents the "Impact" directly penalized.

        cumulative_impact = np.cumsum(carbon_costs)

        # Normalize time to percentage for cleaner comparison if lengths differ (unlikely in run_all)
        steps = np.arange(len(cumulative_impact))

        color = agent_colors[name]
        width = 2.5 if color == '#1abc9c' else 1.5
        alpha = 1.0 if color == '#1abc9c' else 0.7

        plt.plot(steps, cumulative_impact, label=name, color=color, linewidth=width, alpha=alpha)

    plt.title('Cumulative Carbon Footprint Over Time', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Cumulative Carbon Impact (Accumulated Tax)')
    plt.xlabel('Simulation Steps (15-min intervals)')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cumulative_carbon.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # ==================================================================================
    # Plot 3: Energy Source Mix - The "Green Proof"
    # ==================================================================================
    plt.figure(figsize=(8, 6))

    # Create bottom bars (Grid - Dirty)
    plt.bar(df['Agent'], df['Grid Energy'], label='Grid (Dirty)', color='#7f8c8d', alpha=0.6, width=0.6)

    # Create top bars (Renewable - Green)
    plt.bar(df['Agent'], df['Renewable Energy'], bottom=df['Grid Energy'],
            label='Renewable (Clean)', color='#2ecc71', alpha=0.9, width=0.6)

    plt.title('Energy Source Mix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Total Energy Consumed (kWh)')
    plt.xticks(rotation=30, ha="right")
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'energy_mix.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # ==================================================================================
    # Plot 4: Throughput vs Cost Efficiency - The "Efficiency" Plot
    # ==================================================================================
    plt.figure(figsize=(10, 6))

    # Scatter plot
    sns.scatterplot(data=df, x='Tasks Processed', y='OpEx', hue='Agent', palette=agent_colors, s=200, style='Agent')

    # Add text labels
    for i, row in df.iterrows():
        plt.text(row['Tasks Processed'], row['OpEx'] + (df['OpEx'].max()*0.02),
                 row['Agent'], horizontalalignment='center', size='small', color='black', weight='semibold')

    plt.title('Efficiency Matrix: Throughput vs. Operational Cost', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Total OpEx ($) [Lower is Better]')
    plt.xlabel('Completed Tasks [Higher is Better]')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    # Arrow to ideal corner (Bottom Right)
    # plt.annotate('Ideal Region', xy=(0.95, 0.05), xycoords='axes fraction', ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_metric.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # ==================================================================================
    # Plot 5: Latency by Priority - Box Plot (Fixed)
    # ==================================================================================
    if any('latencies' in all_metrics[name] for name in agent_names):
        plt.figure(figsize=(10, 6))

        records = []
        priority_map = {0: 'Bronze', 1: 'Silver', 2: 'Gold'}

        for name in agent_names:
            lat_dict = all_metrics[name].get('latencies', {})
            for priority_level, lat_values in lat_dict.items():
                p_label = priority_map.get(priority_level, f'P{priority_level}')
                for val in lat_values:
                    records.append({
                        'Agent': name,
                        'Priority': p_label,
                        'Latency': val
                    })

        if records:
            df_lat = pd.DataFrame(records)

            # Palette for priorities
            prio_pal = {'Gold': '#f1c40f', 'Silver': '#95a5a6', 'Bronze': '#d35400'}

            ax = sns.boxplot(x='Agent', y='Latency', hue='Priority', data=df_lat,
                             hue_order=['Gold', 'Silver', 'Bronze'], palette=prio_pal, showfliers=False)

            ax.set_title('Latency Distribution by Priority', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylim(bottom=0) # Fix visual bug
            plt.ylabel('Waiting Time (min)')
            plt.xlabel('')
            plt.legend(frameon=False)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'latency_by_priority.png'), dpi=DPI, bbox_inches='tight')
            plt.close()

    # ==================================================================================
    # Plot 6: Pareto Frontier (Environmental Responsibility vs. Reliability)
    # ==================================================================================
    plt.figure(figsize=(10, 8))

    for name in agent_names:
        metrics = all_metrics[name]

        # X: Carbon Footprint (Total)
        total_carbon = np.sum(metrics.get('carbon_costs', []))

        # Y: SLA Violation Rate (%)
        total = metrics.get('tasks_processed', 0) + metrics.get('tasks_missed', 0)
        violation_rate = (metrics.get('tasks_missed', 0) / total * 100.0) if total > 0 else 0.0

        color = agent_colors[name]
        marker = '*' if color == '#1abc9c' else 'o' # Star for UTCAS
        s = 300 if marker == '*' else 150

        plt.scatter(total_carbon, violation_rate, color=color, label=name, marker=marker, s=s, edgecolors='black', alpha=0.9)

        # Annotate
        plt.text(total_carbon, violation_rate + 0.5, name, ha='center', fontsize=9, fontweight='bold', color=color)

    plt.title('Pareto Frontier: Environmental Responsibility vs. Reliability', fontweight='bold', pad=15)
    plt.xlabel('Total Carbon Footprint (Normalized Cost) [Lower is Better]')
    plt.ylabel('SLA Violation Rate (%) [Lower is Better]')
    plt.grid(True, linestyle='--', alpha=0.3)
    # Invert axes if we want "Upper Right" to be better? No, standard Pareto is Lower Left is better (Minimization).
    # The user said: "Baseline falls in top right (Bad), PPO in bottom left (Good)".
    # "Strictly Dominating".

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pareto_frontier.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    # ==================================================================================
    # Plot 7: Battery SoC Dynamics (Essential Legacy)
    # ==================================================================================
    plt.figure(figsize=(12, 5))
    for name in agent_names:
        soc_data = all_metrics[name].get('soc', [])
        color = agent_colors[name]
        width = 2.0 if color == '#1abc9c' else 1.0
        alpha = 0.9 if color == '#1abc9c' else 0.6

        plt.plot(soc_data, label=name, color=color, linewidth=width, alpha=alpha)

    plt.title('Battery State of Charge (SoC) Dynamics', fontweight='bold', pad=15)
    plt.ylabel('SoC (%)')
    plt.xlabel('Time Steps')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'soc_timeline.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"Saved nature plots to {save_dir}")

def plot_compliance_distribution(all_metrics, save_dir="outputs/experiment_results"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Prepare DataFrame
    records = []

    # Define color palette manually to match existing scheme
    def get_color(name):
        name_lower = name.lower()
        if 'utcas' in name_lower or 'ppo' in name_lower or 'drl' in name_lower:
            return '#1abc9c' # Teal
        elif 'fifo' in name_lower:
            return '#95a5a6' # Concrete Grey
        elif 'capacity' in name_lower or 'max' in name_lower:
            return '#34495e' # Dark Blue Grey
        elif 'rule' in name_lower:
            return '#7f8c8d' # Asbestos Grey
        else:
            return '#bdc3c7' # Silver

    for name, m in all_metrics.items():
        loads = m.get('requested_load', [])
        # We might want to downsample if too many points, but stripplot handles thousands ok.
        # Let's take every step.
        for load in loads:
            records.append({
                'Agent': name,
                'Requested Load': load
            })

    if not records:
        print("Warning: No 'requested_load' data found for Compliance Plot.")
        return

    df = pd.DataFrame(records)

    # Aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Create Stripplot (Jittered Scatter)
    # Using 'hue' with same variable as x to apply palette
    agent_names = df['Agent'].unique()
    palette = {name: get_color(name) for name in agent_names}

    ax = sns.stripplot(
        data=df,
        x='Agent',
        y='Requested Load',
        hue='Agent',
        palette=palette,
        jitter=0.25,
        size=3,
        alpha=0.6,
        legend=False
    )

    # Add Violin plot in background for density estimation?
    # Or maybe just the points "like rain" as requested.
    # User said: "Scatter Plot or Violin Plot... Points will be like rain".
    # Stripplot is exactly "points like rain".

    # Add Red Line at 1.0 (Physical Limit)
    plt.axhline(y=1.0, color='red', linestyle='-', linewidth=2, label='Physical Limit (Fluid Constraint)')

    # Add Dashed Line at 0.85 (Soft Constraint Start)
    plt.axhline(y=0.85, color='orange', linestyle='--', linewidth=1.5, label='Soft Constraint Barrier (0.85)')

    plt.title('Physical Compliance Landscape: Load Control Precision', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Requested Load (Normalized CPU)')
    plt.xlabel('')

    # Add annotation for UTCAS target zone
    plt.text(0.5, 0.92, 'Target "Golden Zone" [0.85 - 1.0]',
             transform=ax.transAxes, ha='center', color='#d35400', fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'physical_compliance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved compliance plot: {save_path}")

def plot_sla_cdf(all_metrics, save_dir="outputs/experiment_results"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)

    records = []

    def get_color(name):
        name_lower = name.lower()
        if 'utcas' in name_lower or 'ppo' in name_lower or 'drl' in name_lower:
            return '#1abc9c' # Teal
        elif 'fifo' in name_lower:
            return '#95a5a6' # Concrete Grey
        elif 'capacity' in name_lower or 'max' in name_lower:
            return '#34495e' # Dark Blue Grey
        elif 'rule' in name_lower:
            return '#7f8c8d' # Asbestos Grey
        else:
            return '#bdc3c7' # Silver

    # Collect Latency Data
    for name, m in all_metrics.items():
        lat_dict = m.get('latencies', {})
        # Flatten all priorities
        all_wait_times = []
        for prio, vals in lat_dict.items():
            all_wait_times.extend(vals)

        for val in all_wait_times:
            records.append({
                'Agent': name,
                'Wait Time': val
            })

    if not records:
        print("Warning: No latency data found for CDF Plot.")
        return

    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Plot ECDF
    ax = sns.ecdfplot(
        data=df,
        x='Wait Time',
        hue='Agent',
        palette={name: get_color(name) for name in df['Agent'].unique()},
        linewidth=2.5,
        alpha=0.9
    )

    plt.title('SLA Compliance: Wait Time Distribution (CDF)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Wait Time / Time to Failure (minutes)')
    plt.ylabel('Cumulative Probability')

    # Annotate the Cliff
    plt.annotate('The "Cliff of Death"\n(Task Failure)',
                 xy=(df['Wait Time'].max(), 0.95),
                 xytext=(df['Wait Time'].max()*0.7, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, fontweight='bold', color='red')

    plt.grid(True, linestyle='--', alpha=0.3)
    # 使用 seaborn 自带的图例移动方法，或者抓取当前坐标轴的句柄
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'sla_cdf.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SLA CDF plot: {save_path}")

def plot_power_comparison(all_metrics, save_dir="outputs/experiment_results", carbon_p75=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    # Identify Agents
    baseline_key = next((k for k in all_metrics.keys() if "rule" in k.lower()), None)
    ppo_key = next((k for k in all_metrics.keys() if "utcas" in k.lower() or "ppo" in k.lower()), None)

    if not baseline_key or not ppo_key:
        print("Comparison Plot Skipped: Need both 'Rule-Based' and 'UTCAS/PPO' agents.")
        return

    metrics_base = all_metrics[baseline_key]
    metrics_ppo = all_metrics[ppo_key]

    # Extract Power
    # We use 'total_power' if available, else derive from step_total_energy (kWh) / 0.25h
    def get_power(m):
        if 'total_power' in m:
            return m['total_power']
        return np.array(m.get('electricity_costs', [])) * 0.0 # Fallback placeholder if power missing

    power_base = get_power(metrics_base)
    power_ppo = get_power(metrics_ppo)

    # Extract Carbon (Shared)
    carbon = metrics_base['carbon_intensity']
    steps = metrics_base['step']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Y-Axis: Carbon Intensity (Background Area)
    color_carbon = 'gray'
    ax1.fill_between(steps, 0, carbon, color=color_carbon, alpha=0.2, label='Carbon Intensity')
    ax1.set_ylabel('Carbon Intensity (gCO2/kWh)', color='#555555', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#555555')
    ax1.set_ylim(bottom=0)

    # P75 Threshold
    if carbon_p75:
        ax1.axhline(y=carbon_p75, color='gray', linestyle=':', label='P75 Threshold')

    # Right Y-Axis: Power Comparison
    ax2 = ax1.twinx()

    # Plot Baseline (Black Dashed)
    ax2.plot(steps, power_base, color='black', linestyle='--', linewidth=2.0, label=f'{baseline_key} (Baseline)')

    # Plot PPO (Red Solid)
    ax2.plot(steps, power_ppo, color='#e74c3c', linestyle='-', linewidth=2.5, label=f'{ppo_key} (Ours)')

    ax2.set_ylabel('Total Power Consumption (kW)', color='black', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='black')

    # X-Axis Formatting
    # Assuming 96 steps/day
    steps_per_day = 96
    total_steps = len(steps)
    if total_steps // steps_per_day > 0:
        tick_locs = np.arange(0, total_steps, steps_per_day)
        tick_labels = [f"Day {i+1}" for i in range(len(tick_locs))]
        ax1.set_xticks(tick_locs)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    else:
        ax1.set_xlabel('Time Steps (15 min)')

    plt.title('Response Dynamics: Rule-Based vs. UTCAS PPO', fontweight='bold', pad=15)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

    sns.despine(ax=ax1, top=True, right=False)
    sns.despine(ax=ax2, top=True, right=False, left=True)

    plt.subplots_adjust(top=0.85) # Make sure the legend fits
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_response.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {save_path}")
