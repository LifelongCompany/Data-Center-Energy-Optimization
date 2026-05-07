import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.core.components import Battery, ITModel, CoolingModel, Task
from src.data.data_loader import BaseDataLoader

class DataCenterEnv(gym.Env):
    MAX_PRICE = 1.0         # $/kWh (Safety margin over 0.2)
    MAX_CARBON = 1000.0     # g/kWh (Safety margin over 800)
    MAX_POWER = 100.0       # kW (Approx peak IT + Cooling + Battery)
    MAX_SLA_COST = 5000.0   # Reference cost for normalization (Approx 1 Gold Task failure)

    def __init__(self, data_loader: BaseDataLoader, env_config: dict, decoupled: bool = False):
        super(DataCenterEnv, self).__init__()
        self.decoupled = decoupled

        self.weight_energy = env_config.get('weight_energy', 10.0)  # Rebalanced energy weight
        self.weight_carbon = env_config.get('weight_carbon', 5.0)   # Rebalanced carbon weight
        self.weight_sla = env_config.get('weight_sla', 1.0)
        self.weight_viol = env_config.get('weight_viol', 1.0)
        self.lyapunov_weight = env_config.get('lyapunov_weight', 0.01) # Reasonable default scale

        # Legacy support
        if 'energy_cost_weight' in env_config:
            self.weight_energy = env_config['energy_cost_weight']
            self.weight_carbon = env_config['energy_cost_weight']

        self.sla_penalty_multiplier = env_config.get('sla_penalty_multiplier', 5.0) # Downscaled SLA multiplier
        self.priority_weights = env_config.get('priority_weights', {0: 0.1, 1: 1.0, 2: 10.0})

        # Lagrangian Parameters
        self.target_scaling = 0.95
        self.lagrangian_lambda = 0.0
        self.lambda_lr = 0.05

        # Curriculum Learning
        self.curriculum_phase = env_config.get('curriculum_phase', 1)

        self.data_loader = data_loader
        self.data_loader.setup()
        self.env_df = self.data_loader.get_env_data()
        self.task_df = self.data_loader.get_task_data()

        if 'carbon_intensity' in self.env_df.columns:
            self.carbon_p75 = self.env_df['carbon_intensity'].quantile(0.75)
            self.carbon_p50 = self.env_df['carbon_intensity'].quantile(0.50)
        else:
            self.carbon_p75 = 500.0
            self.carbon_p50 = 300.0

        self.max_steps = len(self.env_df)
        self.step_minutes = 15

        self.use_solar = env_config.get('use_solar', False)
        self.use_wind = env_config.get('use_wind', False)

        self.battery = Battery(capacity=100.0, max_power=50.0)
        self.it_model = ITModel(p_idle=10.0, p_peak=50.0)
        self.cooling_model = CoolingModel(cop_nom=3.0, alpha=0.05, t_ref=20.0, gamma=0.1)

        self.queue = []
        self.suspended_queue = []
        self.active_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []

        self.n_candidate_tasks = env_config.get('n_candidate_tasks', 20)
        self.m_active_slots = 30
        action_dim = 1 + self.n_candidate_tasks + self.m_active_slots

        if self.decoupled:
            continuous_dim = 1 + self.m_active_slots
            discrete_dims = [2] * self.n_candidate_tasks
            self.action_space = spaces.Tuple((
                spaces.Box(low=-1.0, high=1.0, shape=(continuous_dim,), dtype=np.float32),
                spaces.MultiDiscrete(discrete_dims)
            ))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Features: [Solar, Wind, Price, Carbon, BaseLoad, SoC, SinTime, CosTime, Load, Q_Gold, Q_Silver, Q_Bronze, Susp_Count, Susp_Deadline]
        self.env_features = 14
        self.forecast_window = 12
        self.forecast_features = self.forecast_window * 2
        # Task Features: [size, rem_time, waiting_time, priority]
        self.task_features = 4

        obs_dim = (self.env_features + self.forecast_features +
                   (self.n_candidate_tasks * self.task_features) +
                   (self.m_active_slots * self.task_features))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.current_time_min = 0.0
        self.current_load = 0.0
        self.prev_load = 0.0
        self.renewable_usage_kwh = 0.0
        self.total_energy_kwh = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_time_min = 0.0
        self.battery.reset()
        self.queue = []
        self.suspended_queue = []
        self.active_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.renewable_usage_kwh = 0.0
        self.total_energy_kwh = 0.0
        self.current_load = 0.0
        self.prev_load = 0.0
        return self._get_obs(), {}

    def calculate_task_penalty(self, task, delay_time):
        if delay_time <= 0:
            return 0.0

        # Simple Linear Penalty for violation
        pain = 10.0 * delay_time

        # Adjust weight by priority
        if task.priority == 2: # Gold
            pain *= 5.0
        elif task.priority == 0: # Bronze
            pain *= 0.1

        return pain

    def step(self, action):
        if self.decoupled:
            continuous_actions, discrete_actions = action
            battery_action = float(continuous_actions[0])
            resource_actions = continuous_actions[1:]
            admissions = discrete_actions
        else:
            battery_action = float(action[0])
            admissions = action[1 : 1 + self.n_candidate_tasks]
            resource_actions = action[1 + self.n_candidate_tasks :]

        actual_battery_power = 0.0
        # self.battery.step(battery_power_cmd) # Disabled

        current_arrivals = self.task_df[
            (self.task_df['arrival_time'] >= self.current_time_min) &
            (self.task_df['arrival_time'] < self.current_time_min + self.step_minutes)
        ]
        for _, row in current_arrivals.iterrows():
            new_task = Task(row['task_id'], row['arrival_time'], row['duration'],
                            row['deadline'], row['cpu_req'], row['priority'])
            self.queue.append(new_task)
        # Sort queue: Gold first, then FIFO
        self.queue.sort(key=lambda x: (-x.priority, x.arrival_time))


        # Construct Candidate Pool: Suspended First, then Queue
        candidate_pool = self.suspended_queue + self.queue
        tasks_to_remove_from_source = []

        for i in range(min(len(candidate_pool), self.n_candidate_tasks)):
            decision = admissions[i]
            # Simple threshold > 0.5 (or > 0 for discrete)
            if decision > 0.0:
                task = candidate_pool[i]

                # Activate
                if task.status != 'active':
                    if task.status == 'queued': # Fresh from queue
                        task.start_time = self.current_time_min
                    task.status = 'active'
                    self.active_tasks.append(task)
                    tasks_to_remove_from_source.append(task)

        # Cleanup sources
        for t in tasks_to_remove_from_source:
            if t in self.queue: self.queue.remove(t)
            if t in self.suspended_queue: self.suspended_queue.remove(t)

        current_base_load = self.env_df.iloc[self.current_step]['base_load']
        intended_task_load = 0.0
        raw_allocations = []
        tasks_to_evict = []


        active_copy = list(self.active_tasks)

        for i, task in enumerate(active_copy):
            if i < self.m_active_slots:
                # Interpret action [-1, 1] -> [0, 1]
                raw_val = (resource_actions[i] + 1.0) / 2.0
                alloc = max(0.0, min(1.0, raw_val))

                if alloc < 0.01:
                    tasks_to_evict.append(task)
                    alloc = 0.0
            else:
                # No slot available
                tasks_to_evict.append(task)
                alloc = 0.0

            raw_allocations.append(alloc)
            if task not in tasks_to_evict:
                intended_task_load += task.cpu_req * alloc

        for t in tasks_to_evict:
            if t in self.active_tasks:
                self.active_tasks.remove(t)
                t.status = 'suspended'
                self.suspended_queue.append(t)

        total_capacity_available = 1.0 - current_base_load
        scaling_factor = 1.0

        if intended_task_load > total_capacity_available + 1e-6:
            if total_capacity_available > 0:
                scaling_factor = total_capacity_available / intended_task_load
            else:
                scaling_factor = 0.0

        total_requested_cpu = 0.0
        finished_tasks = []
        load_breakdown = {0: 0.0, 1: 0.0, 2: 0.0}

        for i, task in enumerate(active_copy):
            if task in self.active_tasks:
                raw_alloc = raw_allocations[i]
                final_alloc = raw_alloc * scaling_factor

                task.progress(final_alloc, self.step_minutes)
                current_task_load = task.cpu_req * final_alloc
                total_requested_cpu += current_task_load

                # Track load by priority for visualization
                if task.priority in load_breakdown:
                    load_breakdown[task.priority] += current_task_load
                else:
                    # Fallback for unexpected priority
                    load_breakdown[0] += current_task_load

                if task.status == 'completed':
                    finished_tasks.append(task)

        for t in finished_tasks:
            if t in self.active_tasks:
                self.active_tasks.remove(t)
                self.completed_tasks.append(t)

        all_live_tasks = self.queue + self.suspended_queue + self.active_tasks
        expired_tasks = []
        for t in all_live_tasks:
            if self.current_time_min > t.deadline:
                t.status = 'failed'
                self.failed_tasks.append(t)
                expired_tasks.append(t)

        for t in expired_tasks:
            if t in self.queue: self.queue.remove(t)
            if t in self.suspended_queue: self.suspended_queue.remove(t)
            if t in self.active_tasks: self.active_tasks.remove(t)

        total_cpu_utilization = min(1.0, total_requested_cpu + current_base_load)

        it_power = self.it_model.calculate_power(total_cpu_utilization)
        t_ambient = self.env_df.iloc[self.current_step]['temperature_celsius']
        cooling_power = self.cooling_model.calculate_power(it_power, t_ambient, total_cpu_utilization)
        total_load_kw = it_power + cooling_power

        solar_power = 0.0
        if self.use_solar:
             solar_power = self.env_df.iloc[self.current_step]['solar_generation'] * 20.0
        wind_power = 0.0
        if self.use_wind:
             wind_power = self.env_df.iloc[self.current_step]['wind_generation'] * 20.0

        total_renewable = solar_power + wind_power
        # Battery is 0.0
        net_power = max(0.0, total_load_kw - total_renewable)

        price = self.env_df.iloc[self.current_step]['grid_price']
        carbon_intensity = self.env_df.iloc[self.current_step]['carbon_intensity']

        elec_cost_step = net_power * (self.step_minutes / 60.0) * price

        carbon_cost_step = net_power * (self.step_minutes / 60.0) * (carbon_intensity / 1000.0)

        # SLA Penalty: Only for tasks that FAILED this step
        sla_penalty_cost = 0.0
        # Check failed tasks from this step (expired_tasks)
        for t in expired_tasks:
            # Penalty = Multiplier * (Lateness or Work)
            # User simplified: "if wait_time < deadline: penalty = 0"
            # Since these FAILED, wait_time > deadline (or current > deadline).
            # Let's use the calculate_task_penalty with delay > 0
            delay = self.current_time_min - t.deadline
            pain = self.calculate_task_penalty(t, delay)
            sla_penalty_cost += pain * self.sla_penalty_multiplier

        # Normalized Rewards
        norm_elec = elec_cost_step / (self.MAX_POWER * 0.25 * self.MAX_PRICE)
        norm_carbon = carbon_cost_step / (self.MAX_POWER * 0.25 * (self.MAX_CARBON/1000.0))
        norm_sla = sla_penalty_cost / self.MAX_SLA_COST

        # Lyapunov Penalty (Backlog)
        backlog = len(self.queue) + len(self.suspended_queue) + len(self.active_tasks)
        lyapunov_penalty = backlog * self.lyapunov_weight

        total_reward = -(
            self.weight_energy * norm_elec +
            self.weight_carbon * norm_carbon +
            self.weight_sla * norm_sla +
            lyapunov_penalty
        )

        self.current_step += 1
        self.current_time_min += self.step_minutes
        self.current_load = total_cpu_utilization
        terminated = self.current_step >= self.max_steps - 1
        truncated = False

        # Info
        total_cpu_load_actual = current_base_load + load_breakdown[0] + load_breakdown[1] + load_breakdown[2]
        if total_cpu_load_actual > 1e-6:
            frac_critical = (current_base_load + load_breakdown[2]) / total_cpu_load_actual
        else:
            frac_critical = 1.0 # All idle/base

        power_critical = total_load_kw * frac_critical
        power_flexible = total_load_kw * (1.0 - frac_critical)

        info = {
            "cost": elec_cost_step,
            "carbon": carbon_cost_step,
            "penalty": sla_penalty_cost,
            "tasks_processed": len(self.completed_tasks),
            "tasks_missed": len(self.failed_tasks),
            "queue_len": len(self.queue),
            "suspended_len": len(self.suspended_queue),
            "active_len": len(self.active_tasks),
            "soc": 0.0, # Battery disabled
            "renewable_usage": min(total_load_kw, total_renewable) * (self.step_minutes/60.0),
            "step_renewable_energy": min(total_load_kw, total_renewable) * (self.step_minutes/60.0),
            "step_total_energy": total_load_kw * (self.step_minutes/60.0),
            "carbon_intensity": carbon_intensity,
            "solar_generation": solar_power / 20.0, # unscaled
            "lyapunov_penalty": lyapunov_penalty,
            "queue_backlog": backlog,
            "queue_len_bronze": sum(1 for t in (self.queue + self.suspended_queue) if t.priority==0),
            # Visualization Metrics
            "load_gold": load_breakdown[2],
            "load_silver": load_breakdown[1],
            "load_bronze": load_breakdown[0],
            "base_load": current_base_load,
            "total_power_kw": total_load_kw,
            "power_critical": power_critical,
            "power_flexible": power_flexible
        }

        return self._get_obs(), total_reward, terminated, truncated, info

    def _get_obs(self):
        row = self.env_df.iloc[self.current_step]
        time_of_day_min = self.current_time_min % (24 * 60)
        norm_time = time_of_day_min / (24 * 60) * 2 * np.pi

        MAX_Q = 200.0
        all_backlog = self.queue + self.suspended_queue
        gold_q = sum(1 for t in all_backlog if t.priority == 2) / MAX_Q
        silver_q = sum(1 for t in all_backlog if t.priority == 1) / MAX_Q
        bronze_q = sum(1 for t in all_backlog if t.priority == 0) / MAX_Q

        susp_count = len(self.suspended_queue) / MAX_Q
        if self.suspended_queue:
            susp_avg_deadline = np.mean([(t.deadline - self.current_time_min)/60.0 for t in self.suspended_queue])
            susp_avg_deadline_norm = np.clip(susp_avg_deadline / 24.0, -1.0, 1.0)
        else:
            susp_avg_deadline_norm = 0.0

        env_state = [
            row['solar_generation'], row.get('wind_generation', 0.0),
            row['grid_price'], row['carbon_intensity'],
            row['base_load'], 0.0, # Battery SOC is always 0
            np.sin(norm_time), np.cos(norm_time),
            self.current_load, gold_q, silver_q, bronze_q,
            susp_count, susp_avg_deadline_norm
        ]

        # Forecast (Zeros or simplified)
        forecast_vec = np.zeros(self.forecast_features, dtype=np.float32)

        start_idx = self.current_step + 1
        end_idx = start_idx + self.forecast_window
        future_slice = self.env_df.iloc[start_idx : end_idx]
        if not future_slice.empty:
            f_c = np.clip(future_slice['carbon_intensity'].values / 800.0, 0, 1)
            f_s = np.clip(future_slice['solar_generation'].values, 0, 1)
            l = len(f_c)
            forecast_vec[0:l] = f_c
            forecast_vec[self.forecast_window:self.forecast_window+l] = f_s

        # Candidate Pool Features (Suspended + Queue)
        candidate_pool = self.suspended_queue + self.queue
        queue_state = []
        for i in range(self.n_candidate_tasks):
            if i < len(candidate_pool):
                t = candidate_pool[i]
                rem_time = (t.deadline - self.current_time_min) / 60.0
                size = t.total_work / 60.0
                waiting_time = (self.current_time_min - t.arrival_time) / 60.0
                priority_norm = t.priority / 2.0
                queue_state.extend([size, rem_time, waiting_time, priority_norm])
            else:
                queue_state.extend([0, 0, 0, 0])

        active_state = []
        for i in range(self.m_active_slots):
            if i < len(self.active_tasks):
                t = self.active_tasks[i]
                rem_work = t.remaining_work / 60.0
                rem_time = (t.deadline - self.current_time_min) / 60.0
                waiting_time = (self.current_time_min - t.arrival_time) / 60.0
                priority_norm = t.priority / 2.0
                active_state.extend([rem_work, rem_time, waiting_time, priority_norm])
            else:
                active_state.extend([0, 0, 0, 0])

        return np.array(env_state + list(forecast_vec) + queue_state + active_state, dtype=np.float32)
