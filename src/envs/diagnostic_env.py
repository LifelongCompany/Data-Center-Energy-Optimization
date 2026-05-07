import numpy as np
from src.envs.env import DataCenterEnv

class DiagnosticDataCenterEnv(DataCenterEnv):
    """
    Diagnostic Environment for 'Phase 1 Isolation' testing.
    Forces Energy/Carbon weights to 0.0 to test pure Throughput capability.
    Logs internal metrics to stdout for debugging.
    """
    def __init__(self, data_loader, env_config, decoupled=False):
        # Allow energy and carbon weights to be passed naturally or set them to sensible defaults
        # We no longer force them to 0.0 in isolation mode if we want the AI to care about cost
        env_config['weight_energy'] = env_config.get('weight_energy', 10.0)
        env_config['weight_carbon'] = env_config.get('weight_carbon', 5.0)
        env_config['weight_sla'] = env_config.get('weight_sla', 1.0)
        # Use provided lyapunov_weight, defaulting to 0.01 for sensitivity without overload
        env_config['lyapunov_weight'] = env_config.get('lyapunov_weight', 0.01)

        super().__init__(data_loader, env_config, decoupled)

        print("\n[DiagnosticEnv] INITIALIZED in ISOLATION MODE.")
        print(f"[DiagnosticEnv] Weights -> Energy: {self.weight_energy}, Carbon: {self.weight_carbon}, SLA: {self.weight_sla}")

    def step(self, action):
        # Intercept Action for Analysis
        if self.decoupled:
            cont_actions, disc_actions = action
            admit_rate = np.mean(disc_actions)
            phys_mean = np.mean(cont_actions[1:]) # Skip battery, check resource slots
        else:
            # Slicing based on your env definition:
            # 0: Battery, 1..N: Admit, N+1..M: Resource
            admit_actions = action[1 : 1 + self.n_candidate_tasks]
            resource_actions = action[1 + self.n_candidate_tasks :]

            # Admit > 0.5 means accept
            admit_rate = np.mean(admit_actions > 0.0)
            # Resource is [-1, 1], mapped to [0, 1] internally, but let's see raw output
            phys_mean = np.mean(resource_actions)

        # Execute Step
        obs, reward, term, trunc, info = super().step(action)

        # Probes: Print diagnostics for the first 50 steps of the first few episodes
        # or if specific anomalies occur.
        if self.current_step < 20:
            print(f"\n[Step {self.current_step} DIAGNOSTIC]")
            print(f"  > Action Stats: Admit Rate={admit_rate:.2f}, Avg Resource Output={phys_mean:.2f}")

            # Reward Scaling Analysis
            # We access the raw components calculated inside step() via 'info' if available,
            # but standard env might not expose UN-normalized SLA pain easily.
            # Let's approximate reconstruction or check info if you added it.
            # Based on your env code, info['penalty'] is the SLA penalty cost.

            raw_sla = info.get('penalty', 0.0)
            raw_elec = info.get('cost', 0.0)

            print(f"  > Cost Magnitude: Elec=${raw_elec:.4f} vs SLA=${raw_sla:.4f}")
            if raw_elec > 0 and raw_sla > 0:
                ratio = raw_sla / raw_elec
                print(f"  > SLA is {ratio:.1f}x times more expensive than Electricity!")

        return obs, reward, term, trunc, info
