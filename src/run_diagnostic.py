import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os

# 1. 确保环境路径正确
if 'PYTHONPATH' not in os.environ or '.' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] = f".{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

from src.data.data_loader import SyntheticDataLoader
from src.envs.diagnostic_env import DiagnosticDataCenterEnv
from src.core.diagnostic_ai import DiagnosticPPOAgent

# 🌟 引入真实的 Baseline 智能体
from src.agents.baselines.fifo import FIFOAgent
from src.agents.baselines.rule_based import RuleBasedAgent
from src.agents.baselines.max_capacity import MaxCapacityAgent

# 引入绘图函数
from src.utils.plotting import (
    plot_results,
    plot_stability_dynamics,
    plot_compliance_distribution,
    plot_sla_cdf,
    plot_power_comparison
)


class SimplePPOBuffer:
    def __init__(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def clear(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state);
        self.actions.append(action);
        self.log_probs.append(log_prob)
        self.rewards.append(reward);
        self.dones.append(done);
        self.values.append(value)


def run_eval_loop(env, agent, agent_name, is_ppo=True):
    """通用评估循环，用于收集 AI 或 Baseline 的真实表现"""
    print(f"--- Running Real Evaluation for: {agent_name} ---")
    state, _ = env.reset()
    done = False

    metrics = {
        'step': [], 'electricity_costs': [], 'carbon_costs': [], 'total_energy': [],
        'renewable_usage': [], 'latencies': {0: [], 1: [], 2: []},
        'soc': [], 'carbon_intensity': [], 'total_power': [],
        'queue_backlog': [], 'lyapunov_penalty': [], 'requested_load': []
    }

    while not done:
        if is_ppo:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.get_action(state_tensor)
            action_env = action.numpy()[0]
        else:
            # Baseline 直接获取动作
            action_env = agent.get_action(state)

        next_state, reward, term, trunc, info = env.step(action_env)
        done = term or trunc

        # 记录数据
        metrics['step'].append(env.current_step)
        metrics['electricity_costs'].append(info.get('cost', 0))
        metrics['carbon_costs'].append(info.get('carbon', 0))
        metrics['total_energy'].append(info.get('step_total_energy', 0))
        metrics['renewable_usage'].append(info.get('step_renewable_energy', 0))
        metrics['carbon_intensity'].append(info.get('carbon_intensity', 0))
        metrics['total_power'].append(info.get('total_power_kw', 0))
        metrics['queue_backlog'].append(info.get('queue_backlog', 0))
        metrics['lyapunov_penalty'].append(info.get('lyapunov_penalty', 0))
        metrics['soc'].append(info.get('soc', 0))

        req_load = info.get('base_load', 0) + info.get('load_gold', 0) + \
                   info.get('load_silver', 0) + info.get('load_bronze', 0)
        metrics['requested_load'].append(req_load)
        state = next_state

    # 统计任务指标
    metrics['tasks_processed'] = len(env.completed_tasks)
    metrics['tasks_missed'] = len(env.failed_tasks)
    for t in env.completed_tasks:
        metrics['latencies'][t.priority].append((t.start_time - t.arrival_time) / 60.0)

    # 填补空缺类别防止画图报错
    for prio in [0, 1, 2]:
        if len(metrics['latencies'][prio]) == 0: metrics['latencies'][prio].append(0.0)

    return metrics


def run_diagnostic_training(args):
    print("=== STARTING DIAGNOSTIC PROTOCOL (REAL BASELINE MODE) ===")

    # 1. Setup Data & Env
    days = getattr(args, 'days', 30)
    data_loader = SyntheticDataLoader(days=days)  # Use days from args
    env_config = {
        'n_candidate_tasks': 20,
        'sla_penalty_multiplier': getattr(args, 'sla_penalty', getattr(args, 'sla_penalty_multiplier', 5.0)),
        'weight_energy': getattr(args, 'energy_weight', 10.0),
        'curriculum_phase': 1,
        'use_solar': getattr(args, 'use_solar', True),
        'use_wind': getattr(args, 'use_wind', True),
        'lyapunov_weight': getattr(args, 'lyapunov_weight', 0.01)  # 使用命令行传入的权重
    }
    env = DiagnosticDataCenterEnv(data_loader, env_config)

    # 2. Init AI Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DiagnosticPPOAgent(state_dim, action_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    buffer = SimplePPOBuffer()

    # 3. 训练阶段 (Training Loop)
    print(f"--- Training AI for {args.epochs} epochs ---")
    update_interval, gamma, clip_ratio, global_step = 200, 0.99, 0.2, 0
    for epoch in range(args.epochs):
        state, _ = env.reset();
        done = False;
        ep_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _ = agent.get_action(state_tensor)
                value = agent.critic(state_tensor)
            action_np = action.numpy()[0]
            next_state, reward, term, trunc, info = env.step(action_np)
            done = term or trunc
            buffer.store(state, action_np, log_prob.item(), reward, done, value.item())
            state = next_state;
            ep_reward += reward;
            global_step += 1

            if global_step % update_interval == 0:
                # PPO Update Logic (保持不变)
                with torch.no_grad():
                    next_val = agent.critic(torch.FloatTensor(next_state).unsqueeze(0)).item()
                returns, gae, vals = [], 0, buffer.values + [next_val]
                for i in reversed(range(len(buffer.rewards))):
                    delta = buffer.rewards[i] + gamma * vals[i + 1] * (1 - buffer.dones[i]) - vals[i]
                    gae = delta + gamma * 0.95 * (1 - buffer.dones[i]) * gae
                    returns.insert(0, gae + vals[i])
                b_s, b_a, b_lp, b_ret = torch.FloatTensor(np.array(buffer.states)), torch.FloatTensor(
                    np.array(buffer.actions)), \
                    torch.FloatTensor(buffer.log_probs), torch.FloatTensor(returns)
                b_adv = (b_ret - torch.FloatTensor(buffer.values))
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                for _ in range(3):
                    new_lp, new_v, ent = agent.evaluate(b_s, b_a)
                    ratio = (new_lp - b_lp).exp()
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_adv
                    loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(new_v.squeeze(),
                                                                                b_ret) - 0.01 * ent.mean()
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step()
                buffer.clear()
        print(f"Epoch {epoch + 1}/{args.epochs} Reward: {ep_reward:.2f}")

    # 4. 评估阶段 (Evaluation)
    all_metrics = {}

    # 运行 AI 评估
    ai_name = "UTCAS-PPO (Ours)"
    all_metrics[ai_name] = run_eval_loop(env, agent, ai_name, is_ppo=True)

    # 🌟 运行选择的真实 Baseline 评估
    baseline_map = {
        'fifo': FIFOAgent,
        'rule_based': RuleBasedAgent,
        'max_capacity': MaxCapacityAgent
    }
    b_class = baseline_map.get(args.baseline.lower(), RuleBasedAgent)
    # 大多数 Baseline 需要知道观察空间和任务池大小
    real_baseline_agent = b_class(state_dim, env.n_candidate_tasks, env.m_active_slots)

    b_name = f"{args.baseline.upper()} (Baseline)"
    all_metrics[b_name] = run_eval_loop(env, real_baseline_agent, b_name, is_ppo=False)

    # 5. 画图
    print("\n=== GENERATING PLOTS WITH REAL DATA ===")
    save_dir = "outputs/experiment_results"
    os.makedirs(save_dir, exist_ok=True)

    df_hist = pd.DataFrame({
        'step': all_metrics[ai_name]['step'],
        'queue_backlog': all_metrics[ai_name]['queue_backlog'],
        'lyapunov_penalty': all_metrics[ai_name]['lyapunov_penalty']
    })

    try:
        plot_stability_dynamics(df_hist, save_dir=save_dir, suffix="_diagnostic")
        plot_results(all_metrics, save_dir=save_dir)
        plot_compliance_distribution(all_metrics, save_dir=save_dir)
        plot_sla_cdf(all_metrics, save_dir=save_dir)
        plot_power_comparison(all_metrics, save_dir=save_dir, carbon_p75=600.0)
        print(f"\n✅ 成功！已对比 {ai_name} 与 {b_name}")
    except Exception as e:
        print(f"\n❌ 画图失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--baseline", type=str, default="max_capacity", choices=["fifo", "rule_based", "rule_based"])
    parser.add_argument("--lyapunov_weight", type=float, default=0.01)
    parser.add_argument("--sla_penalty_multiplier", type=float, default=100.0)
    args = parser.parse_args()
    run_diagnostic_training(args)