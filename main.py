import argparse
import sys
import os

# 自动把当前目录加入系统路径，防止 Windows 下 Python 找不到 src 文件夹里的模块
if 'PYTHONPATH' not in os.environ or '.' not in os.environ['PYTHONPATH']:
    os.environ['PYTHONPATH'] = f".{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

# 引入另外三个文件的核心函数
from src.verification_script import verify_environment
from src.run_diagnostic import run_diagnostic_training
from src.run_ablation import run_ablation


def main():
    # 建立一个命令行参数解析器，让你可以在终端里灵活控制运行哪些模块
    parser = argparse.ArgumentParser(description="Data Center RL Orchestrator (数据中心能源调度总管)")

    # 核心控制开关
    parser.add_argument("--mode", type=str, choices=["run_all", "verification", "diagnostic", "ablation"],
                        default="run_all", help="选择要运行的模式 (Mode to run)")

    # 智能体选择
    parser.add_argument("--baseline", type=str, choices=["fifo", "rule_based", "max_capacity"],
                        default="rule_based", help="对比的 Baseline 智能体 (Baseline to compare against)")

    # 能源服务定制开关
    parser.add_argument("--use_solar", action="store_true", help="是否启用太阳能 (Enable solar energy)")
    parser.add_argument("--use_wind", action="store_true", help="是否启用风能 (Enable wind energy)")
    parser.add_argument("--use_battery", action="store_true", help="是否启用电池储能 (Enable battery storage)")

    # 物理约束与奖励权重调整
    parser.add_argument("--lyapunov_weight", type=float, default=0.01, help="队列稳定性(李雅普诺夫)的权重 (Queue stability weight)")
    parser.add_argument("--sla_penalty", type=float, default=5.0, help="SLA 违约惩罚倍率 (SLA penalty multiplier)")
    parser.add_argument("--energy_weight", type=float, default=10.0, help="能源与碳排放成本的权重 (Energy & Carbon weight)")

    # 模拟强度参数
    parser.add_argument("--days", type=int, default=30, help="模拟的数据天数 (Number of simulation days)")
    parser.add_argument("--epochs", type=int, default=10, help="诊断训练的轮数 (Training Epochs)")

    args = parser.parse_args()

    # ==========================================
    # Phase 1: 安全检查
    # ==========================================
    if args.mode in ["run_all", "verification"]:
        print("\n" + "🚀" * 25)
        print("▶ PHASE 1: ENVIRONMENT VERIFICATION (环境安全检查)")
        print("🚀" * 25)
        verify_environment()

    # ==========================================
    # Phase 2: 核心训练与画图
    # ==========================================
    if args.mode in ["run_all", "diagnostic"]:
        print("\n" + "🧠" * 25)
        print("▶ PHASE 2: DIAGNOSTIC TRAINING (AI 核心训练与图表生成)")
        print("🧠" * 25)
        # 把接收到的参数传给你刚才写好的 run_diagnostic_training 函数
        run_diagnostic_training(args)

    # ==========================================
    # Phase 3: 消融实验
    # ==========================================
    if args.mode in ["run_all", "ablation"]:
        print("\n" + "🔬" * 25)
        print("▶ PHASE 3: ABLATION STUDY (消融实验对照组)")
        print("🔬" * 25)
        run_ablation()

    if args.mode == "run_all":
        print("\n" + "=" * 50)
        print("✅ ALL PHASES COMPLETED SUCCESSFULLY (所有阶段完美收工)")
        print("=" * 50)


if __name__ == "__main__":
    main()