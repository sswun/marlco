"""
HRG Environment Performance Benchmark

比较原版和快速版HRG环境的性能
"""

import time
import numpy as np
from typing import Dict


def benchmark_env(env, num_episodes: int = 100, max_steps: int = 100) -> Dict:
    """
    基准测试环境性能
    
    Args:
        env: 环境实例
        num_episodes: 测试episode数量
        max_steps: 每个episode最大步数
    
    Returns:
        Dict: 性能统计
    """
    total_time = 0
    total_steps = 0
    episode_lengths = []
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_length = 0
        
        for step in range(max_steps):
            # 随机动作
            actions = {}
            for agent_id in env.agent_ids:
                actions[agent_id] = np.random.randint(0, 8)
            
            obs, rewards, dones, infos = env.step(actions)
            episode_length += 1
            total_steps += 1
            
            if all(dones.values()):
                break
        
        episode_lengths.append(episode_length)
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'total_steps': total_steps,
        'num_episodes': num_episodes,
        'avg_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'steps_per_second': total_steps / total_time,
        'episodes_per_minute': (num_episodes / total_time) * 60
    }


def run_benchmark():
    """运行完整基准测试"""
    print("=" * 80)
    print("HRG Environment Performance Benchmark")
    print("=" * 80)
    print()
    
    num_episodes = 100
    max_steps = 200
    
    # 测试原版环境
    print("⏳ 测试原版HRG环境...")
    try:
        from env_hrg_ctde import create_hrg_ctde_env
        env_normal = create_hrg_ctde_env(config_name="normal_ctde", render_mode="")
        
        stats_normal = benchmark_env(env_normal, num_episodes, max_steps)
        env_normal.close()
        
        print("✅ 原版测试完成")
        print(f"   总耗时: {stats_normal['total_time']:.2f}s")
        print(f"   Episode数: {stats_normal['num_episodes']}")
        print(f"   总步数: {stats_normal['total_steps']}")
        print(f"   平均episode长度: {stats_normal['avg_episode_length']:.1f} ± {stats_normal['std_episode_length']:.1f}")
        print(f"   吞吐量: {stats_normal['steps_per_second']:.1f} steps/s")
        print(f"   训练速度: {stats_normal['episodes_per_minute']:.1f} episodes/min")
        print()
    except Exception as e:
        print(f"❌ 原版测试失败: {e}")
        stats_normal = None
        print()
    
    # 测试快速版环境
    print("⏳ 测试快速版HRG环境...")
    try:
        from env_hrg_fast_ctde import create_hrg_fast_ctde_env
        env_fast = create_hrg_fast_ctde_env(difficulty="fast_training")
        
        stats_fast = benchmark_env(env_fast, num_episodes, max_steps)
        env_fast.close()
        
        print("✅ 快速版测试完成")
        print(f"   总耗时: {stats_fast['total_time']:.2f}s")
        print(f"   Episode数: {stats_fast['num_episodes']}")
        print(f"   总步数: {stats_fast['total_steps']}")
        print(f"   平均episode长度: {stats_fast['avg_episode_length']:.1f} ± {stats_fast['std_episode_length']:.1f}")
        print(f"   吞吐量: {stats_fast['steps_per_second']:.1f} steps/s")
        print(f"   训练速度: {stats_fast['episodes_per_minute']:.1f} episodes/min")
        print()
    except Exception as e:
        print(f"❌ 快速版测试失败: {e}")
        stats_fast = None
        print()
    
    # 测试超快速版环境
    print("⏳ 测试超快速版HRG环境...")
    try:
        from env_hrg_fast_ctde import create_hrg_ultra_fast_ctde_env
        env_ultra = create_hrg_ultra_fast_ctde_env()
        
        stats_ultra = benchmark_env(env_ultra, num_episodes, max_steps)
        env_ultra.close()
        
        print("✅ 超快速版测试完成")
        print(f"   总耗时: {stats_ultra['total_time']:.2f}s")
        print(f"   Episode数: {stats_ultra['num_episodes']}")
        print(f"   总步数: {stats_ultra['total_steps']}")
        print(f"   平均episode长度: {stats_ultra['avg_episode_length']:.1f} ± {stats_ultra['std_episode_length']:.1f}")
        print(f"   吞吐量: {stats_ultra['steps_per_second']:.1f} steps/s")
        print(f"   训练速度: {stats_ultra['episodes_per_minute']:.1f} episodes/min")
        print()
    except Exception as e:
        print(f"❌ 超快速版测试失败: {e}")
        stats_ultra = None
        print()
    
    # 性能对比
    print("=" * 80)
    print("性能对比总结")
    print("=" * 80)
    print()
    
    if stats_normal and stats_fast:
        speedup_time = stats_normal['total_time'] / stats_fast['total_time']
        speedup_throughput = stats_fast['steps_per_second'] / stats_normal['steps_per_second']
        speedup_training = stats_fast['episodes_per_minute'] / stats_normal['episodes_per_minute']
        
        print(f"📊 快速版 vs 原版:")
        print(f"   时间加速比: {speedup_time:.2f}x (原版{stats_normal['total_time']:.1f}s → 快速版{stats_fast['total_time']:.1f}s)")
        print(f"   吞吐量提升: {speedup_throughput:.2f}x ({stats_normal['steps_per_second']:.1f} → {stats_fast['steps_per_second']:.1f} steps/s)")
        print(f"   训练速度提升: {speedup_training:.2f}x ({stats_normal['episodes_per_minute']:.1f} → {stats_fast['episodes_per_minute']:.1f} eps/min)")
        print()
    
    if stats_normal and stats_ultra:
        speedup_time = stats_normal['total_time'] / stats_ultra['total_time']
        speedup_throughput = stats_ultra['steps_per_second'] / stats_normal['steps_per_second']
        speedup_training = stats_ultra['episodes_per_minute'] / stats_normal['episodes_per_minute']
        
        print(f"📊 超快速版 vs 原版:")
        print(f"   时间加速比: {speedup_time:.2f}x (原版{stats_normal['total_time']:.1f}s → 超快速{stats_ultra['total_time']:.1f}s)")
        print(f"   吞吐量提升: {speedup_throughput:.2f}x ({stats_normal['steps_per_second']:.1f} → {stats_ultra['steps_per_second']:.1f} steps/s)")
        print(f"   训练速度提升: {speedup_training:.2f}x ({stats_normal['episodes_per_minute']:.1f} → {stats_ultra['episodes_per_minute']:.1f} eps/min)")
        print()
    
    # 环境维度对比
    print("=" * 80)
    print("环境维度对比")
    print("=" * 80)
    print()
    
    if stats_normal:
        from env_hrg_ctde import create_hrg_ctde_env
        env = create_hrg_ctde_env(config_name="normal_ctde", render_mode="")
        info = env.get_env_info()
        print(f"原版环境:")
        print(f"   观测维度: {info.get('obs_dims', 'N/A')}")
        print(f"   全局状态维度: {info.get('global_state_dim', 'N/A')}")
        print(f"   智能体数: {info.get('n_agents', 'N/A')}")
        env.close()
        print()
    
    if stats_fast:
        from env_hrg_fast_ctde import create_hrg_fast_ctde_env
        env = create_hrg_fast_ctde_env(difficulty="fast_training")
        info = env.get_env_info()
        print(f"快速版环境:")
        print(f"   观测维度: {info.get('obs_dims', 'N/A')}")
        print(f"   全局状态维度: {info.get('global_state_dim', 'N/A')}")
        print(f"   智能体数: {info.get('n_agents', 'N/A')}")
        env.close()
        print()
    
    print("=" * 80)
    print("基准测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
