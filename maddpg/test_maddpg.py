"""
MADDPG 快速测试脚本
用于验证实现是否正确
"""
import torch
import numpy as np
from src.utils import load_config, set_seed, get_device
from src.envs import EnvWrapper
from src.models import MADDPGNetworks
from src.algos import MADDPG
from src.buffer import ReplayBuffer
from src.trainer import MADDPGTrainer


def test_basic_components():
    """测试基本组件"""
    print("="*60)
    print("测试1: 基本组件初始化")
    print("="*60)
    
    # 设置随机种子
    set_seed(42)
    
    # 加载配置
    config = load_config('config.yaml')
    device = get_device()
    
    print(f"✅ 配置加载成功")
    print(f"✅ 设备: {device}")
    
    # 创建环境
    env = EnvWrapper(config)
    env_info = env.get_env_info()
    
    print(f"✅ 环境创建成功: {env_info['n_agents']}个智能体")
    
    # 创建网络
    networks = MADDPGNetworks(env_info, config, device)
    print(f"✅ 网络创建成功")
    
    # 创建算法
    algorithm = MADDPG(networks, config, device)
    print(f"✅ 算法创建成功")
    
    # 创建缓冲区
    buffer = ReplayBuffer(
        capacity=100,
        n_agents=env_info['n_agents'],
        obs_dim=env_info['obs_dims'][0],
        state_dim=0,
        device=device
    )
    print(f"✅ 缓冲区创建成功")
    
    env.close()
    print("\n✅ 基本组件测试通过!\n")


def test_forward_pass():
    """测试前向传播"""
    print("="*60)
    print("测试2: 前向传播")
    print("="*60)
    
    set_seed(42)
    config = load_config('config.yaml')
    device = get_device()
    
    # 创建环境和网络
    env = EnvWrapper(config)
    env_info = env.get_env_info()
    networks = MADDPGNetworks(env_info, config, device)
    algorithm = MADDPG(networks, config, device)
    
    # 重置环境
    obs, _ = env.reset()
    
    # 转换观测为tensor
    obs_tensor = {}
    for agent_id, agent_obs in obs.items():
        obs_tensor[agent_id] = torch.FloatTensor(agent_obs).to(device)
    
    # 选择动作
    actions = algorithm.select_actions(obs_tensor, noise_scale=0.1)
    
    print(f"✅ 动作选择成功: {actions}")
    
    # 执行动作
    next_obs, rewards, dones, _ = env.step(actions)
    
    print(f"✅ 环境步进成功")
    print(f"   奖励: {rewards}")
    print(f"   终止: {dones}")
    
    env.close()
    print("\n✅ 前向传播测试通过!\n")


def test_training_step():
    """测试训练步骤"""
    print("="*60)
    print("测试3: 训练步骤")
    print("="*60)
    
    set_seed(42)
    config = load_config('config.yaml')
    device = get_device()
    
    # 创建环境和网络
    env = EnvWrapper(config)
    env_info = env.get_env_info()
    networks = MADDPGNetworks(env_info, config, device)
    algorithm = MADDPG(networks, config, device)
    
    # 创建缓冲区
    buffer = ReplayBuffer(
        capacity=100,
        n_agents=env_info['n_agents'],
        obs_dim=env_info['obs_dims'][0],
        state_dim=0,
        device=device
    )
    
    # 收集一些经验
    print("收集经验...")
    for _ in range(10):
        obs, _ = env.reset()
        
        for step in range(20):
            # 转换观测
            obs_tensor = {}
            for agent_id, agent_obs in obs.items():
                obs_tensor[agent_id] = torch.FloatTensor(agent_obs).to(device)
            
            # 选择动作
            actions = algorithm.select_actions(obs_tensor, noise_scale=0.1)
            
            # 执行动作
            next_obs, rewards, dones, _ = env.step(actions)
            
            # 存储经验
            buffer.push(obs, actions, rewards, next_obs, dones,
                       np.zeros(1), np.zeros(1))
            
            obs = next_obs
            
            if all(dones.values()):
                break
    
    print(f"✅ 收集了 {len(buffer)} 条经验")
    
    # 采样并训练
    if len(buffer) >= 64:
        batch = buffer.sample(64)
        loss_info = algorithm.update(batch)
        
        print(f"✅ 训练步骤成功")
        print(f"   Critic损失: {loss_info['critic_loss']:.4f}")
        print(f"   Actor损失: {loss_info['actor_loss']:.4f}")
        print(f"   总损失: {loss_info['total_loss']:.4f}")
    
    env.close()
    print("\n✅ 训练步骤测试通过!\n")


def test_full_training():
    """测试完整训练流程（少量episodes）"""
    print("="*60)
    print("测试4: 完整训练流程 (10 episodes)")
    print("="*60)
    
    set_seed(42)
    
    # 修改配置为快速测试
    config = load_config('config.yaml')
    config['training']['total_episodes'] = 10
    config['training']['warmup_episodes'] = 2
    config['training']['eval_interval'] = 5
    config['training']['save_interval'] = 999999  # 不保存
    
    # 创建训练器
    trainer = MADDPGTrainer(config)
    
    # 训练
    episode_rewards = trainer.train()
    
    print(f"\n✅ 训练完成!")
    print(f"   完成episodes: {len(episode_rewards)}")
    print(f"   平均奖励: {np.mean(episode_rewards):.2f}")
    
    trainer.close()
    print("\n✅ 完整训练流程测试通过!\n")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("MADDPG 实现测试")
    print("="*60 + "\n")
    
    try:
        # 测试1: 基本组件
        test_basic_components()
        
        # 测试2: 前向传播
        test_forward_pass()
        
        # 测试3: 训练步骤
        test_training_step()
        
        # 测试4: 完整训练
        test_full_training()
        
        print("="*60)
        print("🎉 所有测试通过!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
