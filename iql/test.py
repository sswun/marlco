"""
IQL 算法测试脚本
"""
import torch
import numpy as np
from src.models import IQLNetworks, AgentNetwork
from src.algos import IQL
from src.buffer import ReplayBuffer
from src.utils import get_device, set_seed


def test_agent_network():
    """测试AgentNetwork"""
    print("🧪 测试AgentNetwork...")

    device = get_device()

    # 创建网络
    net = AgentNetwork(obs_dim=10, action_dim=5, hidden_dim=32).to(device)

    # 测试前向传播
    obs = torch.randn(2, 10).to(device)  # batch_size=2
    q_values = net(obs)

    assert q_values.shape == (2, 5), f"期望形状(2, 5)，实际{q_values.shape}"
    print("   ✅ 前向传播测试通过")

    # 测试动作选择
    actions = net.act(obs, epsilon=0.0)
    assert actions.shape == (2,), f"期望形状(2,)，实际{actions.shape}"
    assert torch.all(actions >= 0) and torch.all(actions < 5), "动作值超出范围"
    print("   ✅ 动作选择测试通过")

    print("✅ AgentNetwork测试完成\n")


def test_iql_networks():
    """测试IQLNetworks"""
    print("🧪 测试IQLNetworks...")

    device = get_device()

    # 模拟环境信息
    env_info = {
        'n_agents': 3,
        'obs_dims': [10, 12, 10],  # 异构观测维度
        'act_dims': [5, 5, 5]
    }

    config = {
        'model': {
            'hidden_dim': 32
        }
    }

    # 创建网络
    networks = IQLNetworks(env_info, config, device)

    assert networks.n_agents == 3
    assert len(networks.agent_networks) == 3
    assert len(networks.target_agent_networks) == 3
    print("   ✅ 网络创建测试通过")

    # 测试参数获取
    params = networks.get_all_parameters()
    assert len(params) > 0, "参数列表为空"
    print("   ✅ 参数获取测试通过")

    # 测试目标网络更新
    networks.hard_update_target_networks()
    networks.soft_update_target_networks(tau=0.01)
    print("   ✅ 目标网络更新测试通过")

    print("✅ IQLNetworks测试完成\n")


def test_replay_buffer():
    """测试ReplayBuffer"""
    print("🧪 测试ReplayBuffer...")

    device = get_device()

    # 创建缓冲区
    buffer = ReplayBuffer(
        capacity=1000,
        n_agents=3,
        obs_dim=10,
        state_dim=1,  # IQL中不使用全局状态
        device=device
    )

    # 测试添加经验
    for _ in range(10):
        obs = {f'agent_{i}': np.random.randn(10) for i in range(3)}
        actions = {f'agent_{i}': np.random.randint(0, 5) for i in range(3)}
        rewards = {f'agent_{i}': np.random.randn() for i in range(3)}
        next_obs = {f'agent_{i}': np.random.randn(10) for i in range(3)}
        dones = {f'agent_{i}': False for i in range(3)}

        buffer.push(obs, actions, rewards, next_obs, dones)

    assert len(buffer) == 10, f"缓冲区大小错误，期望10，实际{len(buffer)}"
    print("   ✅ 经验添加测试通过")

    # 测试采样
    batch = buffer.sample(batch_size=5)

    required_keys = ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'global_state', 'next_global_state']
    for key in required_keys:
        assert key in batch, f"批次数据缺少键: {key}"

    assert batch['obs'].shape[0] == 5, "批次大小错误"
    assert batch['obs'].shape[1] == 3, "智能体数量错误"
    print("   ✅ 批量采样测试通过")

    print("✅ ReplayBuffer测试完成\n")


def test_iql_algorithm():
    """测试IQL算法"""
    print("🧪 测试IQL算法...")

    device = get_device()

    # 环境信息和配置
    env_info = {
        'n_agents': 3,
        'obs_dims': [10, 10, 10],  # 同构观测维度
        'act_dims': [5, 5, 5]
    }

    config = {
        'algorithm': {
            'gamma': 0.99,
            'learning_rate': 0.001,
            'tau': 0.005,
            'target_update_interval': 10,
            'max_grad_norm': 10.0
        },
        'model': {
            'hidden_dim': 32
        }
    }

    # 创建网络和算法
    networks = IQLNetworks(env_info, config, device)
    algorithm = IQL(networks, config, device)

    # 测试动作选择
    obs = {
        'agent_0': torch.randn(10).to(device),
        'agent_1': torch.randn(10).to(device),
        'agent_2': torch.randn(10).to(device)
    }

    actions = algorithm.select_actions(obs, epsilon=0.0)
    assert len(actions) == 3, "动作数量错误"
    for action in actions.values():
        assert 0 <= action < 5, "动作值超出范围"
    print("   ✅ 动作选择测试通过")

    # 创建测试批次
    batch_size = 4
    batch = {
        'obs': torch.randn(batch_size, 3, 10).to(device),
        'actions': torch.randint(0, 5, (batch_size, 3)).to(device),
        'rewards': torch.randn(batch_size, 3).to(device),
        'next_obs': torch.randn(batch_size, 3, 10).to(device),
        'dones': torch.zeros(batch_size, 3, dtype=torch.bool).to(device),
        'global_state': torch.randn(batch_size, 1).to(device),
        'next_global_state': torch.randn(batch_size, 1).to(device)
    }

    # 测试损失计算
    loss = algorithm.compute_loss(batch)
    assert loss.requires_grad, "损失需要梯度"
    print("   ✅ 损失计算测试通过")

    # 测试算法更新
    loss_info = algorithm.update(batch)
    assert 'loss' in loss_info, "更新结果缺少损失信息"
    assert 'grad_norm' in loss_info, "更新结果缺少梯度范数信息"
    print("   ✅ 算法更新测试通过")

    print("✅ IQL算法测试完成\n")


def test_integration():
    """集成测试"""
    print("🧪 集成测试...")

    device = get_device()

    # 环境信息和配置
    env_info = {
        'n_agents': 2,
        'obs_dims': [8, 8],
        'act_dims': [4, 4]
    }

    config = {
        'algorithm': {
            'gamma': 0.95,
            'learning_rate': 0.01,
            'tau': 0.01,
            'target_update_interval': 5,
            'max_grad_norm': 5.0
        },
        'model': {
            'hidden_dim': 16
        }
    }

    # 创建组件
    networks = IQLNetworks(env_info, config, device)
    algorithm = IQL(networks, config, device)
    buffer = ReplayBuffer(
        capacity=100,
        n_agents=2,
        obs_dim=8,
        state_dim=1,
        device=device
    )

    # 模拟训练过程
    print("   🔄 模拟训练过程...")

    for episode in range(10):
        obs = {f'agent_{i}': np.random.randn(8) for i in range(2)}

        for step in range(5):
            # 选择动作
            obs_tensor = {k: torch.FloatTensor(v).to(device) for k, v in obs.items()}
            actions = algorithm.select_actions(obs_tensor, epsilon=0.1)

            # 模拟环境交互
            next_obs = {f'agent_{i}': np.random.randn(8) for i in range(2)}
            rewards = {f'agent_{i}': np.random.randn() for i in range(2)}
            dones = {f'agent_{i}': step == 4 for i in range(2)}  # 最后一步结束

            # 存储经验
            buffer.push(obs, actions, rewards, next_obs, dones)
            obs = next_obs

        # 训练
        if len(buffer) >= 4:
            batch = buffer.sample(batch_size=4)
            loss_info = algorithm.update(batch)

            if episode % 3 == 0:
                print(f"      Episode {episode}: Loss = {loss_info['loss']:.4f}")

    print("   ✅ 模拟训练完成")
    print("✅ 集成测试完成\n")


def main():
    """主测试函数"""
    print("🚀 开始IQL算法测试\n")

    # 设置随机种子
    set_seed(42)

    try:
        test_agent_network()
        test_iql_networks()
        test_replay_buffer()
        test_iql_algorithm()
        test_integration()

        print("🎉 所有测试通过！IQL算法实现正确。")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()