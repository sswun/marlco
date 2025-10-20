"""
快速测试HRG Fast环境
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_fast_env():
    """测试快速版环境"""
    print("=" * 60)
    print("测试HRG Fast环境")
    print("=" * 60)
    print()
    
    try:
        from env_hrg_fast_ctde import create_hrg_fast_ctde_env
        
        # 创建环境
        print("✅ 创建fast_training环境...")
        env = create_hrg_fast_ctde_env(difficulty="fast_training")
        
        # 检查环境信息
        info = env.get_env_info()
        print(f"   智能体数: {info['n_agents']}")
        print(f"   观测维度: {info['obs_dims']}")
        print(f"   动作维度: {info['act_dims']}")
        print(f"   全局状态维度: {info['global_state_dim']}")
        print()
        
        # 运行几个episode
        print("✅ 运行测试episodes...")
        for ep in range(3):
            obs = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(100):
                actions = {agent_id: 0 for agent_id in env.agent_ids}
                obs, rewards, dones, infos = env.step(actions)
                total_reward += sum(rewards.values())
                steps += 1
                
                if all(dones.values()):
                    break
            
            print(f"   Episode {ep+1}: {steps}步, 总奖励={total_reward:.2f}")
        
        env.close()
        print()
        print("✅ 快速版测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_ultra_fast_env():
    """测试超快速版环境"""
    print()
    print("=" * 60)
    print("测试HRG Ultra Fast环境")
    print("=" * 60)
    print()
    
    try:
        from env_hrg_fast_ctde import create_hrg_ultra_fast_ctde_env
        
        # 创建环境
        print("✅ 创建ultra_fast环境...")
        env = create_hrg_ultra_fast_ctde_env()
        
        # 检查环境信息
        info = env.get_env_info()
        print(f"   智能体数: {info['n_agents']}")
        print(f"   观测维度: {info['obs_dims']}")
        print(f"   全局状态维度: {info['global_state_dim']}")
        print()
        
        # 运行测试
        print("✅ 运行测试episodes...")
        obs = env.reset()
        
        for step in range(10):
            actions = {agent_id: step % 8 for agent_id in env.agent_ids}
            obs, rewards, dones, infos = env.step(actions)
            
            if all(dones.values()):
                break
        
        env.close()
        print("✅ 超快速版测试通过!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success1 = test_fast_env()
    success2 = test_ultra_fast_env()
    
    print()
    print("=" * 60)
    if success1 and success2:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败")
    print("=" * 60)
