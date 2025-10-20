"""
Data Loader for MARL Training Results
Robust JSON parsing and data extraction from training checkpoints
"""
import json
import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingDataLoader:
    """Load and parse training data from various algorithms and environments"""
    
    def __init__(self, data_store_root: str = "../data_store"):
        """
        Initialize data loader
        
        Args:
            data_store_root: Root directory of data storage
        """
        self.data_store_root = Path(data_store_root)
        self.algorithms = ["QMIX", "IQL", "VDN", "COMA", "MADDPG", "MAPPO"]
        self.environments = [
            "CM_hard", "DEM_hard", "DEM_normal", 
            "HRG_ultrafast", "MSFS_hard", "MSFS_normal",
            "multiwalker", "simple_crypto", "simple_spread"
        ]
        
    def find_json_files(self, algorithm: str, environment: str) -> List[Path]:
        """
        Find all JSON training data files for given algorithm and environment
        
        Args:
            algorithm: Algorithm name (QMIX, IQL, VDN, COMA)
            environment: Environment name
            
        Returns:
            List of JSON file paths
        """
        checkpoint_dir = self.data_store_root / algorithm / environment / "checkpoints"
        
        if not checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return []
        
        # Find all JSON files
        json_files = list(checkpoint_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {checkpoint_dir}")
            
        return json_files
    
    def load_json_robust(self, file_path: Path) -> Optional[Dict]:
        """
        Robustly load JSON file, handling corrupted or malformed data
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON dict or None if failed
        """
        try:
            # Method 1: Standard JSON loading
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Standard JSON decode failed for {file_path}: {e}")
            
            # Method 2: Try to fix common issues and extract valid data
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to extract episode_rewards array using regex
                rewards_match = re.search(
                    r'"episode_rewards"\s*:\s*\[([\s\S]*?)\]',
                    content,
                    re.MULTILINE
                )
                
                if rewards_match:
                    rewards_str = rewards_match.group(1)
                    # Extract numbers (handle negative numbers and decimals)
                    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', rewards_str)
                    episode_rewards = [float(x) for x in numbers]
                    
                    logger.info(f"Extracted {len(episode_rewards)} episode rewards from {file_path.name}")
                    
                    return {
                        "metrics": {
                            "episode_rewards": episode_rewards
                        },
                        "config": {},
                        "_extracted": True  # Mark as extracted
                    }
                else:
                    logger.error(f"Could not extract episode_rewards from {file_path}")
                    return None
                    
            except Exception as e2:
                logger.error(f"Failed to extract data from {file_path}: {e2}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {e}")
            return None
    
    def extract_rewards(self, data: Dict) -> List[float]:
        """
        Extract episode rewards from loaded data
        
        Args:
            data: Loaded JSON data
            
        Returns:
            List of episode rewards
        """
        try:
            # Standard format
            if "metrics" in data and "episode_rewards" in data["metrics"]:
                rewards = data["metrics"]["episode_rewards"]
                
                # Convert to float and filter invalid values
                valid_rewards = []
                for r in rewards:
                    if isinstance(r, (int, float)) and not np.isnan(r) and not np.isinf(r):
                        valid_rewards.append(float(r))
                
                return valid_rewards
            
            # Alternative format: direct list
            elif "episode_rewards" in data:
                rewards = data["episode_rewards"]
                valid_rewards = []
                for r in rewards:
                    if isinstance(r, (int, float)) and not np.isnan(r) and not np.isinf(r):
                        valid_rewards.append(float(r))
                return valid_rewards
            
            else:
                logger.warning("No episode_rewards found in data")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting rewards: {e}")
            return []
    
    def load_algorithm_environment_data(
        self, 
        algorithm: str, 
        environment: str,
        select_latest: bool = True
    ) -> Optional[List[float]]:
        """
        Load training data for specific algorithm and environment
        
        Args:
            algorithm: Algorithm name
            environment: Environment name
            select_latest: If True, use the latest JSON file; otherwise use the longest run
            
        Returns:
            List of episode rewards or None if not found
        """
        json_files = self.find_json_files(algorithm, environment)
        
        if not json_files:
            return None
        
        # Load all valid data files
        valid_data = []
        for json_file in json_files:
            data = self.load_json_robust(json_file)
            if data:
                rewards = self.extract_rewards(data)
                if rewards:
                    valid_data.append({
                        'file': json_file,
                        'rewards': rewards,
                        'length': len(rewards)
                    })
        
        if not valid_data:
            logger.warning(f"No valid data found for {algorithm} - {environment}")
            return None
        
        # Select which data to use
        if select_latest:
            # Sort by file modification time
            valid_data.sort(key=lambda x: x['file'].stat().st_mtime, reverse=True)
            selected = valid_data[0]
        else:
            # Select the longest run
            valid_data.sort(key=lambda x: x['length'], reverse=True)
            selected = valid_data[0]
        
        logger.info(
            f"Loaded {selected['length']} episodes for {algorithm} - {environment} "
            f"from {selected['file'].name}"
        )
        
        return selected['rewards']
    
    def load_all_data(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Load all available training data
        
        Returns:
            Nested dict: {algorithm: {environment: [rewards]}}
        """
        all_data = {}
        
        for algorithm in self.algorithms:
            all_data[algorithm] = {}
            
            for environment in self.environments:
                rewards = self.load_algorithm_environment_data(algorithm, environment)
                
                if rewards is not None:
                    all_data[algorithm][environment] = rewards
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Data Loading Summary")
        logger.info("="*60)
        for algo in all_data:
            logger.info(f"{algo}: {len(all_data[algo])} environments loaded")
            for env in all_data[algo]:
                logger.info(f"  - {env}: {len(all_data[algo][env])} episodes")
        logger.info("="*60)
        
        return all_data


def compute_statistics(rewards: List[float], window: int = 100) -> Dict:
    """
    Compute statistics for reward sequence
    
    Args:
        rewards: List of episode rewards
        window: Window size for smoothing
        
    Returns:
        Dictionary of statistics
    """
    if not rewards:
        return {}
    
    rewards_array = np.array(rewards)
    
    # Smooth rewards using moving average
    smoothed = np.convolve(
        rewards_array, 
        np.ones(window) / window, 
        mode='valid'
    )
    
    # Final performance (last 10% of episodes)
    final_portion = max(1, len(rewards) // 10)
    final_rewards = rewards_array[-final_portion:]
    
    stats = {
        'mean': float(np.mean(rewards_array)),
        'std': float(np.std(rewards_array)),
        'min': float(np.min(rewards_array)),
        'max': float(np.max(rewards_array)),
        'final_mean': float(np.mean(final_rewards)),
        'final_std': float(np.std(final_rewards)),
        'smoothed_mean': float(np.mean(smoothed)) if len(smoothed) > 0 else 0,
        'total_episodes': len(rewards),
        'raw_rewards': rewards,
        'smoothed_rewards': smoothed.tolist() if len(smoothed) > 0 else []
    }
    
    return stats


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = TrainingDataLoader()
    all_data = loader.load_all_data()
    
    # Compute statistics
    print("\n" + "="*60)
    print("Statistics Summary")
    print("="*60)
    
    for algo in all_data:
        print(f"\n{algo}:")
        for env in all_data[algo]:
            stats = compute_statistics(all_data[algo][env])
            print(f"  {env}:")
            print(f"    Final Mean: {stats['final_mean']:.2f} ± {stats['final_std']:.2f}")
            print(f"    Overall Mean: {stats['mean']:.2f}")
            print(f"    Episodes: {stats['total_episodes']}")
