#!/usr/bin/env python3
"""
从保存的训练数据文件生成图表

使用方法:
    # 从指定文件绘图
    python plot_from_data.py --file checkpoints/DEM_normal_training_data_20231015_123456.json
    
    # 使用最新的数据文件
    python plot_from_data.py --latest
    
    # 列出所有可用的数据文件
    python plot_from_data.py --list
    
    # 显示数据摘要
    python plot_from_data.py --file checkpoints/DEM_normal_training_data_20231015_123456.json --summary
    
    # 显示图表（不仅仅保存）
    python plot_from_data.py --file checkpoints/DEM_normal_training_data_20231015_123456.json --show
"""

import argparse
import os
import sys
from src.utils import (
    plot_from_file, 
    list_training_data_files,
    print_training_data_summary
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='从保存的训练数据生成图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--file', '-f', type=str,
                       help='训练数据文件路径')
    parser.add_argument('--latest', action='store_true',
                       help='使用最新的训练数据文件')
    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有可用的训练数据文件')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='显示训练数据摘要')
    parser.add_argument('--show', action='store_true',
                       help='显示图表（而不仅仅是保存）')
    parser.add_argument('--plot-dir', type=str, default='plots',
                       help='图表保存目录 (默认: plots)')
    parser.add_argument('--data-dir', type=str, default='checkpoints',
                       help='训练数据目录 (默认: checkpoints)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 列出所有数据文件
    if args.list:
        print("\n📁 可用的训练数据文件:\n")
        files = list_training_data_files(args.data_dir)
        
        if not files:
            print(f"   ⚠️  在 {args.data_dir} 中未找到训练数据文件")
            print(f"   提示: 训练数据文件应以 '_training_data_' 结尾并以 .json 格式保存")
            return
        
        for i, filepath in enumerate(files, 1):
            filename = os.path.basename(filepath)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   {i:2d}. {filename}")
            print(f"       大小: {size:.1f} KB")
            print(f"       路径: {filepath}\n")
        
        print(f"总计: {len(files)} 个文件\n")
        return
    
    # 确定要使用的文件
    filepath = None
    
    if args.latest:
        files = list_training_data_files(args.data_dir)
        if files:
            filepath = files[0]  # 已按时间排序，第一个是最新的
            print(f"📄 使用最新的训练数据文件:")
            print(f"   {os.path.basename(filepath)}\n")
        else:
            print(f"❌ 在 {args.data_dir} 中未找到训练数据文件")
            return
    elif args.file:
        filepath = args.file
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            return
    else:
        print("❌ 请指定数据文件 (--file) 或使用最新文件 (--latest)")
        print("   使用 --help 查看帮助信息")
        print("   使用 --list 查看所有可用文件")
        return
    
    # 显示数据摘要
    if args.summary:
        print_training_data_summary(filepath)
        
        # 如果只是要摘要，就不继续绘图了
        if not args.file and not args.latest:
            return
    
    # 生成图表
    try:
        plot_files = plot_from_file(
            filepath=filepath,
            save_dir=args.plot_dir,
            show_plots=args.show
        )
        
        print(f"\n✅ 成功生成图表:")
        for i, pf in enumerate(plot_files, 1):
            print(f"   {i:2d}. {os.path.basename(pf)}")
        
        print(f"\n📁 所有图表已保存到: {args.plot_dir}\n")
        
    except Exception as e:
        print(f"\n❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
