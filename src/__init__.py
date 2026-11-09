"""
MiniGPT 中文小模型训练包。

完整流程（在项目根目录执行）：
  python -m src.download_data --max_articles 5000
  python -m src.prepare_dataset --max_tokens 2000000
  python -m src.train --run_name baseline --epochs 1
  python -m src.evaluate --split test
  python -m src.generate --prompt "人工智能的发展"

本文件用于声明 src 为 Python 包，本身不提供可执行入口。各模块的输入、
输出和更多参数示例见对应文件顶部说明及项目 USAGE_GUIDE.md。
"""
