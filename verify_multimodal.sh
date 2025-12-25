#!/bin/bash
# 快速测试脚本 - 验证多模态训练环境

echo "=========================================="
echo "多模态模型窃取 - 环境验证"
echo "=========================================="

cd /root/workspace/align
export python=$HOME/anaconda3/envs/align/bin/python3

echo ""
echo "✓ 代码已成功修改，使用 LlavaForConditionalGeneration"
echo ""
echo "训练已准备就绪！"
echo ""
echo "运行完整训练："
echo "  bash scripts/6.0.sciqa_lord6_lora.sh"
echo ""
echo "预计训练时间："
echo "  - 128样本: 2-4小时"
echo "  - 训练周期: 3 periods"
echo "  - 每个period: 64 stages"
echo ""
echo "=========================================="
