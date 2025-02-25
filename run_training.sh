#!/bin/bash

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 默认参数
EPOCHS=8
HIDDEN_DIM=512
NUM_LAYERS=3
LEARNING_RATE=0.001
OUTPUT_DIR="output"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --hidden_dim)
      HIDDEN_DIM="$2"
      shift
      shift
      ;;
    --num_layers)
      NUM_LAYERS="$2"
      shift
      shift
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

echo "==========================================="
echo "开始训练语音分离模型"
echo "训练参数:"
echo "  epochs: $EPOCHS"
echo "  hidden_dim: $HIDDEN_DIM"
echo "  num_layers: $NUM_LAYERS"
echo "  learning_rate: $LEARNING_RATE"
echo "  output_dir: $OUTPUT_DIR"
echo "==========================================="

# 修改train1.py中的参数并运行
sed -i "s/epochs = [0-9]*/epochs = $EPOCHS/g" train1.py
sed -i "s/hidden_dim = [0-9]*/hidden_dim = $HIDDEN_DIM/g" train1.py
sed -i "s/num_layers = [0-9]*/num_layers = $NUM_LAYERS/g" train1.py
sed -i "s/lr=0\.[0-9]*/lr=$LEARNING_RATE/g" train1.py
sed -i "s/output_dir = '[^']*'/output_dir = '$OUTPUT_DIR'/g" train1.py

# 运行训练脚本
python train1.py

echo "训练完成！结果保存在 $OUTPUT_DIR 目录" 