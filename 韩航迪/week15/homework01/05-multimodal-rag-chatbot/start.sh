#!/bin/bash
# 多模态RAG聊天机器人启动脚本

echo "=========================================="
echo "多模态RAG聊天机器人 - 启动服务"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.8+"
    exit 1
fi

echo "✓ Python环境检查通过"

# 检查依赖
if [ ! -f "requirements.txt" ]; then
    echo "错误: 找不到requirements.txt文件"
    exit 1
fi

echo "正在安装依赖..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi

echo "✓ 依赖安装完成"

# 创建必要的目录
mkdir -p uploads
mkdir -p processed

echo "✓ 目录创建完成"

# 启动服务
echo ""
echo "=========================================="
echo "启动FastAPI服务..."
echo "访问地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo "按 Ctrl+C 停止服务"
echo "=========================================="
echo ""

python main.py
