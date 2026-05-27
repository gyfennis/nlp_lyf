@echo off
REM 多模态RAG聊天机器人启动脚本 (Windows)

echo ==========================================
echo 多模态RAG聊天机器人 - 启动服务
echo ==========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo ✓ Python环境检查通过

REM 检查依赖
if not exist "requirements.txt" (
    echo 错误: 找不到requirements.txt文件
    pause
    exit /b 1
)

echo 正在安装依赖...
pip install -r requirements.txt

if errorlevel 1 (
    echo 错误: 依赖安装失败
    pause
    exit /b 1
)

echo ✓ 依赖安装完成

REM 创建必要的目录
if not exist "uploads" mkdir uploads
if not exist "processed" mkdir processed

echo ✓ 目录创建完成

echo.
echo ==========================================
echo 启动FastAPI服务...
echo 访问地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo 按 Ctrl+C 停止服务
echo ==========================================
echo.

python main.py

pause
