# DevOps Copilot 一键启动脚本
# 用法: 在项目根目录下运行 .\start.ps1

Write-Host "===== DevOps Copilot 环境自检 =====" -ForegroundColor Cyan

# 检查虚拟环境是否已激活
if ($env:CONDA_DEFAULT_ENV -ne "devops-copilot") {
    Write-Host "未激活 devops-copilot 虚拟环境,正在激活..." -ForegroundColor Yellow
    conda activate devops-copilot
}
else {
    Write-Host "[OK] 虚拟环境已激活: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
}

# 检查 .env 文件是否存在
if (-not (Test-Path ".env")) {
    Write-Host "[ERROR] 未找到 .env 文件,请先创建并配置 ZHIPU_API_KEY" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "[OK] .env 文件存在" -ForegroundColor Green
}

# 检查向量库是否已建好,没有的话提示用户先建索引
if (-not (Test-Path "data\chroma_db")) {
    Write-Host "[WARN] 未检测到向量数据库,首次运行请先执行: python -m app.tools.vectorstore" -ForegroundColor Yellow
}
else {
    Write-Host "[OK] 向量数据库已存在" -ForegroundColor Green
}

# 启动服务
Write-Host "正在启动 DevOps Copilot 服务..." -ForegroundColor Cyan
uvicorn main:app --reload