@echo off
cd /d "%~dp0"
if not exist ".tools\uv\uv.exe" call setup.cmd
if errorlevel 1 exit /b 1
if not exist ".venv-fullgpu\Scripts\python.exe" ".tools\uv\uv.exe" venv .venv-fullgpu --python ".tools\python\cpython-3.12-windows-x86_64-none\python.exe"
if errorlevel 1 exit /b 1
".tools\uv\uv.exe" pip install --python ".venv-fullgpu\Scripts\python.exe" "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 exit /b 1
".tools\uv\uv.exe" pip install --python ".venv-fullgpu\Scripts\python.exe" -r requirements-gpu.txt
if errorlevel 1 exit /b 1
".venv-fullgpu\Scripts\python.exe" -c "import torch, mujoco, mujoco_warp, warp; assert torch.version.cuda == '13.0', 'CUDA 13.0 PyTorch wheel required'; assert torch.cuda.is_available(), 'CUDA GPU unavailable: check NVIDIA driver'; print('CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0)); print('Environment ready; benchmark before long training.')"
if errorlevel 1 exit /b 1
echo Full GPU environment ready. Select Full GPU explicitly in humanoid_browser.cmd.
pause
