CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | tr -d '.')
echo "CUDA VERSION", $CUDA_VER

pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu$CUDA_VER

if [ ! -f "/root/.cache/pip/diffusers.tar.gz" ]; then
    wget -O /root/.cache/pip/diffusers.tar.gz https://github.com/huggingface/diffusers/archive/a00d536450c6cb83824366f4b4d22426cba9165c.tar.gz
fi
pip install /root/.cache/pip/diffusers.tar.gz

# pip freeze
python -m uvicorn service:app --host 0.0.0.0 --port 8000