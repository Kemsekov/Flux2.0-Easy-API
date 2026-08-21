pip install -r requirements.txt
pip freeze | grep torch
# python -m uvicorn service:app --host 0.0.0.0 --port 8000