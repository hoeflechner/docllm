FROM python:3.12-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
COPY config.toml config.toml
COPY docllm.py docllm.py 

CMD ["streamlit", "run", "--server.enableStaticServing=true", "docllm.py"]