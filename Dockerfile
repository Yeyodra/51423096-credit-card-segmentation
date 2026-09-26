FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/ ./model/
COPY credit_card_clustered.csv .

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
