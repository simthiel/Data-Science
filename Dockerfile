FROM python:3.11-slim

WORKDIR /app

##packages installieren
RUN pip install --no-cache-dir --upgrade pip
COPY Packages.txt .
RUN pip install --no-cache-dir -r Packages.txt
COPY *.csv .
COPY MachineLearning_Showcase.py .
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "MachineLearning_Showcase.py"]
