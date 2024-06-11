FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="team-erc"

ENV WORKERS_PER_CORE=4
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT="200"

RUN mkdir /yolov5-fastapi

COPY requirements.in /yolov5-fastapi

WORKDIR /yolov5-fastapi

# Install pip-tools to provide pip-compile
RUN pip install pip-tools

# Generate the requirements.txt file from requirements.in
RUN pip-compile requirements.in && pip install -r requirements.txt

# Install libgl1-mesa-glx to provide libGL.so.1
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY . /yolov5-fastapi

EXPOSE 60333

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "60333"]
