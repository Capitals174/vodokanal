FROM python:3.9

RUN pip install uvicorn
RUN pip install uvicorn[standard]
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR fastapi_app
COPY . /fastapi_app

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]