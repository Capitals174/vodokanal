FROM python:3.9

WORKDIR service
COPY src/service /service

RUN pip install uvicorn
RUN pip install uvicorn[standard]
RUN pip3 install -r requirements.txt

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]