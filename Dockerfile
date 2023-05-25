FROM python:3.9

WORKDIR service
COPY src/service /service

RUN pip install uvicorn
RUN pip install uvicorn[standard]
RUN pip3 install -r /service/requirements.txt

EXPOSE 8080

CMD ["uvicorn", "service:app", "--reload", "--port", "8080"]