FROM python:3.7

WORKDIR /trial

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8080

COPY . /trial

CMD streamline run —server.port 8080 —server.enableCORS false trial.py