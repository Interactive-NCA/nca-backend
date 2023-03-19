FROM python:3.9

ENV PYTHONPATH=/app:$PYTHONPATH

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

RUN apt-get update && apt-get install libgl1 -y

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]