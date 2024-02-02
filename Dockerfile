#
FROM ubuntu

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade pip


#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app
COPY ./test /code/test

#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]