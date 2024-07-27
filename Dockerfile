FROM python:3.6-slim

RUN mkdir /application

WORKDIR /application

COPY requirement.txt

RUN command pip install -r requirement.txt


COPY ..

ENV venv

EXPOSE 5000


STOPSIGNAL SIGNINT

ENTRYPOINT ['python']

CMD ["app.py"]


