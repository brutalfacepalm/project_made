###########
# BUILDER #
###########

FROM python:3.9.5-slim-buster as builder

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc

RUN pip install --upgrade pip
RUN pip install flake8==3.9.1
COPY ./flask_app /usr/src/app/
# RUN flake8 --ignore=E501,F401 .

COPY ./flask_app/requirements.txt .
RUN apt-get update -y && apt-get install -y gcc
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt


#########
# FINAL #
#########

FROM python:3.9.5-slim-buster

RUN mkdir -p /home/app

RUN addgroup --system app && adduser --system --group app

ENV HOME=/home/app
ENV APP_HOME=/home/app/flask_app
ENV PRODMODEL=/home/app/prodmodel
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends netcat
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache /wheels/*

# COPY ./entrypoint.prod.sh $APP_HOME

COPY ./flask_app $APP_HOME
COPY ./prodmodel $PRODMODEL

RUN chown -R app:app $HOME
RUN chmod -R 777 /home/app

USER app

# ENTRYPOINT ["/home/app/entrypoint.prod.sh"]