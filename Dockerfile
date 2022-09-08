FROM python:3.9.12 as release-base

WORKDIR /app

#RUN apt-get update \
#  && apt-get install gfortran libopenblas-dev liblapack-dev -y

RUN pip3 install poetry
ENV PATH="/root/.local/bin:${PATH}"
COPY poetry.lock /app/
COPY pyproject.toml /app/

RUN poetry install

COPY src /app/src

EXPOSE 8888

CMD poetry run jupyter lab