FROM continuumio/anaconda3:2019.03
MAINTAINER Shiyu Liu "liushiyu1994@gmail.com"

ENV PYTHONPATH=/lactate_exchange PARAM=""
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install -y cvxopt python-ternary
WORKDIR /lactate_exchange/
ENTRYPOINT python src/new_model_main.py $PARAM