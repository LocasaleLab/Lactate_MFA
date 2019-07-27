FROM continuumio/anaconda3:latest
MAINTAINER Shiyu Liu "liushiyu1994@gmail.com"

ENV PATH /opt/conda/bin:$PATH
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install -y cvxopt python-ternary
ENV PYTHONPATH=/lactate_exchange PARAM=""
WORKDIR /lactate_exchange/
ENTRYPOINT python src/new_model_main.py $PARAM