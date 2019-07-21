FROM continuumio/anaconda3:latest

RUN conda config --add channels conda-forge && \
    conda install -y cvxopt python-ternary
ENV PYTHONPATH=/lactate_exchange PARAM=""
WORKDIR /lactate_exchange/
ENTRYPOINT python src/new_model_main.py $PARAM