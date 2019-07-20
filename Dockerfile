FROM continuumio/anaconda:latest

RUN conda config --add channels conda-forge \
    conda install -f cvxopt python-ternary \
ENV PYTHONPATH=/lactate_exchange
WORKDIR /lactate_exchange/
CMD ["python", "src/new_model_main.py"]