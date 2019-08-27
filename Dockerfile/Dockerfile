FROM continuumio/anaconda3:2019.03
MAINTAINER Shiyu Liu "liushiyu1994@gmail.com"

ENV PYTHONPATH=/Lactate_MFA PARAM=""
WORKDIR /$PYTHONPATH/
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install -y python-ternary && \
    git clone https://github.com/LocasaleLab/Lactate_MFA $PYTHONPATH
ENTRYPOINT python src/new_model_main.py $PARAM