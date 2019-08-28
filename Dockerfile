FROM continuumio/anaconda3:2019.03
MAINTAINER Shiyu Liu "liushiyu1994@gmail.com"

ENV PYTHONPATH=/Lactate_MFA
WORKDIR /$PYTHONPATH/
COPY src /$PYTHONPATH/src
COPY data /$PYTHONPATH/data
RUN conda update conda && \
    conda config --add channels conda-forge && \
    conda install -y --freeze-installed python-ternary && \
    conda clean -afy
ENTRYPOINT ["python", "src/new_model_main.py"]
CMD ["-h"]