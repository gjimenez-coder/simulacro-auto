FROM continuumio/miniconda3:latest

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

WORKDIR /workspace
CMD ["conda", "run", "--no-capture-output", "-n", "datalab", "bash"]
