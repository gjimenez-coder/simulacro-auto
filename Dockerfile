FROM anaconda/miniconda:latest

COPY environment.yml /tmp/environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda env create -f /tmp/environment.yml \
    && conda run -n datalab python -m ipykernel install --sys-prefix --name datalab --display-name "datalab" \
    && conda clean -afy

WORKDIR /workspace
CMD ["conda", "run", "--no-capture-output", "-n", "datalab", "bash"]
