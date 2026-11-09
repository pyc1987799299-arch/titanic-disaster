FROM r-base:4.4.2@sha256:fe9b29520eeb5292d814b0958783c0ddfcdab37402967a3e67307604354f98d7
WORKDIR /app
COPY src/r/install_packages.R /app/install_packages.R
RUN Rscript /app/install_packages.R
COPY src/r/train.R /app/train.R
ENTRYPOINT ["Rscript", "/app/train.R"]
CMD ["/app/data", "/app/results"]
