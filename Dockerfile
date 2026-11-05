FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir uv==0.13.6
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:${PATH}"
EXPOSE 8000
ENTRYPOINT ["valendra-privacy-filter", "serve"]
CMD ["--checkpoint", "/models/valendra-privacy-filter", "--host", "0.0.0.0"]
