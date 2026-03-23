# Base layer for building dependencies and release images
FROM debian:bookworm-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Base build image
FROM base AS build

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Setup python dependencies
RUN python3 -m venv --without-pip /venv

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir \
  --target="/venv/lib/python3.11/site-packages" \
  -r requirements.txt

# Setup external dependencies
ARG CLEVERTAGGER_SHA
ARG WAPITI_SHA

RUN mkdir -p clevertagger wapiti \
 && curl -L "https://github.com/rsennrich/clevertagger/archive/${CLEVERTAGGER_SHA}.tar.gz" \
    | tar -xzC clevertagger --strip-components=1 \
 && curl -L "https://github.com/rsennrich/Wapiti/archive/${WAPITI_SHA}.tar.gz" \
    | tar -xzC wapiti --strip-components=1 \
 && make -C wapiti -j $(nproc) \
 && mv wapiti/wapiti /usr/bin/wapiti \
 && rm -rf wapiti

COPY docker/configs/clevertagger.py clevertagger/config.py

# Runtime
FROM base AS runtime

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libutf8-all-perl \
    sfst \
    # SWI-Prolog installation without GUI components
    swi-prolog-nox \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /venv /venv
COPY --from=build /build/clevertagger external/clevertagger
COPY --from=build /usr/bin/wapiti /usr/bin/wapiti

COPY docker/configs/parzu.ini parzu.ini
COPY parzu/. .

ENV PATH="/venv/bin:$PATH"

CMD python3 parzu_server.py --host 0.0.0.0
