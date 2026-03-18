ARG PYTHON_VERSION=3.11

# Base layer for building dependencies and release images
FROM debian:bookworm-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Base build image
FROM base AS build
ARG PYTHON_VERSION

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    make \
    python3-dev \
    python3-venv \
    python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Setup python dependencies
RUN python3 -m venv --without-pip /venv

COPY requirements.txt .

RUN python3 -m pip install \
  --no-cache-dir \
  --target="/venv/lib/python${PYTHON_VERSION}/site-packages" \
  -r requirements.txt

# Setup external dependencies
ARG CLEVERTAGGER_SHA
ARG WAPITI_SHA

RUN curl \
  -L https://github.com/rsennrich/clevertagger/archive/${CLEVERTAGGER_SHA}.tar.gz \
  -o ct.tar.gz \
 && curl \
  -L https://github.com/rsennrich/Wapiti/archive/${WAPITI_SHA}.tar.gz \
  -o wp.tar.gz \
 && mkdir clevertagger wapiti \
 && tar -xzf ct.tar.gz --strip-components=1 -C clevertagger \
 && tar -xzf wp.tar.gz --strip-components=1 -C wapiti \
 && make -C wapiti -j $(nproc) \
 && rm -f ct.tar.gz wp.tar.gz

ARG SMOR_MODEL
ARG CRF_MODEL
ARG CRF_BACKEND_EXEC

# Configure clevertagger
RUN sed -i \
  -e "s,^SMOR_MODEL =.*$,SMOR_MODEL = '${SMOR_MODEL}'," \
  -e "s,^CRF_MODEL =.*$,CRF_MODEL = '${CRF_MODEL}'," \
  -e "s,^CRF_BACKEND_EXEC =.*$,CRF_BACKEND_EXEC = '${CRF_BACKEND_EXEC}'," \
  clevertagger/config.py

# Runtime
FROM base AS runtime
ARG PYTHON_VERSION

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libutf8-all-perl \
    sfst \
    swi-prolog \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /venv /venv
COPY --from=build /build/clevertagger /app/external/clevertagger
COPY --from=build /build/wapiti/wapiti /usr/local/bin/wapiti

COPY models/ /app/external/
COPY . /app/

ENV PATH="/venv/bin:$PATH"
ENV PYTHON_VERSION=${PYTHON_VERSION}

CMD /venv/bin/python${PYTHON_VERSION} parzu_server.py --host 0.0.0.0
