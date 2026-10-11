FROM postgres:16

# Install build dependencies for HypoPG
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install HypoPG extension
RUN git clone https://github.com/HypoPG/hypopg.git /tmp/hypopg \
    && cd /tmp/hypopg \
    && make \
    && make install \
    && rm -rf /tmp/hypopg

# Copy initialization SQL and application schema
COPY infra/postgres/init.sql /docker-entrypoint-initdb.d/01-init.sql
COPY sql/pagila_schema.sql /docker-entrypoint-initdb.d/02-pagila_schema.sql

EXPOSE 5432
