-- ════════════════════════════════════════════════════════════════

-- Enable extensions on target DB
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS hypopg;

-- ─── Customers ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customer (
    customer_id     serial      PRIMARY KEY,
    email           text        NOT NULL UNIQUE,
    first_name      text        NOT NULL,
    last_name       text        NOT NULL,
    country         text        NOT NULL DEFAULT 'US',
    created_at      timestamptz NOT NULL DEFAULT now(),
    last_login      timestamptz,
    is_active       boolean     NOT NULL DEFAULT true
);

-- ─── Products ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS category (
    category_id serial  PRIMARY KEY,
    name        text    NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS product (
    product_id      serial          PRIMARY KEY,
    category_id     int             NOT NULL REFERENCES category(category_id),
    name            text            NOT NULL,
    description     text,
    price           numeric(10, 2)  NOT NULL,
    stock_quantity  int             NOT NULL DEFAULT 0,
    created_at      timestamptz     NOT NULL DEFAULT now(),
    is_active       boolean         NOT NULL DEFAULT true
);

-- ─── Orders ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    order_id        serial          PRIMARY KEY,
    customer_id     int             NOT NULL REFERENCES customer(customer_id),
    status          text            NOT NULL DEFAULT 'pending',  -- pending|confirmed|shipped|delivered|cancelled
    total_amount    numeric(12, 2)  NOT NULL,
    created_at      timestamptz     NOT NULL DEFAULT now(),
    updated_at      timestamptz     NOT NULL DEFAULT now(),
    shipped_at      timestamptz,
    country         text
);

CREATE TABLE IF NOT EXISTS order_item (
    order_item_id   serial          PRIMARY KEY,
    order_id        int             NOT NULL REFERENCES orders(order_id),
    product_id      int             NOT NULL REFERENCES product(product_id),
    quantity        int             NOT NULL,
    unit_price      numeric(10, 2)  NOT NULL
);

-- ─── Inventory & Warehouse ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS warehouse (
    warehouse_id    serial  PRIMARY KEY,
    name            text    NOT NULL,
    country         text    NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    inventory_id    serial  PRIMARY KEY,
    product_id      int     NOT NULL REFERENCES product(product_id),
    warehouse_id    int     NOT NULL REFERENCES warehouse(warehouse_id),
    quantity        int     NOT NULL DEFAULT 0,
    last_updated    timestamptz NOT NULL DEFAULT now()
);

-- ─── Reviews ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS review (
    review_id       serial  PRIMARY KEY,
    product_id      int     NOT NULL REFERENCES product(product_id),
    customer_id     int     NOT NULL REFERENCES customer(customer_id),
    rating          smallint NOT NULL CHECK (rating BETWEEN 1 AND 5),
    body            text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- ─── Sessions (high-volume, good for index testing) ────────────────────
CREATE TABLE IF NOT EXISTS user_session (
    session_id      text        PRIMARY KEY,
    customer_id     int         REFERENCES customer(customer_id),
    started_at      timestamptz NOT NULL DEFAULT now(),
    ended_at        timestamptz,
    ip_address      inet,
    user_agent      text
);

-- ─── Event Log (append-only, large table) ─────────────────────────────
CREATE TABLE IF NOT EXISTS event_log (
    event_id      bigserial,
    customer_id   int REFERENCES customer(customer_id),
    event_type    text NOT NULL,
    entity_type   text,
    entity_id     int,
    occurred_at   timestamptz NOT NULL DEFAULT now(),
    metadata      jsonb,
    PRIMARY KEY (event_id, occurred_at)
) PARTITION BY RANGE (occurred_at);

-- Create partitions (monthly)
CREATE TABLE IF NOT EXISTS event_log_default
    PARTITION OF event_log DEFAULT;

-- ─── Intentionally missing useful indexes (for advisor to find) ────────
-- orders.customer_id, orders.status, orders.created_at — no index
-- order_item.order_id — no index (FK but no btree)
-- event_log.customer_id, event_log.event_type — no index
-- review.product_id — no index
-- user_session.customer_id, user_session.started_at — no index
