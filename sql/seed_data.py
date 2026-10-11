"""
Data seeder: populates the sample schema with realistic synthetic data.
Run this once after creating the schema.
"""
import os
import random
import string
from datetime import datetime, timedelta, timezone

import psycopg2
from faker import Faker
from dotenv import load_dotenv

load_dotenv()

TARGET_DSN = os.getenv("TARGET_DB_DSN", "postgresql://targetuser:targetpass@localhost:5433/targetdb")

fake = Faker()
rng = random.Random(42)   # reproducible

CATEGORIES = ["Electronics", "Books", "Clothing", "Home & Garden", "Sports",
               "Toys", "Beauty", "Automotive", "Music", "Movies"]

WAREHOUSES = [
    ("US East", "US"), ("US West", "US"), ("EU Central", "DE"),
    ("Asia Pacific", "SG"),
]

EVENT_TYPES = ["view", "add_to_cart", "purchase", "search", "wishlist", "review"]


def insert_batch(cur, sql, rows):
    cur.executemany(sql, rows)


def seed_data(dsn: str, num_customers: int = 50_000, num_products: int = 5_000,
              num_orders: int = 200_000, num_events: int = 1_000_000):
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cur = conn.cursor()

    print("Seeding categories...")
    cur.executemany(
        "INSERT INTO category (name) VALUES (%s) ON CONFLICT DO NOTHING",
        [(c,) for c in CATEGORIES]
    )
    conn.commit()

    print("Seeding warehouses...")
    cur.executemany(
        "INSERT INTO warehouse (name, country) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        WAREHOUSES
    )
    conn.commit()

    # Get category IDs
    cur.execute("SELECT category_id FROM category")
    cat_ids = [r[0] for r in cur.fetchall()]

    print(f"Seeding {num_customers:,} customers...")
    BATCH = 1000
    customers = []
    for i in range(num_customers):
        customers.append((
            fake.unique.email(),
            fake.first_name(),
            fake.last_name(),
            rng.choice(["US", "GB", "DE", "FR", "IN", "AU", "CA", "BR"]),
            fake.date_time_between(start_date="-2y", end_date="now", tzinfo=timezone.utc),
            rng.random() > 0.2,
        ))
        if len(customers) >= BATCH:
            cur.executemany(
                "INSERT INTO customer (email, first_name, last_name, country, created_at, is_active) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                customers
            )
            conn.commit()
            customers = []
            print(f"  {i+1}/{num_customers}", end="\r")
    if customers:
        cur.executemany(
            "INSERT INTO customer (email, first_name, last_name, country, created_at, is_active) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            customers
        )
        conn.commit()
    print(f"  {num_customers}/{num_customers} customers done")

    print(f"Seeding {num_products:,} products...")
    products = []
    for i in range(num_products):
        products.append((
            rng.choice(cat_ids),
            fake.bs().title()[:80],
            fake.sentence(nb_words=12),
            round(rng.uniform(1.99, 999.99), 2),
            rng.randint(0, 500),
        ))
        if len(products) >= BATCH:
            cur.executemany(
                "INSERT INTO product (category_id, name, description, price, stock_quantity) "
                "VALUES (%s, %s, %s, %s, %s)",
                products
            )
            conn.commit()
            products = []
    if products:
        cur.executemany(
            "INSERT INTO product (category_id, name, description, price, stock_quantity) "
            "VALUES (%s, %s, %s, %s, %s)",
            products
        )
        conn.commit()
    print(f"  {num_products} products done")

    # Get customer and product IDs (sampled)
    cur.execute("SELECT customer_id FROM customer ORDER BY random() LIMIT 10000")
    cust_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT product_id FROM product ORDER BY random() LIMIT 1000")
    prod_ids = [r[0] for r in cur.fetchall()]

    print(f"Seeding {num_orders:,} orders...")
    STATUSES = ["pending", "confirmed", "shipped", "delivered", "cancelled"]
    STATUS_WEIGHTS = [0.05, 0.10, 0.15, 0.55, 0.15]
    orders = []
    order_items_pending = []
    for i in range(num_orders):
        cust = rng.choice(cust_ids)
        created = fake.date_time_between(start_date="-1y", end_date="now", tzinfo=timezone.utc)
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        n_items = rng.randint(1, 5)
        total = 0.0
        items_spec = []
        for _ in range(n_items):
            qty = rng.randint(1, 3)
            price = round(rng.uniform(1.99, 299.99), 2)
            total += qty * price
            items_spec.append((rng.choice(prod_ids), qty, price))
        orders.append((cust, status, round(total, 2), created, created))
        order_items_pending.append(items_spec)

        if len(orders) >= BATCH:
            cur.execute(
                "INSERT INTO orders (customer_id, status, total_amount, created_at, updated_at) "
                f"VALUES {','.join(['(%s, %s, %s, %s, %s)'] * len(orders))} RETURNING order_id",
                [val for order in orders for val in order]
            )
            returned_ids = [r[0] for r in cur.fetchall()]
            flattened_items = []
            for oid, items in zip(returned_ids, order_items_pending):
                for pid, qty, price in items:
                    flattened_items.append((oid, pid, qty, price))
            cur.executemany(
                "INSERT INTO order_item (order_id, product_id, quantity, unit_price) "
                "VALUES (%s, %s, %s, %s)",
                flattened_items
            )
            conn.commit()
            orders = []
            order_items_pending = []
            print(f"  {i+1}/{num_orders}", end="\r")
    if orders:
        cur.execute(
            "INSERT INTO orders (customer_id, status, total_amount, created_at, updated_at) "
            f"VALUES {','.join(['(%s, %s, %s, %s, %s)'] * len(orders))} RETURNING order_id",
            [val for order in orders for val in order]
        )
        returned_ids = [r[0] for r in cur.fetchall()]
        flattened_items = []
        for oid, items in zip(returned_ids, order_items_pending):
            for pid, qty, price in items:
                flattened_items.append((oid, pid, qty, price))
        cur.executemany(
            "INSERT INTO order_item (order_id, product_id, quantity, unit_price) "
            "VALUES (%s, %s, %s, %s)",
            flattened_items
        )
        conn.commit()
    print(f"  {num_orders} orders done")

    print(f"Seeding {num_events:,} events...")
    events = []
    now = datetime.now(timezone.utc)
    for i in range(num_events):
        occurred = now - timedelta(days=rng.uniform(0, 365))
        events.append((
            rng.choice(cust_ids) if rng.random() > 0.3 else None,
            rng.choice(EVENT_TYPES),
            "product",
            rng.choice(prod_ids),
            occurred,
            None,  # metadata
        ))
        if len(events) >= BATCH * 5:
            cur.executemany(
                "INSERT INTO event_log (customer_id, event_type, entity_type, entity_id, occurred_at, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                events
            )
            conn.commit()
            events = []
            print(f"  {i+1}/{num_events}", end="\r")
    if events:
        cur.executemany(
            "INSERT INTO event_log (customer_id, event_type, entity_type, entity_id, occurred_at, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            events
        )
        conn.commit()
    print(f"  {num_events} events done")

    cur.execute("ANALYZE")
    conn.commit()
    cur.close()
    conn.close()
    print("\n✅ Seeding complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed sample data into target DB")
    parser.add_argument("--customers", type=int, default=50_000)
    parser.add_argument("--products",  type=int, default=5_000)
    parser.add_argument("--orders",    type=int, default=200_000)
    parser.add_argument("--events",    type=int, default=1_000_000)
    parser.add_argument("--dsn",       type=str, default=TARGET_DSN)
    args = parser.parse_args()
    seed_data(args.dsn, args.customers, args.products, args.orders, args.events)
