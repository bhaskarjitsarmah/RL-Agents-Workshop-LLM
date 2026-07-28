"""The text-to-SQL environment: a deterministic toy "shop" database + scorer.

This is the *environment* the agent acts in, and the *eval interface (V)* from the
agent-harness framework H = (E, T, C, S, L, V). Everything is local, free, and
reproducible -- no GPU, no network. The DB is rebuilt identically every time
from a fixed seed, so gold SQL results are stable across machines.
"""

from __future__ import annotations

import os
import random
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
DB_PATH = os.path.join(DATA_DIR, "shop.db")

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT NOT NULL,
    segment     TEXT NOT NULL,          -- Consumer | SMB | Enterprise
    signup_date TEXT NOT NULL           -- YYYY-MM-DD
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,           -- Electronics | Books | Clothing | Home | Toys
    price      REAL NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date  TEXT NOT NULL,          -- YYYY-MM-DD
    status      TEXT NOT NULL,          -- completed | pending | cancelled | returned
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE TABLE order_items (
    item_id    INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity   INTEGER NOT NULL,
    FOREIGN KEY (order_id)   REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

# A compact, human-readable schema string to hand the model in prompts.
SCHEMA_TEXT = """\
Tables:
  customers(customer_id, name, city, segment, signup_date)
      segment in (Consumer, SMB, Enterprise)
  products(product_id, name, category, price)
      category in (Electronics, Books, Clothing, Home, Toys)
  orders(order_id, customer_id, order_date, status)
      status in (completed, pending, cancelled, returned)
  order_items(item_id, order_id, product_id, quantity)

Relationships:
  orders.customer_id    -> customers.customer_id
  order_items.order_id  -> orders.order_id
  order_items.product_id-> products.product_id

Notes:
  - Dates are TEXT in 'YYYY-MM-DD' format; compare lexicographically or with strftime.
  - "Revenue" of an order item = order_items.quantity * products.price.
  - A "sale" / completed revenue counts only orders with status = 'completed'.
"""

_CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Kolkata"]
_SEGMENTS = ["Consumer", "SMB", "Enterprise"]
_STATUSES = ["completed", "pending", "cancelled", "returned"]
_CATEGORIES = ["Electronics", "Books", "Clothing", "Home", "Toys"]
_FIRST = ["Aarav", "Diya", "Vivaan", "Ananya", "Aditya", "Isha", "Kabir", "Meera",
          "Rohan", "Saanvi", "Arjun", "Riya", "Devansh", "Tara", "Yash", "Nisha",
          "Karan", "Pooja", "Manav", "Sara"]
_PRODUCTS = [
    ("Wireless Earbuds", "Electronics", 2999.0),
    ("Laptop Stand", "Electronics", 1499.0),
    ("USB-C Hub", "Electronics", 1999.0),
    ("Mechanical Keyboard", "Electronics", 4999.0),
    ("Data Science Handbook", "Books", 799.0),
    ("Deep Learning Primer", "Books", 1199.0),
    ("Cotton T-Shirt", "Clothing", 599.0),
    ("Running Shoes", "Clothing", 3499.0),
    ("Denim Jacket", "Clothing", 2799.0),
    ("Ceramic Mug", "Home", 349.0),
    ("Desk Lamp", "Home", 1299.0),
    ("Throw Blanket", "Home", 1599.0),
    ("Building Blocks", "Toys", 899.0),
    ("Puzzle 1000pc", "Toys", 699.0),
    ("RC Car", "Toys", 2499.0),
]


def build_db(path: str = DB_PATH, seed: int = 42) -> str:
    """(Re)create the toy shop DB deterministically. Returns the db path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)

    rng = random.Random(seed)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)

    # customers
    customers = []
    for cid, first in enumerate(_FIRST, start=1):
        city = rng.choice(_CITIES)
        seg = rng.choice(_SEGMENTS)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        signup = f"2024-{month:02d}-{day:02d}"
        customers.append((cid, first, city, seg, signup))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    # products
    products = [(pid, n, c, p) for pid, (n, c, p) in enumerate(_PRODUCTS, start=1)]
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    # orders + items
    orders = []
    items = []
    item_id = 1
    # Reserve the last 2 customers and last 3 products as "never ordered" so the
    # set-difference questions (16, 29, 39) have stable, non-empty gold answers.
    max_cust = len(_FIRST) - 2
    max_prod = len(_PRODUCTS) - 3
    for oid in range(1, 81):
        cid = rng.randint(1, max_cust)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        odate = f"2025-{month:02d}-{day:02d}"
        status = rng.choices(_STATUSES, weights=[60, 20, 12, 8])[0]
        orders.append((oid, cid, odate, status))
        for _ in range(rng.randint(1, 4)):
            pid = rng.randint(1, max_prod)
            qty = rng.randint(1, 5)
            items.append((item_id, oid, pid, qty))
            item_id += 1
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)

    con.commit()
    con.close()
    return path


def run_sql(query: str, path: str = DB_PATH):
    """Execute a read-only query. Returns (rows, error). rows is None on error."""
    try:
        con = sqlite3.connect(path)
        con.execute("PRAGMA query_only = ON;")
        cur = con.execute(query)
        rows = cur.fetchall()
        con.close()
        return rows, None
    except Exception as e:  # noqa: BLE001 - we want to surface any SQL error verbatim
        return None, f"{type(e).__name__}: {e}"


def _normalize(rows, ordered: bool):
    norm = []
    for r in rows:
        cells = []
        for c in r:
            if isinstance(c, float):
                cells.append(round(c, 2))
            else:
                cells.append(c)
        norm.append(tuple(cells))
    return norm if ordered else sorted(norm, key=lambda t: [str(x) for x in t])


def score_sql(pred_sql: str, gold_sql: str, path: str = DB_PATH) -> bool:
    """Execution-match: do predicted and gold queries return the same result set?

    Order matters only if the gold query uses ORDER BY. This is the reward
    signal (V) -- objective, automatic, and the foundation of every experiment.
    """
    gold_rows, gold_err = run_sql(gold_sql, path)
    if gold_err is not None:
        raise ValueError(f"Gold SQL failed (fix the dataset): {gold_err}\n{gold_sql}")
    pred_rows, pred_err = run_sql(pred_sql, path)
    if pred_err is not None:
        return False
    ordered = "order by" in gold_sql.lower()
    return _normalize(pred_rows, ordered) == _normalize(gold_rows, ordered)


def load_tasks():
    """Return the NL -> gold SQL eval set (list of {id, question, gold, level})."""
    from .tasks import TASKS  # local import to avoid a cycle
    return list(TASKS)
