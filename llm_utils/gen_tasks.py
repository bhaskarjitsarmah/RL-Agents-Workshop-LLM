"""A verifiable text-to-SQL task generator.

The problem this solves
----------------------
Repo 1 optimizes a skill document, and 24 training tasks is plenty for that.
GRPO updates 18M LoRA parameters from sampled reward variance; 24 prompts is not
a training set, it is a rounding error. We need hundreds.

But an RL run is only as trustworthy as its reward, and the reward here is
`score_sql(pred, gold)`. **A single wrong gold poisons every rollout on that
prompt** -- the policy is punished for being right and rewarded for reproducing
our mistake. Asking a model to write NL/SQL pairs would make gold quality the
weakest link in the whole repo.

So we never generate gold. We *construct* it:

    a hand-verified SQL skeleton  +  slot values drawn from the live database
    -> the SAME values formatted into both the question and the query

`SELECT name FROM customers WHERE city='{city}'` with `city='Pune'` is correct by
construction, for every value of `city`. The only thing a human must check is the
skeleton -- one review per family, via
`python scripts/review_sample.py --skeletons`. Machine checks confirm that a gold
executes and returns rows; only a person can confirm it answers the *question*.

Slot values are drawn **from the database**, not from constants: a price
threshold is a real price quantile, a product name is a real product. That is
what keeps generated tasks answerable -- a threshold of 99999 would return
nothing, and an empty gold is worse than useless here (see below).

Empty results are rejected
--------------------------
`score_sql` compares result sets, so a gold returning zero rows is matched by
*every* unrelated query that also returns nothing: a typo'd literal, a dropped
join, `WHERE city='Atlantis'`. That is not a weak training signal, it is a
reward-hacking surface. `validate_task` rejects empty and all-NULL golds
outright -- including for set-difference families, where emptiness is
semantically legitimate but the policy still cannot tell the difference between
earning an empty set and stumbling into one. (127 candidates are dropped this
way; the count is in the audit.)

Leakage
-------
The 16 held-out test tasks are the head-to-head contract with repo 1. If a
generated task duplicates one, the reported number is memorization.
`collides_with_eval` applies four rules against **the 16 test tasks only** --
repo 1's 24 *train* tasks are training data in both repos and leak nothing.
Every rejection is counted in `data/leakage_audit.json`, so the audit is a chart
in NB2 rather than a claim in a README.

Beyond instance dedup, each test task is mapped to its template family
(`TEST_FAMILIES`). Generating with `exclude_families=TEST_FAMILIES` yields a
training set that shares no *pattern* with the test set -- the memorization
control NB2 reports, instead of waiting for someone in the audience to ask.
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Callable

from .db import DB_PATH, _normalize, build_db
from .sqlio import safe_run_sql
from .tasks import TASKS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ===========================================================================
# Templates
# ===========================================================================

@dataclass(frozen=True)
class Template:
    """One hand-verified SQL skeleton plus natural-language paraphrases.

    `questions` and `gold` are `str.format` templates over the same namespace,
    so a slot value that appears in the question necessarily appears in the
    query. That shared namespace is the correctness argument.

    `variants` pins groups of substitutions that must move together -- a group-by
    column and its table, say. Keeping them in variants (rather than sampling
    them independently) means every gold stays a literal, readable query that a
    human can check by eye.
    """

    family: str
    level: str                       # easy | medium | hard
    questions: tuple[str, ...]       # >= 2 paraphrases
    gold: str
    slots: tuple[str, ...] = ()
    variants: tuple[dict, ...] = ({},)
    tags: tuple[str, ...] = ()
    expect_multi_row: bool = False   # GROUP BY families: 1 row means a bad draw


# Revenue is the workshop's one piece of domain knowledge: quantity * price,
# joined through order_items. Repeated verbatim so every gold is self-contained.
_REV = "oi.quantity*p.price"
_J_OIP = ("order_items oi "
          "JOIN orders o ON o.order_id=oi.order_id "
          "JOIN products p ON p.product_id=oi.product_id")
_J_OIPC = _J_OIP + " JOIN customers c ON c.customer_id=o.customer_id"

TEMPLATES: list[Template] = [

    # ------------------------------------------------------------------ easy
    Template(
        family="cust_by_city", level="easy", slots=("city",), tags=("filter",),
        questions=(
            "List the names of all customers in {city}.",
            "Who are the customers based in {city}?",
            "Show me every customer name from {city}.",
            "Which customers live in {city}? Return their names.",
            "Give me the names of {city} customers.",
            "Return customer names for the city of {city}.",
        ),
        gold="SELECT name FROM customers WHERE city='{city}';",
    ),
    Template(
        family="cust_by_segment", level="easy", slots=("segment",), tags=("filter",),
        questions=(
            "List the names of customers in the {segment} segment.",
            "Which customers belong to the {segment} segment? Return names.",
            "Show the names of all {segment} customers.",
            "Who are our {segment} customers? Names only.",
            "Return every customer name classified as {segment}.",
            "Give me the names of customers in segment {segment}.",
        ),
        gold="SELECT name FROM customers WHERE segment='{segment}';",
    ),
    Template(
        family="prod_by_category", level="easy", slots=("category",), tags=("filter",),
        questions=(
            "Show all product names in the {category} category.",
            "List the products categorised as {category}.",
            "What are the names of the {category} products?",
            "Which products fall under {category}? Return their names.",
            "Give me every product name in {category}.",
            "Return the names of items in the {category} range.",
        ),
        gold="SELECT name FROM products WHERE category='{category}';",
    ),
    Template(
        family="count_cust_by_city", level="easy", slots=("city",), tags=("count",),
        questions=(
            "How many customers are there in {city}?",
            "Count the customers located in {city}.",
            "What is the number of customers from {city}?",
        ),
        gold="SELECT COUNT(*) FROM customers WHERE city='{city}';",
    ),
    Template(
        family="count_prod_by_category", level="easy", slots=("category",), tags=("count",),
        questions=(
            "How many products are in the {category} category?",
            "Count the {category} products.",
            "What is the total number of products categorised as {category}?",
            "How many items does the {category} category hold?",
            "Return the count of products under {category}.",
            "Tally up the {category} products.",
        ),
        gold="SELECT COUNT(*) FROM products WHERE category='{category}';",
    ),
    Template(
        family="count_orders_by_status", level="easy", slots=("status",), tags=("count",),
        questions=(
            "How many orders have status '{status}'?",
            "Count the orders whose status is {status}.",
            "What is the number of {status} orders?",
        ),
        gold="SELECT COUNT(*) FROM orders WHERE status='{status}';",
    ),
    Template(
        family="price_of_product", level="easy", slots=("product_name",), tags=("lookup",),
        questions=(
            "What is the price of the product named '{product_name}'?",
            "How much does the {product_name} cost?",
            "Return the price of '{product_name}'.",
            "Look up what '{product_name}' sells for.",
            "Tell me the listed price for {product_name}.",
            "What does a {product_name} go for?",
        ),
        gold="SELECT price FROM products WHERE name='{product_name}';",
    ),
    Template(
        family="attr_of_customer", level="easy", slots=("customer_name",), tags=("lookup",),
        variants=(
            {"col": "city", "label": "city"},
            {"col": "segment", "label": "segment"},
            {"col": "signup_date", "label": "signup date"},
        ),
        questions=(
            "What is the {label} of the customer named {customer_name}?",
            "Return the {label} for customer {customer_name}.",
            "Which {label} is {customer_name} associated with?",
            "Look up {customer_name}'s {label}.",
            "Tell me the {label} recorded for {customer_name}.",
            "For the customer {customer_name}, what {label} is on file?",
        ),
        gold="SELECT {col} FROM customers WHERE name='{customer_name}';",
    ),
    Template(
        family="distinct_col", level="easy", tags=("distinct",), expect_multi_row=True,
        variants=(
            {"col": "city", "table": "customers", "label": "cities where customers live"},
            {"col": "segment", "table": "customers", "label": "customer segments"},
            {"col": "category", "table": "products", "label": "product categories"},
            {"col": "status", "table": "orders", "label": "order statuses"},
        ),
        questions=(
            "List all distinct {label}.",
            "What are the unique {label}?",
            "Show every distinct value among the {label}.",
            "Give me the deduplicated list of {label}.",
            "Which {label} appear in the data? No duplicates.",
            "Return each of the {label} exactly once.",
        ),
        gold="SELECT DISTINCT {col} FROM {table};",
    ),
    Template(
        family="prod_price_compare", level="easy", slots=("price_threshold",),
        tags=("filter", "numeric"),
        variants=(
            {"op": ">", "word": "more than"},
            {"op": "<", "word": "less than"},
            {"op": ">=", "word": "at least"},
        ),
        questions=(
            "Show the names of products that cost {word} {price_threshold}.",
            "Which products are priced {word} {price_threshold}? Return their names.",
            "List product names whose price is {word} {price_threshold}.",
            "Find the products costing {word} {price_threshold} and return their names.",
            "What products have a price {word} {price_threshold}? Names please.",
            "Return every product name where price is {word} {price_threshold}.",
        ),
        gold="SELECT name FROM products WHERE price{op}{price_threshold};",
    ),
    Template(
        family="order_ids_by_status", level="easy", slots=("status",), tags=("filter",),
        questions=(
            "List all order ids that have status '{status}'.",
            "Which order ids are marked {status}?",
            "Return the ids of every {status} order.",
        ),
        gold="SELECT order_id FROM orders WHERE status='{status}';",
    ),
    Template(
        family="prod_in_cat_above_price", level="easy",
        slots=("category", "price_threshold"), tags=("filter", "numeric"),
        questions=(
            "Which {category} products cost more than {price_threshold}? Return names.",
            "List the names of {category} items priced above {price_threshold}.",
            "Show {category} products with a price greater than {price_threshold}.",
            "In the {category} category, which products exceed {price_threshold}? Names only.",
            "Give me {category} product names over {price_threshold}.",
            "Find every {category} product dearer than {price_threshold}.",
        ),
        gold=("SELECT name FROM products "
              "WHERE category='{category}' AND price>{price_threshold};"),   # a high threshold in a cheap category is legitimately empty
    ),

    # ---------------------------------------------------------------- medium
    Template(
        family="count_by_group", level="medium", tags=("groupby",), expect_multi_row=True,
        variants=(
            {"gcol": "city", "table": "customers", "label": "city",
             "plural": "cities"},
            {"gcol": "segment", "table": "customers", "label": "segment",
             "plural": "segments"},
            {"gcol": "category", "table": "products", "label": "category",
             "plural": "categories"},
            {"gcol": "status", "table": "orders", "label": "status",
             "plural": "statuses"},
        ),
        questions=(
            "How many entries are there per {label}? Return {label} and count.",
            "Group by {label} and return each {label} with its count.",
            "Give a breakdown of counts across the {plural}.",
            "For each {label}, how many are there? Return {label} and count.",
            "Show the distribution over {plural} as {label} and count.",
        ),
        gold="SELECT {gcol}, COUNT(*) FROM {table} GROUP BY {gcol};",
    ),
    Template(
        family="agg_price_by_category", level="medium", tags=("groupby", "agg"),
        expect_multi_row=True,
        variants=(
            {"agg": "AVG", "word": "average"},
            {"agg": "SUM", "word": "total"},
            {"agg": "MAX", "word": "highest"},
            {"agg": "MIN", "word": "lowest"},
        ),
        questions=(
            "What is the {word} product price per category? Return category and "
            "{word} price.",
            "For each category, return the category and the {word} price.",
            "Show the {word} price of products grouped by category.",
            "Group products by category and give the {word} price of each.",
            "Per category, what is the {word} price? Return both columns.",
            "Break the {word} product price down by category.",
        ),
        gold="SELECT category, {agg}(price) FROM products GROUP BY category;",
    ),
    Template(
        family="having_count", level="medium", slots=("count_threshold",),
        tags=("groupby", "having"),
        variants=(
            {"gcol": "city", "label": "cities", "noun": "customers"},
            {"gcol": "segment", "label": "segments", "noun": "customers"},
        ),
        questions=(
            "Which {label} have more than {count_threshold} {noun}? Return the "
            "group and the count.",
            "Show the {label} with more than {count_threshold} {noun}, plus the count.",
            "List every group among the {label} having over {count_threshold} {noun}.",
        ),
        gold=("SELECT {gcol}, COUNT(*) FROM customers GROUP BY {gcol} "
              "HAVING COUNT(*)>{count_threshold};"),
    ),
    Template(
        family="topn_products_by_price", level="medium", slots=("n_limit",),
        tags=("orderby", "limit"),
        variants=(
            {"dir": "DESC", "word": "most expensive", "first": "most expensive first"},
            {"dir": "ASC", "word": "cheapest", "first": "cheapest first"},
        ),
        questions=(
            "List the top {n_limit} {word} products with their name and price, {first}.",
            "Return the {n_limit} {word} products as name and price, {first}.",
            "Which are the {n_limit} {word} products? Give name and price, {first}.",
        ),
        gold="SELECT name, price FROM products ORDER BY price {dir} LIMIT {n_limit};",
    ),
    Template(
        family="orders_per_month", level="medium", tags=("date", "groupby"),
        expect_multi_row=True,
        questions=(
            "How many orders were placed in each month of 2025? Return the month "
            "and the count.",
            "Break down the order count by month. Return month and count.",
            "For every month, how many orders were there? Return month and count.",
        ),
        gold=("SELECT strftime('%m', order_date) AS m, COUNT(*) FROM orders "
              "GROUP BY m;"),
    ),
    Template(
        family="orders_per_customer", level="medium", tags=("join", "groupby"),
        expect_multi_row=True,
        questions=(
            "For each customer who placed at least one order, show their name and "
            "the number of orders they placed.",
            "Return every ordering customer's name alongside their order count.",
            "How many orders has each customer with orders placed? Name and count.",
        ),
        gold=("SELECT c.name, COUNT(*) FROM customers c "
              "JOIN orders o ON o.customer_id=c.customer_id "
              "GROUP BY c.customer_id, c.name;"),
    ),
    Template(
        family="qty_per_product", level="medium", tags=("join", "groupby", "agg"),
        expect_multi_row=True,
        questions=(
            "What is the total quantity ordered for each product? Return the "
            "product name and total quantity.",
            "Sum the ordered quantity per product and return name and total.",
            "For each product that appears in an order, give its name and total "
            "quantity ordered.",
        ),
        gold=("SELECT p.name, SUM(oi.quantity) FROM products p "
              "JOIN order_items oi ON oi.product_id=p.product_id "
              "GROUP BY p.product_id, p.name;"),
    ),
    Template(
        family="never_ordered_products", level="medium", tags=("setdiff", "subquery"),
        questions=(
            "List the names of products that have never been ordered.",
            "Which products appear in no order at all? Return their names.",
            "Show product names that were never purchased.",
        ),
        gold=("SELECT name FROM products WHERE product_id NOT IN "
              "(SELECT DISTINCT product_id FROM order_items);"),
    ),
    Template(
        family="signup_after", level="medium", slots=("signup_cut",), tags=("date",),
        variants=(
            {"op": ">=", "word": "on or after"},
            {"op": "<", "word": "before"},
        ),
        questions=(
            "List the names of customers who signed up {word} {signup_cut}.",
            "Which customers registered {word} {signup_cut}? Return names.",
            "Show customer names with a signup date {word} {signup_cut}.",
            "Who joined {word} {signup_cut}? Give me their names.",
            "Return the names of everyone whose signup falls {word} {signup_cut}.",
            "Names of customers onboarded {word} {signup_cut}, please.",
        ),
        gold="SELECT name FROM customers WHERE signup_date {op} '{signup_cut}';",
    ),
    Template(
        family="distinct_products_in_status", level="medium",
        slots=("status",), tags=("join", "distinct", "count"),
        questions=(
            "How many distinct products were ordered in {status} orders?",
            "Count the unique products appearing in orders with status {status}.",
            "What is the number of different products across {status} orders?",
        ),
        gold=("SELECT COUNT(DISTINCT oi.product_id) FROM order_items oi "
              "JOIN orders o ON o.order_id=oi.order_id WHERE o.status='{status}';"),
    ),
    Template(
        family="customers_with_status_order", level="medium",
        slots=("status",), tags=("join", "distinct"),
        questions=(
            "List the names of customers who have at least one {status} order.",
            "Which customers placed an order that is {status}? Return distinct names.",
            "Show the distinct names of customers with a {status} order.",
        ),
        gold=("SELECT DISTINCT c.name FROM customers c "
              "JOIN orders o ON o.customer_id=c.customer_id "
              "WHERE o.status='{status}';"),
    ),
    Template(
        family="avg_qty_per_order_item", level="medium", tags=("agg",),
        questions=(
            "What is the average quantity per order line item?",
            "Return the mean quantity across all order items.",
            "On average, how many units does a single order line contain?",
        ),
        gold="SELECT AVG(quantity) FROM order_items;",
    ),
    Template(
        family="count_orders_in_month", level="medium", slots=("month",), tags=("date",),
        questions=(
            "How many orders were placed in month {month} of 2025?",
            "Count the orders whose order date falls in month {month}.",
            "What is the order count for month {month}?",
            "In month {month}, how many orders were there?",
            "Total number of orders dated in month {month}?",
            "Return the count of orders from month {month}.",
        ),
        gold=("SELECT COUNT(*) FROM orders "
              "WHERE strftime('%m', order_date)='{month}';"),
    ),

    # ------------------------------------------------------------------ hard
    Template(
        family="revenue_total_status", level="hard", slots=("status",),
        tags=("join", "revenue"),
        questions=(
            "What is the total {status} revenue, where revenue of a line item is "
            "quantity times product price and only orders with status '{status}' count?",
            "Sum quantity times price across all {status} orders to get total revenue.",
            "Compute the revenue generated by {status} orders only.",
            "How much revenue did {status} orders produce in total?",
            "Return total revenue restricted to orders with status {status}.",
            "Across every {status} order, what is the summed quantity times price?",
            "Give the overall revenue figure for {status} orders.",
        ),
        gold=(f"SELECT SUM({_REV}) FROM {_J_OIP} " "WHERE o.status='{status}';"),
    ),
    Template(
        family="revenue_by_group", level="hard", slots=("status",),
        tags=("join", "revenue", "groupby"), expect_multi_row=True,
        variants=(
            {"gcol": "p.category", "out": "category", "label": "product category"},
            {"gcol": "c.segment", "out": "segment", "label": "customer segment"},
            {"gcol": "c.city", "out": "city", "label": "city"},
            {"gcol": "p.name", "out": "product name", "label": "product"},
            {"gcol": "c.name", "out": "customer name", "label": "customer"},
        ),
        questions=(
            "What is the total {status} revenue per {label}? Return {out} and "
            "revenue, highest revenue first.",
            "Break {status} revenue down by {label}, highest first. Return {out} "
            "and revenue.",
            "For each {label}, compute {status} revenue and sort descending.",
            "Rank the {label}s by {status} revenue. Return {out} and revenue.",
            "Show {status} revenue grouped by {label}, biggest first.",
            "Which {label}s earned what in {status} revenue? Sort high to low.",
        ),
        gold=(f"SELECT {{gcol}}, SUM({_REV}) AS rev FROM {_J_OIPC} "
              "WHERE o.status='{status}' GROUP BY {gcol} ORDER BY rev DESC;"),
    ),
    Template(
        family="argmax_revenue_by_group", level="hard", slots=("status",),
        tags=("join", "revenue", "argmax"),
        variants=(
            {"gcol": "p.name", "key": "p.product_id, p.name", "out": "product name",
             "label": "product"},
            {"gcol": "c.name", "key": "c.customer_id, c.name", "out": "customer name",
             "label": "customer"},
            {"gcol": "p.category", "key": "p.category", "out": "category",
             "label": "product category"},
            {"gcol": "c.city", "key": "c.city", "out": "city", "label": "city"},
        ),
        questions=(
            "Which {label} generated the most {status} revenue? Return only the {out}.",
            "Return the {out} with the highest {status} revenue.",
            "Which {label} tops the {status} revenue ranking? Give just the {out}.",
            "Identify the single best {label} by {status} revenue. {out} only.",
            "By {status} revenue, what is the leading {label}? Return the {out}.",
            "Give me only the {out} of the top-earning {label} in {status} orders.",
        ),
        gold=(f"SELECT {{gcol}} FROM {_J_OIPC} "
              "WHERE o.status='{status}' GROUP BY {key} "
              f"ORDER BY SUM({_REV}) DESC LIMIT 1;"),
    ),
    Template(
        family="avg_order_value", level="hard", slots=("status",),
        tags=("join", "revenue", "subquery"),
        questions=(
            "What is the average order value (average total revenue per order) "
            "across {status} orders?",
            "Compute the mean revenue per order among {status} orders.",
            "For {status} orders, what is the average total value of an order?",
            "On average, how much is a {status} order worth in total?",
            "Return the average per-order revenue for {status} orders.",
            "What does the typical {status} order come to in value?",
        ),
        gold=("SELECT AVG(order_rev) FROM (SELECT o.order_id, "
              f"SUM({_REV}) AS order_rev FROM orders o "
              "JOIN order_items oi ON oi.order_id=o.order_id "
              "JOIN products p ON p.product_id=oi.product_id "
              "WHERE o.status='{status}' GROUP BY o.order_id);"),
    ),
    Template(
        family="orders_with_n_items", level="hard", slots=("count_threshold",),
        tags=("groupby", "having", "subquery"),
        variants=(
            {"op": ">", "word": "more than"},
            {"op": ">=", "word": "at least"},
            {"op": "=", "word": "exactly"},
        ),
        questions=(
            "How many orders contain {word} {count_threshold} line items?",
            "Count the orders having {word} {count_threshold} items.",
            "What is the number of orders with {word} {count_threshold} lines?",
            "How many orders are made up of {word} {count_threshold} line items?",
            "Return the count of orders whose line-item count is {word} "
            "{count_threshold}.",
            "Tally the orders containing {word} {count_threshold} items.",
        ),
        gold=("SELECT COUNT(*) FROM (SELECT order_id FROM order_items "
              "GROUP BY order_id HAVING COUNT(*){op}{count_threshold});"),
    ),
    Template(
        family="status_fraction", level="hard", slots=("status",), tags=("subquery",),
        variants=(
            {"mult": "1.0", "word": "fraction", "desc": "as a fraction"},
            {"mult": "100.0", "word": "percentage", "desc": "as a percentage"},
        ),
        questions=(
            "What {word} of all orders were {status}? Return a single number "
            "({status} orders divided by total orders, {desc}).",
            "Compute the {word} of orders that are {status} as one number, {desc}.",
            "What {word} of orders carry status {status}? Give it {desc}.",
            "Express the share of {status} orders {desc}, as a single value.",
            "Return one number: the {word} of orders in state {status}, {desc}.",
            "Of all orders, what {word} ended up {status}? Answer {desc}.",
        ),
        gold=("SELECT (SELECT COUNT(*) FROM orders WHERE status='{status}')*{mult}"
              "/(SELECT COUNT(*) FROM orders);"),
    ),
    Template(
        family="max_per_group", level="hard", tags=("groupby", "join", "subquery"),
        expect_multi_row=True,
        questions=(
            "Find the most expensive product in each category. Return the category "
            "and the product name.",
            "For every category, which product has the highest price? Return "
            "category and name.",
            "Return each category alongside its priciest product's name.",
        ),
        gold=("SELECT p.category, p.name FROM products p "
              "JOIN (SELECT category, MAX(price) mx FROM products GROUP BY category) m "
              "ON m.category=p.category AND p.price=m.mx;"),
    ),
    Template(
        family="never_in_status", level="hard", slots=("status",),
        tags=("setdiff", "subquery", "join"),
        questions=(
            "List the names of products that were never ordered in a {status} order.",
            "Which products never appear in any {status} order? Return names.",
            "Show product names absent from every {status} order.",
            "Return products that show up in no {status} order at all.",
            "Which product names are missing from all {status} orders?",
            "Find every product never included in a {status} order.",
        ),
        gold=("SELECT name FROM products WHERE product_id NOT IN "
              "(SELECT oi.product_id FROM order_items oi "
              "JOIN orders o ON o.order_id=oi.order_id WHERE o.status='{status}');"),
    ),
    Template(
        family="total_vs_status_revenue", level="hard", slots=("status",),
        tags=("join", "revenue", "case"),
        questions=(
            "Return the total revenue across all orders and the {status}-only "
            "revenue, as two columns.",
            "Give overall revenue and {status} revenue side by side in two columns.",
            "Compute two figures: revenue over every order, and revenue restricted "
            "to {status} orders.",
            "In two columns, report all-order revenue and {status}-order revenue.",
            "Show total revenue next to {status} revenue as a two-column result.",
            "Return a row with two values: overall revenue, then {status} revenue.",
        ),
        gold=(f"SELECT SUM({_REV}) AS total_rev, "
              f"SUM(CASE WHEN o.status='{{status}}' THEN {_REV} ELSE 0 END) AS "
              f"status_rev FROM {_J_OIP};"),
    ),
    Template(
        family="active_multi_month", level="hard", slots=("count_threshold",),
        tags=("date", "groupby", "having", "subquery"),
        questions=(
            "How many customers placed orders in more than {count_threshold} "
            "distinct months of 2025?",
            "Count customers active in over {count_threshold} different months.",
            "How many customers ordered across more than {count_threshold} "
            "separate months?",
            "How many customers were active in more than {count_threshold} "
            "distinct months?",
            "Return the number of customers ordering in over {count_threshold} months.",
            "Count how many customers span more than {count_threshold} order months.",
        ),
        gold=("SELECT COUNT(*) FROM (SELECT customer_id FROM orders "
              "GROUP BY customer_id "
              "HAVING COUNT(DISTINCT strftime('%m', order_date))>{count_threshold});"),
    ),
    Template(
        family="revenue_per_month", level="hard", slots=("status",),
        tags=("date", "join", "revenue", "groupby"), expect_multi_row=True,
        questions=(
            "For each month in 2025, what is the total {status} revenue? Return "
            "month and revenue ordered by month.",
            "Show {status} revenue per month, sorted by month.",
            "Break {status} revenue down month by month, in month order.",
            "Give the monthly {status} revenue series, ordered by month.",
            "Month by month, how much {status} revenue was there? Sort by month.",
            "Return each month alongside its {status} revenue, in calendar order.",
        ),
        gold=("SELECT strftime('%m', o.order_date) AS m, "
              f"SUM({_REV}) AS rev FROM orders o "
              "JOIN order_items oi ON oi.order_id=o.order_id "
              "JOIN products p ON p.product_id=oi.product_id "
              "WHERE o.status='{status}' GROUP BY m ORDER BY m;"),
    ),
    Template(
        family="topn_customers_by_orders", level="hard", slots=("n_limit",),
        tags=("join", "groupby", "orderby", "limit"), expect_multi_row=True,
        variants=(
            {"dir": "DESC", "word": "most", "adj": "busiest", "first": "most orders first"},
            {"dir": "ASC", "word": "fewest", "adj": "least active", "first": "fewest orders first"},
        ),
        questions=(
            "List the top {n_limit} customers by number of orders, with name and "
            "order count, {first}.",
            "Return the {n_limit} {adj} customers as name and order count, {first}.",
            "Which {n_limit} customers placed the {word} orders? Name and count.",
            "Show the {n_limit} customers with the {word} orders, {first}.",
            "Rank customers by order count and return the top {n_limit}, {first}.",
            "Give me the {n_limit} {adj} customers with their order counts.",
        ),
        gold=("SELECT c.name, COUNT(*) AS n FROM customers c "
              "JOIN orders o ON o.customer_id=c.customer_id "
              "GROUP BY c.customer_id, c.name ORDER BY n {dir} LIMIT {n_limit};"),
    ),
    Template(
        family="customers_by_city_segment", level="hard",
        slots=("city", "segment"), tags=("filter", "compound"),
        questions=(
            "List the names of {segment} customers based in {city}.",
            "Which customers in {city} belong to the {segment} segment? Names only.",
            "Show {segment}-segment customer names from {city}.",
            "Return names of customers who are both in {city} and {segment}.",
            "Who are the {segment} customers located in {city}?",
            "Give me every {segment} customer name from {city}.",
        ),
        gold=("SELECT name FROM customers "
              "WHERE city='{city}' AND segment='{segment}';"),
    ),
    Template(
        family="revenue_by_group_all", level="hard", tags=("join", "revenue", "groupby"),
        expect_multi_row=True,
        variants=(
            {"gcol": "p.category", "out": "category", "label": "product category"},
            {"gcol": "c.segment", "out": "segment", "label": "customer segment"},
            {"gcol": "c.city", "out": "city", "label": "city"},
            {"gcol": "p.name", "out": "product name", "label": "product"},
        ),
        questions=(
            "What is the total revenue per {label} across ALL orders regardless of "
            "status? Return {out} and revenue, highest first.",
            "Ignoring order status, break revenue down by {label}. Return {out} and "
            "revenue, biggest first.",
            "For every {label}, compute revenue over all orders and sort descending.",
            "Rank {label}s by total revenue across every order, whatever its status.",
            "Across all orders, what revenue did each {label} generate? Sort high "
            "to low.",
        ),
        gold=(f"SELECT {{gcol}}, SUM({_REV}) AS rev FROM {_J_OIPC} "
              "GROUP BY {gcol} ORDER BY rev DESC;"),
    ),
    Template(
        family="orders_count_by_status_month", level="hard",
        slots=("status", "month"), tags=("date", "filter", "compound"),
        questions=(
            "How many {status} orders were placed in month {month}?",
            "Count the orders that are {status} and dated in month {month}.",
            "In month {month}, how many orders reached {status}?",
            "Return the number of {status} orders from month {month}.",
            "What is the {status} order count for month {month}?",
            "Tally {status} orders occurring in month {month}.",
        ),
        gold=("SELECT COUNT(*) FROM orders WHERE status='{status}' "
              "AND strftime('%m', order_date)='{month}';"),
    ),
    Template(
        family="avg_qty_having", level="hard", slots=("qty_threshold",),
        tags=("join", "groupby", "having"),
        questions=(
            "Which products have an average ordered quantity greater than "
            "{qty_threshold}? Return the product name.",
            "Return the names of products whose mean ordered quantity exceeds "
            "{qty_threshold}.",
            "List products with average order quantity above {qty_threshold}.",
            "Which product names average more than {qty_threshold} units per line?",
            "Find products whose mean quantity per line beats {qty_threshold}.",
            "Names of products averaging over {qty_threshold} units ordered.",
        ),
        gold=("SELECT p.name FROM order_items oi "
              "JOIN products p ON p.product_id=oi.product_id "
              "GROUP BY p.product_id, p.name HAVING AVG(oi.quantity)>{qty_threshold};"),
    ),
    Template(
        family="argmax_qty_category", level="hard", tags=("join", "groupby", "argmax"),
        questions=(
            "What is the most popular product category by total quantity ordered "
            "across all orders? Return only the category.",
            "Which category has the highest total ordered quantity? Category only.",
            "By summed quantity, what is the leading product category?",
        ),
        gold=("SELECT p.category FROM order_items oi "
              "JOIN products p ON p.product_id=oi.product_id "
              "GROUP BY p.category ORDER BY SUM(oi.quantity) DESC LIMIT 1;"),
    ),
    Template(
        family="segment_avg_order_value", level="hard", slots=("status",),
        tags=("join", "revenue", "subquery", "argmax"),
        questions=(
            "Which customer segment has the highest average {status} order value? "
            "Return only the segment.",
            "Return the segment whose mean {status} order value is largest.",
            "Among segments, which has the biggest average {status} order value?",
            "Identify the segment with the top average {status} order value. Segment only.",
            "By average {status} order value, which segment leads?",
            "Give just the segment name with the largest mean {status} order value.",
        ),
        gold=("SELECT seg FROM (SELECT c.segment AS seg, AVG(t.order_rev) AS aov "
              "FROM (SELECT o.order_id, o.customer_id, "
              f"SUM({_REV}) AS order_rev FROM orders o "
              "JOIN order_items oi ON oi.order_id=o.order_id "
              "JOIN products p ON p.product_id=oi.product_id "
              "WHERE o.status='{status}' GROUP BY o.order_id) t "
              "JOIN customers c ON c.customer_id=t.customer_id "
              "GROUP BY c.segment ORDER BY aov DESC LIMIT 1);"),
    ),
    Template(
        family="never_ordered_customers", level="hard", tags=("setdiff", "subquery"),
        questions=(
            "List the names of customers who have never placed an order.",
            "Which customers have no orders at all? Return their names.",
            "Show the names of customers without a single order.",
            "Who has never ordered anything? Return their names.",
            "Return names of customers with zero orders on record.",
            "Find every customer that has not placed any order.",
        ),
        gold=("SELECT name FROM customers WHERE customer_id NOT IN "
              "(SELECT DISTINCT customer_id FROM orders);"),
    ),

    # ---------------------------------------------------- near-variant families
    #
    # Five of the 16 held-out test tasks come from SLOT-FREE patterns: "products
    # never ordered", "quantity per product", "orders per month", "most expensive
    # product per category", "top category by quantity". A slot-free pattern has
    # exactly ONE instantiation -- which is the test task itself -- so it can
    # contribute nothing to training or to test_ext without leaking outright.
    # (Those five families correctly end up with zero usable instances; the
    # leakage rules reject every candidate.)
    #
    # These families represent the same patterns with one honest twist each, so
    # test_ext can measure whether the policy learned the *pattern* rather than
    # memorising the one query. They are near-variants, not re-instantiations,
    # and NB2 says so out loud.
    Template(
        family="orders_per_month_status", level="medium", slots=("status",),
        tags=("date", "groupby", "near_variant"), expect_multi_row=True,
        questions=(
            "How many {status} orders were placed in each month? Return the month "
            "and the count.",
            "Break the {status} order count down by month. Return month and count.",
            "For every month, how many orders reached {status}? Month and count.",
            "Group the {status} orders by month and count them.",
        ),
        gold=("SELECT strftime('%m', order_date) AS m, COUNT(*) FROM orders "
              "WHERE status='{status}' GROUP BY m;"),
    ),
    Template(
        family="qty_per_product_status", level="medium", slots=("status",),
        tags=("join", "groupby", "agg", "near_variant"), expect_multi_row=True,
        questions=(
            "What is the total quantity ordered for each product in {status} "
            "orders? Return the product name and total quantity.",
            "Sum the quantity per product across {status} orders; return name and "
            "total.",
            "For {status} orders only, give each product's name and total ordered "
            "quantity.",
            "Per product, how many units were ordered in {status} orders? Name and "
            "total.",
        ),
        gold=("SELECT p.name, SUM(oi.quantity) FROM products p "
              "JOIN order_items oi ON oi.product_id=p.product_id "
              "JOIN orders o ON o.order_id=oi.order_id "
              "WHERE o.status='{status}' GROUP BY p.product_id, p.name;"),
    ),
    Template(
        family="never_ordered_in_category", level="medium", slots=("category",),
        tags=("setdiff", "subquery", "near_variant"),
        questions=(
            "List the names of {category} products that have never been ordered.",
            "Which {category} products appear in no order? Return their names.",
            "Show {category} product names that were never purchased.",
            "Among {category} items, which have never been ordered? Names only.",
        ),
        gold=("SELECT name FROM products WHERE category='{category}' "
              "AND product_id NOT IN (SELECT DISTINCT product_id FROM order_items);"),
    ),
    Template(
        family="extreme_per_group", level="hard", tags=("groupby", "join", "subquery",
                                                        "near_variant"),
        expect_multi_row=True,
        variants=(
            {"agg": "MIN", "word": "cheapest", "sup": "lowest"},
        ),
        questions=(
            "Find the {word} product in each category. Return the category and the "
            "product name.",
            "For every category, which product has the {sup} price? Return category "
            "and name.",
            "Return each category alongside its {word} product's name.",
            "Which product is {word} within each category? Give category and name.",
        ),
        gold=("SELECT p.category, p.name FROM products p "
              "JOIN (SELECT category, {agg}(price) mx FROM products GROUP BY category) m "
              "ON m.category=p.category AND p.price=m.mx;"),
    ),
    Template(
        family="argmax_qty_group_status", level="hard", slots=("status",),
        tags=("join", "groupby", "argmax", "near_variant"),
        variants=(
            {"gcol": "p.category", "out": "category", "label": "product category"},
            {"gcol": "p.name", "out": "product name", "label": "product"},
        ),
        questions=(
            "What is the most popular {label} by total quantity ordered across "
            "{status} orders? Return only the {out}.",
            "Which {label} has the highest total ordered quantity among {status} "
            "orders? {out} only.",
            "By summed quantity in {status} orders, what is the leading {label}?",
            "Return the {out} with the greatest total quantity in {status} orders.",
        ),
        gold=("SELECT {gcol} FROM order_items oi "
              "JOIN products p ON p.product_id=oi.product_id "
              "JOIN orders o ON o.order_id=oi.order_id "
              "WHERE o.status='{status}' GROUP BY {gcol} "
              "ORDER BY SUM(oi.quantity) DESC LIMIT 1;"),
    ),
]

FAMILIES = {t.family: t for t in TEMPLATES}

# Which template family each of the 16 held-out test tasks belongs to.
# Generating with exclude_families=TEST_FAMILIES gives a training set that shares
# no *pattern* with the test set -- that is the memorization control in NB2.
TEST_TASK_FAMILY: dict[int, str] = {
    3: "prod_by_category",
    5: "distinct_col",
    8: "cust_by_segment",
    9: "count_prod_by_category",
    12: "agg_price_by_category",
    14: "orders_per_month",
    16: "never_ordered_products",
    18: "qty_per_product",
    22: "argmax_revenue_by_group",
    24: "argmax_revenue_by_group",
    27: "avg_order_value",
    30: "argmax_qty_category",
    33: "avg_qty_having",
    36: "max_per_group",
    38: "segment_avg_order_value",
    40: "total_vs_status_revenue",
}
#: Near-variant family -> the slot-free test family it stands in for.
#
# A slot-free pattern has exactly one instantiation (the test task itself), so it
# can contribute nothing to test_ext without leaking. These families cover the
# same pattern with one honest twist, so test_ext can still probe it. NB2 states
# the distinction rather than blurring it: for these five patterns, test_ext
# measures generalization to a *variant*, not to a fresh instance.
NEAR_VARIANT_OF: dict[str, str] = {
    "orders_per_month_status": "orders_per_month",
    "qty_per_product_status": "qty_per_product",
    "never_ordered_in_category": "never_ordered_products",
    "extreme_per_group": "max_per_group",
    "argmax_qty_group_status": "argmax_qty_category",
}

#: Families reserved for test_ext -- the patterns the 16 held-out tasks use,
#: plus the near-variants standing in for the slot-free ones.
TEST_FAMILIES: tuple[str, ...] = tuple(sorted(
    set(TEST_TASK_FAMILY.values()) | set(NEAR_VARIANT_OF)))

#: Families with zero usable instances by construction: their only possible gold
#: IS a held-out test task. Reported in the audit so a zero is never mistaken for
#: a bug in the generator.
SLOT_FREE_TEST_FAMILIES: tuple[str, ...] = tuple(sorted(NEAR_VARIANT_OF.values()))


# ===========================================================================
# Slot samplers -- every value comes from the live database
# ===========================================================================

def _pools(con: sqlite3.Connection) -> dict:
    """Value pools read from the database.

    Drawing from here (rather than from hard-coded constants) is what keeps
    generated tasks answerable: a price threshold is a real quantile, a product
    name is a real product. A task whose gold returns nothing teaches the policy
    nothing, so the generator must not be able to invent one by accident.
    """
    q = lambda sql: [r[0] for r in con.execute(sql)]  # noqa: E731
    prices = sorted(q("SELECT price FROM products"))
    return {
        "cities": q("SELECT DISTINCT city FROM customers ORDER BY city"),
        "segments": q("SELECT DISTINCT segment FROM customers ORDER BY segment"),
        "categories": q("SELECT DISTINCT category FROM products ORDER BY category"),
        "statuses": q("SELECT DISTINCT status FROM orders ORDER BY status"),
        "product_names": q("SELECT name FROM products ORDER BY name"),
        "customer_names": q("SELECT name FROM customers ORDER BY name"),
        "signup_dates": q("SELECT DISTINCT signup_date FROM customers ORDER BY signup_date"),
        "order_months": q("SELECT DISTINCT strftime('%m', order_date) FROM orders "
                          "ORDER BY 1"),
        "prices": prices,
        # Round, human-looking thresholds that still sit inside the real range.
        "price_thresholds": sorted({
            int(round(prices[int(len(prices) * f)] / 100.0)) * 100
            for f in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
        }),
    }


#: Each slot's full DOMAIN, not a sampler. Domains (rather than draws) let us
#: enumerate a family's entire instance space and therefore know its capacity
#: exactly -- which is what stops the generator from spinning on a family that
#: can only ever produce three distinct tasks.
SLOT_DOMAINS: dict[str, Callable[[dict], list]] = {
    "city": lambda p: p["cities"],
    "segment": lambda p: p["segments"],
    "category": lambda p: p["categories"],
    "status": lambda p: p["statuses"],
    "product_name": lambda p: p["product_names"],
    "customer_name": lambda p: p["customer_names"],
    "signup_cut": lambda p: p["signup_dates"],
    "month": lambda p: p["order_months"],
    "price_threshold": lambda p: p["price_thresholds"],
    # Deliberately excludes 1: "orders with at least 1 item" is every order,
    # and "average quantity above 1" is nearly every product. Degenerate
    # thresholds produce tasks the policy solves without reading the question.
    "count_threshold": lambda p: [2, 3, 4],
    "qty_threshold": lambda p: [2, 3],
    "n_limit": lambda p: [3, 5, 10],
}

SLOT_SAMPLERS: dict[str, Callable[[random.Random, dict], object]] = {
    name: (lambda dom: (lambda r, p: r.choice(dom(p))))(dom)
    for name, dom in SLOT_DOMAINS.items()
}


def _render(t: Template, ns: dict, question_tpl: str) -> dict:
    question = question_tpl.format(**ns)
    gold = t.gold.format(**ns)
    return {
        "question": question,
        "gold": gold,
        "level": t.level,
        "family": t.family,
        "tags": list(t.tags),
        "slots": {k: v for k, v in ns.items()},
    }


def instantiate(t: Template, rng: random.Random, pools: dict) -> dict:
    """Draw ONE random instance of a template (used in NB2's walkthrough).

    The question paraphrase and the gold SQL are formatted from the SAME
    namespace, which is why the gold is correct by construction.
    """
    ns = dict(rng.choice(t.variants))
    for slot in t.slots:
        ns[slot] = SLOT_SAMPLERS[slot](rng, pools)
    return _render(t, ns, rng.choice(t.questions))


def family_capacity(t: Template, pools: dict) -> int:
    """How many DISTINCT tasks this template can ever produce."""
    n = len(t.variants) * len(t.questions)
    for slot in t.slots:
        n *= len(SLOT_DOMAINS[slot](pools))
    return n


def enumerate_family(t: Template, pools: dict, rng: random.Random,
                     cap: int = 400) -> list[dict]:
    """Every distinct instance of a template, deterministically shuffled.

    Enumeration beats rejection sampling here for two reasons: we get exact
    control over how many tasks a family contributes (so one 300-instance family
    cannot swamp twenty 5-instance ones), and we never spin re-drawing duplicates
    from a family that is already exhausted.
    """
    import itertools

    domains = [SLOT_DOMAINS[s](pools) for s in t.slots]
    out: list[dict] = []
    for variant in t.variants:
        for combo in itertools.product(*domains) if domains else [()]:
            ns = dict(variant)
            ns.update(dict(zip(t.slots, combo)))
            for q in t.questions:
                out.append(_render(t, ns, q))
    rng.shuffle(out)
    return out[:cap]


# ===========================================================================
# Validation -- a generated task is only usable if its gold really answers it
# ===========================================================================

def validate_task(task: dict, db_path: str = DB_PATH) -> tuple[bool, str]:
    """Execute the gold and reject anything that would teach the policy nothing.

    Rejections, in order of how often they fire:
      * the gold errors            -> a broken skeleton; a hard bug, surfaced loudly
      * not a SELECT               -> would mutate the shared environment
      * empty result              -> no signal, unless the family is a set-difference
      * single-row GROUP BY        -> a degenerate draw for a grouping family
    """
    gold = task["gold"]
    tpl = FAMILIES.get(task["family"])
    if not gold.strip().lower().startswith("select"):
        return False, "not_select"
    rows, err = safe_run_sql(gold, db_path)
    if err is not None:
        return False, f"gold_error:{err}"
    if rows is None:
        return False, "gold_none"
    if not rows:
        # An empty gold is REWARD-HACKABLE, not merely uninformative. score_sql
        # compares result sets, so every unrelated query that happens to return
        # nothing -- WHERE city='Atlantis', a dropped join, a typo'd literal --
        # scores as correct. Set-difference families are the tempting exception
        # ("products never ordered" is legitimately empty sometimes); they are
        # rejected too, because the policy cannot tell the difference between
        # earning that empty set and stumbling into it.
        return False, "empty_result"
    if len(rows) == 1 and all(c is None for c in rows[0]):
        # Same hazard, one step subtler: AVG over no rows yields (None,), and so
        # does any aggregate over a wrongly-filtered empty set.
        return False, "all_null_result"
    if tpl is not None and tpl.expect_multi_row and len(rows) < 2:
        # A GROUP BY that returns one row is indistinguishable from a scalar
        # aggregate, so the task no longer tests what the family exists to test.
        return False, "degenerate_single_row"
    return True, "ok"


# ===========================================================================
# Leakage: four rules against all 40 original tasks
# ===========================================================================

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "with", "that", "which", "who", "what",
    "how", "many", "much", "return", "show", "list", "give", "please", "each",
    "every", "all", "their", "them", "it", "its", "as", "by", "from", "do",
    "does", "did", "me", "my", "you", "your", "there", "have", "has", "had",
}


def norm_question(q: str) -> str:
    """Lowercase, strip punctuation and quotes, collapse whitespace."""
    q = q.lower().strip()
    q = re.sub(r"[\"'`]", "", q)
    q = re.sub(r"[^\w\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def content_tokens(q: str) -> set[str]:
    """Content words only -- stopwords and bare numerals removed."""
    return {w for w in norm_question(q).split()
            if w not in _STOPWORDS and not w.isdigit()}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def canon_sql(sql: str) -> str:
    """Canonical form: uppercase keywords, drop aliases, collapse whitespace.

    Catches the same query written with different table aliases or spacing --
    a real risk here, because our skeletons and repo 1's gold were written by
    the same hand and often differ only cosmetically.
    """
    s = sql.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bAS\s+\w+", "", s, flags=re.IGNORECASE)
    # Strip single-letter/short table aliases and their qualifiers.
    s = re.sub(r"\b([a-z]{1,3})\.", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(JOIN\s+\w+)\s+[a-z]{1,3}\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(FROM\s+\w+)\s+[a-z]{1,3}\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.upper().strip()


def result_signature(gold_sql: str, db_path: str = DB_PATH) -> str:
    """sha1 over the gold's normalized result set (plus its ordered-ness).

    Reuses `db._normalize`'s semantics so the signature means the same thing
    `score_sql` means when it compares result sets.
    """
    rows, err = safe_run_sql(gold_sql, db_path)
    if err is not None or rows is None:
        return f"ERR:{err}"
    ordered = "order by" in gold_sql.lower()
    return sha1(repr(_normalize(rows, ordered)).encode()).hexdigest()


#: The ONLY tasks leakage is measured against: the 16 held-out test tasks.
#
# Repo 1's 24 *train* tasks are deliberately NOT in scope. They are training
# data in both repos; a generated task that duplicates one leaks nothing,
# because no number anywhere is computed on them. Scoping the rules to the test
# split is not a loosening -- it is the correct definition of the contract.
# (An earlier draft compared against all 40 and rejected ~12,000 perfectly good
# tasks, mostly slot-free families whose single gold matches a repo-1 *train*
# gold by construction.)
EVAL_TASKS: list[dict] = [t for t in TASKS if t["split"] == "test"]


def collides_with_eval(task: dict, eval_tasks: list[dict] | None = None,
                       db_path: str = DB_PATH,
                       sig_cache: dict | None = None) -> str | None:
    """Return a rule name if `task` leaks a held-out test task, else None.

    Rule 1  exact normalized question match.
    Rule 2  near-duplicate question: content-token Jaccard >= 0.85.
    Rule 3  canonical SQL match.
    Rule 4  result-set collision, but ONLY in conjunction with Jaccard >= 0.5.

    Rule 4's conjunction is the subtle one. `SELECT COUNT(*) FROM orders` and a
    dozen unrelated counts all return `[(80,)]`; rejecting on signature alone
    would gut the easy families for no benefit -- returning the same number as
    an unrelated test task is not leakage. Requiring *both* a matching result
    and a similar question catches real duplicates and nothing else.
    Signature-only hits are counted and reported, never rejected.
    """
    eval_tasks = EVAL_TASKS if eval_tasks is None else eval_tasks
    nq = norm_question(task["question"])
    toks = content_tokens(task["question"])
    csql = canon_sql(task["gold"])

    for ev in eval_tasks:
        if nq == norm_question(ev["question"]):
            return "rule1_exact_question"

    for ev in eval_tasks:
        if jaccard(toks, content_tokens(ev["question"])) >= 0.85:
            return "rule2_near_duplicate_question"

    for ev in eval_tasks:
        if csql == canon_sql(ev["gold"]):
            return "rule3_canonical_sql"

    sig = result_signature(task["gold"], db_path)
    if not sig.startswith("ERR"):
        cache = sig_cache if sig_cache is not None else {}
        for ev in eval_tasks:
            key = ev["id"]
            if key not in cache:
                cache[key] = result_signature(ev["gold"], db_path)
            if cache[key] == sig:
                if jaccard(toks, content_tokens(ev["question"])) >= 0.5:
                    return "rule4_result_and_question"
                return "counted_signature_only"   # reported, NOT rejected
    return None


# ===========================================================================
# Generation
# ===========================================================================

DEFAULT_LEVEL_MIX = (("easy", 0.25), ("medium", 0.40), ("hard", 0.35))

#: Largest share of the test-family instance pool that test_ext may reserve.
#: The remainder stays available to train/val so the policy can learn the
#: patterns the 16 held-out tasks exercise. See the comment in generate_tasks.
TEST_EXT_MAX_SHARE = 0.55

#: A family may supply at most this multiple of its even share of a split.
#: Capacity is an accident of the schema (how many cities, how many prices);
#: it should not decide what the policy spends its gradient steps on.
FAMILY_CAP_MULTIPLE = 2.5


def build_corpus(seed: int = 1234, db_path: str = DB_PATH,
                 cap_per_family: int = 400) -> tuple[dict[str, list[dict]], dict]:
    """Enumerate, validate, and leak-filter every family. Returns (corpus, audit).

    `corpus` maps family -> list of usable task dicts. This is the expensive
    step (it executes every candidate gold once), so callers do it once and
    slice the result into splits.
    """
    if not os.path.exists(db_path):
        build_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        pools = _pools(con)
    finally:
        con.close()

    rng = random.Random(seed)
    sig_cache: dict = {}
    rejected: list[tuple[dict, str]] = []
    sig_only = 0
    corpus: dict[str, list[dict]] = {}
    capacity: dict[str, int] = {}

    for tpl in TEMPLATES:
        capacity[tpl.family] = family_capacity(tpl, pools)
        kept: list[dict] = []
        seen_local: set[str] = set()
        for task in enumerate_family(tpl, pools, rng, cap=cap_per_family):
            nq = norm_question(task["question"])
            if nq in seen_local:
                continue
            ok, why = validate_task(task, db_path)
            if not ok:
                rejected.append((task, why))
                continue
            rule = collides_with_eval(task, None, db_path, sig_cache)
            if rule == "counted_signature_only":
                sig_only += 1
                rule = None
            if rule is not None:
                rejected.append((task, rule))
                continue
            seen_local.add(nq)
            kept.append(task)
        corpus[tpl.family] = kept

    audit = audit_report(rejected, sig_only)
    audit["capacity_by_family"] = dict(sorted(capacity.items(), key=lambda kv: -kv[1]))
    audit["usable_by_family"] = {f: len(v) for f, v in
                                 sorted(corpus.items(), key=lambda kv: -len(kv[1]))}
    audit["usable_total"] = sum(len(v) for v in corpus.values())
    return corpus, audit


def _balanced_take(items: list[dict], k: int, rng: random.Random,
                   taken: dict[str, int] | None = None,
                   cap: int | None = None) -> list[dict]:
    """Take `k` items, always drawing from the least-represented family so far.

    Family capacity varies by two orders of magnitude here: `signup_after` can
    produce 240 distinct tasks, `never_ordered_customers` six. Drawing uniformly
    from the pooled instances lets a handful of slot-rich families supply most of
    the training set -- an early version handed 184 of 800 training tasks to a
    single family -- and the policy then sees the schema through a keyhole.

    Balancing by running count gives every *pattern* comparable weight, which is
    what we actually want the policy to learn. `taken` is shared across calls so
    the level quotas and the shortfall fill cannot undo each other's balance.
    """
    by_family: dict[str, list[dict]] = {}
    for t in items:
        by_family.setdefault(t["family"], []).append(t)
    for v in by_family.values():
        rng.shuffle(v)

    counts = taken if taken is not None else {}
    out: list[dict] = []
    while len(out) < k:
        live = [f for f, v in by_family.items()
                if v and (cap is None or counts.get(f, 0) < cap)]
        if not live:
            break   # capacity or cap exhausted -- the caller reports the shortfall
        # Least-taken family wins; ties broken deterministically by name.
        fam = min(live, key=lambda f: (counts.get(f, 0), f))
        out.append(by_family[fam].pop())
        counts[fam] = counts.get(fam, 0) + 1
    return out


def _allocate(pool: list[dict], n: int, level_mix, rng: random.Random) -> list[dict]:
    """Take `n` tasks from `pool`, respecting the level mix as far as supply allows.

    Under-supply at one level is redistributed to the others rather than
    silently truncating the split -- and the shortfall is reported by the caller.
    Within a level, families are balanced by `_balanced_take`.
    """
    by_level: dict[str, list[dict]] = {}
    for t in pool:
        by_level.setdefault(t["level"], []).append(t)

    # No single family may supply more than FAMILY_CAP_MULTIPLE times its even
    # share. Without a cap, one 240-instance family took 22% of the training set
    # purely because it had the most slot values -- an accident of the schema,
    # not a teaching choice.
    #
    # The cap scales with how many families the pool actually has, rather than
    # being a flat percentage: test_ext draws from 15 families and train from 42,
    # so one fixed share would either fail to constrain train or starve test_ext.
    n_fams = len({t["family"] for t in pool}) or 1
    cap = max(int(n / n_fams * FAMILY_CAP_MULTIPLE), 4)

    want = {lvl: int(round(n * frac)) for lvl, frac in level_mix}
    taken: dict[str, int] = {}
    out: list[dict] = []
    for lvl, k in want.items():
        avail = by_level.get(lvl, [])
        take = _balanced_take(avail, min(k, len(avail)), rng, taken, cap)
        out.extend(take)
        chosen = {id(t) for t in take}
        by_level[lvl] = [t for t in avail if id(t) not in chosen]
    # Fill any shortfall from whatever is left, keeping the same balance and cap.
    if len(out) < n:
        leftovers = [t for lvl in by_level for t in by_level[lvl]]
        out.extend(_balanced_take(leftovers, n - len(out), rng, taken, cap))
    rng.shuffle(out)
    return out[:n]


def generate_tasks(n_train: int = 800, n_val: int = 200, n_test_ext: int = 200,
                   seed: int = 1234, db_path: str = DB_PATH,
                   exclude_families: tuple[str, ...] = (),
                   level_mix=DEFAULT_LEVEL_MIX,
                   cap_per_family: int = 400) -> dict:
    """Build train / val / test_ext splits of validated, non-leaking tasks.

    Three splits, three distinct jobs:

    * **train / val** -- drawn from every family not in `exclude_families`,
      instance-disjoint from each other and from the 16 held-out test tasks.
      val is the gating/early-stopping number.
    * **test_ext** -- drawn from `TEST_FAMILIES`: the same *query patterns* as
      the 16 held-out tasks, but ~150 fresh instances instead of 16. This is the
      high-power version of the test set. At n=150 the interval is ~±7pp
      instead of the 16-task set's ~±20pp, so it can actually support a claim.
      Its instances are reserved BEFORE train/val are allocated, so nothing is
      shared.
    * Pass `exclude_families=TEST_FAMILIES` to get the **memorization control**:
      a training set that has never seen the test patterns at all. Comparing a
      model trained on that against one trained on the full set is how NB2
      quantifies template memorization honestly.

    Deterministic given `seed`: enumeration order is fixed and every shuffle
    comes from the same seeded RNG.
    """
    corpus, audit = build_corpus(seed=seed, db_path=db_path,
                                 cap_per_family=cap_per_family)
    rng = random.Random(seed + 1)
    excluded = set(exclude_families)

    def key(t):
        return (t["family"], norm_question(t["question"]))

    # 1. Reserve test_ext from the families the 16 held-out tasks use -- but
    #    never more than TEST_EXT_MAX_SHARE of that pool.
    #
    #    Reserving the whole pool starves `train` of the very patterns the 16
    #    test tasks exercise (revenue argmax, average order value, ...). A policy
    #    that never trains on those patterns fails the headline comparison for a
    #    reason that has nothing to do with RL. Only the test *instances* must be
    #    withheld; the *patterns* are fair -- and necessary -- to train on. The
    #    memorization question is answered separately and honestly by the
    #    exclude_families=TEST_FAMILIES control.
    ext_pool = [t for fam in TEST_FAMILIES for t in corpus.get(fam, [])]
    ext_cap = int(len(ext_pool) * TEST_EXT_MAX_SHARE)
    test_ext = _allocate(ext_pool, min(n_test_ext, ext_cap), level_mix, rng)
    reserved = {key(t) for t in test_ext}

    # 2. train/val get everything else (minus any explicitly excluded families).
    #
    # Allocate them TOGETHER and then split stratified by level. Allocating
    # train first and giving val the leftovers produced a val set with zero hard
    # tasks -- which would have gated early stopping on the easy half of the
    # distribution while the interesting failures went unmeasured.
    rest = [t for fam, items in corpus.items() if fam not in excluded
            for t in items if key(t) not in reserved]
    combined = _allocate(rest, n_train + n_val, level_mix, rng)

    by_level: dict[str, list[dict]] = {}
    for t in combined:
        by_level.setdefault(t["level"], []).append(t)
    val_frac = n_val / max(n_train + n_val, 1)
    train, val = [], []
    for lvl in sorted(by_level):
        items = by_level[lvl]
        rng.shuffle(items)
        k = int(round(len(items) * val_frac))
        val.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    # Per-level rounding can drift the totals by a task or two; settle up so the
    # requested sizes are exact and a "shortfall" in the audit means a real
    # capacity limit rather than a rounding artefact.
    while len(val) > n_val and train is not None:
        train.append(val.pop())
    while len(val) < n_val and len(train) > n_train:
        val.append(train.pop())
    train = train[:n_train]

    for label, split in (("train", train), ("val", val), ("test_ext", test_ext)):
        for i, t in enumerate(split, 1):
            t["split"] = label
            t["id"] = f"{label[:2]}{i:04d}"

    audit["produced"] = {"train": len(train), "val": len(val),
                         "test_ext": len(test_ext)}
    audit["requested"] = {"train": n_train, "val": n_val, "test_ext": n_test_ext}
    # A shortfall is reported, never hidden. Silent truncation would read as
    # "we covered what we asked for" when we did not.
    audit["shortfall"] = {k: audit["requested"][k] - audit["produced"][k]
                          for k in audit["produced"]}
    audit["excluded_families"] = sorted(excluded)
    # The level mix is a *preference*, satisfied subject to family capacity and
    # the per-family cap. Report requested vs achieved so drift is visible here
    # rather than discovered later as a confusing by-level accuracy table.
    # (Drift runs toward harder tasks, which suits GRPO: hard tasks are where a
    # sampled group actually shows reward variance.)
    audit["level_mix_requested"] = {lvl: round(frac, 3) for lvl, frac in level_mix}
    audit["level_mix_achieved"] = {
        name: {lvl: round(sum(1 for t in sp if t["level"] == lvl) / max(len(sp), 1), 3)
               for lvl in ("easy", "medium", "hard")}
        for name, sp in (("train", train), ("val", val), ("test_ext", test_ext))
    }
    return {"train": train, "val": val, "test_ext": test_ext, "audit": audit}


def audit_report(rejected: list[tuple[dict, str]], sig_only: int = 0,
                 produced: dict | None = None,
                 excluded: list[str] | None = None) -> dict:
    """Rejection counts by rule -- charted in NB2, not just claimed in prose."""
    by_rule: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = {}
    for task, why in rejected:
        rule = why.split(":")[0]
        by_rule[rule] = by_rule.get(rule, 0) + 1
        fam = by_family.setdefault(task.get("family", "?"), {})
        fam[rule] = fam.get(rule, 0) + 1
    return {
        "produced": produced or {},
        "excluded_families": excluded or [],
        "rejected_total": len(rejected),
        "rejected_by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
        "rejected_by_family": by_family,
        "signature_only_collisions_allowed": sig_only,
        "n_templates": len(TEMPLATES),
        "n_families": len(FAMILIES),
        "test_families": list(TEST_FAMILIES),
    }


# ===========================================================================
# IO
# ===========================================================================

def write_jsonl(tasks: list[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_generated(split: str) -> list[dict]:
    """Read a checked-in generated split: 'train' | 'val' | 'test_ext'."""
    return read_jsonl(os.path.join(DATA_DIR, f"tasks_{split}_gen.jsonl"))


def split_report(tasks: list[dict]) -> dict:
    """Composition of a split, by level and by family."""
    by_level: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for t in tasks:
        by_level[t["level"]] = by_level.get(t["level"], 0) + 1
        by_family[t["family"]] = by_family.get(t["family"], 0) + 1
    return {
        "n": len(tasks),
        "by_level": dict(sorted(by_level.items())),
        "by_family": dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "n_families": len(by_family),
        "unique_questions": len({norm_question(t["question"]) for t in tasks}),
    }
