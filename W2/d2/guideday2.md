# Day 2: Aggregation & Joins — Comprehensive Guide & Reference (`guideday2`)

Welcome to **Day 2** of Week 2! Today is focused on the engine room of data transformation, business intelligence, and relational data processing: **Aggregation, Reshaping, and Merging**.

---

## Table of Contents
1. [The Split-Apply-Combine Philosophy](#1-the-split-apply-combine-philosophy)
2. [Modern Group Aggregation: Named Aggregation (`agg`)](#2-modern-group-aggregation-named-aggregation-agg)
3. [Window & Broadcast Operations: `.transform()`](#3-window--broadcast-operations-transform)
4. [Group-Level Filtering: `.filter()`](#4-group-level-filtering-filter)
5. [Reshaping Data: `pivot_table` vs `crosstab`](#5-reshaping-data-pivot_table-vs-crosstab)
6. [Relational Joins with `pd.merge()`](#6-relational-joins-with-pdmerge)
7. [Defensive Merging: The `validate` Parameter](#7-defensive-merging-the-validate-parameter)
8. [Anti-Joins & Audit Trails with `indicator=True`](#8-anti-joins--audit-trails-with-indicatortrue)
9. [Combining DataFrames: `pd.concat` & `.join()`](#9-combining-dataframes-pdconcat--join)
10. [Why Avoid `.apply()`? The Cython Fast Path](#10-why-avoid-apply-the-cython-fast-path)
11. [The 6 Critical Gotchas & Best Practices](#11-the-6-critical-gotchas--best-practices)
12. [20-Problem Aggregation & Joins Drill Guide](#12-20-problem-aggregation--joins-drill-guide)

---

## 1. The Split-Apply-Combine Philosophy

Coined by Hadley Wickham, the **Split-Apply-Combine** pattern is the core architecture behind group operations in tabular analysis:

1. **Split**: Break a `DataFrame` into smaller subsets based on the keys of one or more columns.
2. **Apply**: Execute an independent operation on each subset:
   - **Aggregation**: Reduce each group to a single summary value (e.g., `sum`, `mean`, `max`, `count`). Returns 1 row per group.
   - **Transformation**: Compute group-aware metrics broadcast back to the original group shape (e.g., group z-score, percentage of group total). Returns the exact same number of rows as the input.
   - **Filtration**: Discard entire groups based on a boolean condition evaluated on the aggregate (e.g., keep groups with at least 10 transactions).
3. **Combine**: Assemble the individual results into a single clean data structure.

```
Original DataFrame
[ Dept: A, Sales: 10 ]  ---\   Split by Dept   /--- Dept A [ 10, 30 ] -> Sum: 40  ---\ Combine
[ Dept: B, Sales: 50 ]  --->                 --->                                    ---> [ A: 40 ]
[ Dept: A, Sales: 30 ]  ---/                   \--- Dept B [ 50 ]     -> Sum: 50  ---/     [ B: 50 ]
```

---

## 2. Modern Group Aggregation: Named Aggregation (`agg`)

Prior to Pandas 0.25, aggregating multiple columns with custom target names produced messy MultiIndex column tuples that required awkward renaming. 

Pandas introduced **Named Aggregation** (`namedagg`), which is the modern standard for writing readable, production-grade pipelines.

### 2.1 The Named Aggregation Syntax
Use keyword arguments inside `.agg()` where:
- The **keyword** is the desired new column name.
- The **value** is a tuple: `(source_column_name, aggregation_function)`.

```python
import pandas as pd

df = pd.DataFrame({
    'category': ['Electronics', 'Electronics', 'Clothing', 'Clothing', 'Home'],
    'price': [1200, 150, 45, 90, 300],
    'quantity': [1, 4, 3, 2, 5],
    'rating': [4.5, 4.0, 3.8, 4.2, 4.8]
})

# Named aggregation: clean flat column names in one step!
summary = df.groupby('category', as_index=False).agg(
    total_revenue=('price', lambda s: (s * df.loc[s.index, 'quantity']).sum()),
    avg_price=('price', 'mean'),
    max_rating=('rating', 'max'),
    order_count=('price', 'count')
)
```

### 2.2 Built-in Fast Aggregations (String Identifiers)
Always prefer string identifiers when available. Pandas routes these to highly optimized C/Cython kernels:

| String Name | Description | Handles NaNs |
| :--- | :--- | :--- |
| `'mean'` | Arithmetic average | Yes (skipped by default) |
| `'sum'` | Total sum | Yes |
| `'median'` | 50th percentile | Yes |
| `'std'`, `'var'`| Standard deviation & variance | Yes |
| `'min'`, `'max'`| Extremum values | Yes |
| `'count'` | Number of non-null elements | Excludes NaNs |
| `'size'` | Total row count (including NaNs) | Includes NaNs |
| `'first'`, `'last'` | First/last valid value | Yes |
| `'nunique'` | Number of unique non-null elements | Excludes NaNs |

### 2.3 `as_index=False` vs `.reset_index()`
When grouping, Pandas puts grouping columns into the index by default:
```python
# Result index is 'category':
df.groupby('category').agg(avg_price=('price', 'mean'))

# Result keeps 'category' as a regular column (SQL-style):
df.groupby('category', as_index=False).agg(avg_price=('price', 'mean'))
```

---

## 3. Window & Broadcast Operations: `.transform()`

While `.agg()` collapses $N$ rows per group into $1$ row, `.transform()` executes an operation across each group and **broadcasts the result back to the original index**.

The output of `.transform()` **always has the exact same row count and index alignment** as the input DataFrame.

### 3.1 Common Use Cases for `.transform()`

#### 1. Calculating Percentage / Share of Group Total
```python
# Compute share of category revenue for each individual product
df['cat_total_price'] = df.groupby('category')['price'].transform('sum')
df['pct_of_cat_price'] = (df['price'] / df['cat_total_price']) * 100
```

#### 2. Group Standardization (Z-Score Normalization per Group)
```python
# Standardize price relative to other products in the same category
cat_mean = df.groupby('category')['price'].transform('mean')
cat_std = df.groupby('category')['price'].transform('std')
df['price_zscore_cat'] = (df['price'] - cat_mean) / cat_std
```

#### 3. Group Imputation (Imputing Missing Values by Group Median)
```python
# Fill missing salary with department median
df['salary'] = df['salary'].fillna(
    df.groupby('department')['salary'].transform('median')
)
```

> [!TIP]
> **Why use `.transform('mean')` instead of looping or `.apply()`?**
> Built-in transform strings like `'sum'`, `'mean'`, `'min'`, `'max'`, `'rank'`, and `'shift'` run in Cython with vectorized memory layouts, often 50x–200x faster than python loops or user-defined lambdas.

---

## 4. Group-Level Filtering: `.filter()`

`.filter()` takes a function that returns a single boolean scalar for each sub-DataFrame. 
If the function evaluates to `True`, all rows belonging to that group are retained; if `False`, all rows of that group are discarded.

```python
# Keep only departments with at least 5 employees
df_large_depts = df.groupby('department').filter(lambda g: len(g) >= 5)

# Keep only product categories with total sales exceeding $100,000
df_high_volume = df.groupby('category').filter(lambda g: (g['price'] * g['quantity']).sum() > 100000)
```

> [!NOTE]
> `.filter()` preserves the original index and schema of the original DataFrame.

---

## 5. Reshaping Data: `pivot_table` vs `crosstab`

Both tools convert long-format data into wide-format summary matrices, but have distinct intentions and optimizations.

### 5.1 `pd.pivot_table()`
Best for aggregating continuous numeric values across two categorical dimensions (row index and column headers).

```python
pivot = df.pivot_table(
    index='department',
    columns='gender',
    values='salary',
    aggfunc='mean',       # default is 'mean'
    fill_value=0,         # replace NaN cells with 0
    margins=True,         # add 'All' total row and column
    margins_name='Total'  # label for the margin
)
```

Key Parameters:
- `index`: Column(s) to group by on the rows (Y-axis).
- `columns`: Column(s) to spread out into headers (X-axis).
- `values`: Column(s) to aggregate.
- `aggfunc`: Function or list/dict of functions (e.g., `'sum'`, `['mean', 'count']`).
- `fill_value`: Value to substitute for empty combinations.
- `margins`: Boolean to generate subtotal/grand total rows and columns.

### 5.2 `pd.crosstab()`
Specifically tailored for categorical frequency counts, contingency tables, and cross-tabulation percentages.

```python
# Frequency count table of Department vs City
ct = pd.crosstab(
    index=df['department'],
    columns=df['city'],
    margins=True
)

# Relative proportions (normalize by rows or columns)
# normalize='index' ensures each row sums to 1.0 (100%)
ct_prop = pd.crosstab(
    index=df['department'],
    columns=df['city'],
    normalize='index'
) * 100
```

`normalize` options in `crosstab`:
- `normalize='all'` or `True`: Divides all cells by the grand total.
- `normalize='index'`: Divides each cell by its row sum (row percentages).
- `normalize='columns'`: Divides each cell by its column sum (column percentages).

---

## 6. Relational Joins with `pd.merge()`

`pd.merge()` is Pandas' implementation of relational database joins (SQL).

```python
merged = pd.merge(
    left=orders,
    right=customers,
    how='inner',
    on='customer_id'
)
```

### 6.1 The 5 Join Types (`how=...`)

| Join Type | Pandas Syntax | SQL Equivalent | Behavior |
| :--- | :--- | :--- | :--- |
| **Inner** | `how='inner'` | `INNER JOIN` | Keep only keys present in **both** tables (intersection). |
| **Left** | `how='left'` | `LEFT OUTER JOIN` | Keep **all** rows from left; fill missing right values with `NaN`. |
| **Right** | `how='right'` | `RIGHT OUTER JOIN` | Keep **all** rows from right; fill missing left values with `NaN`. |
| **Outer** | `how='outer'` | `FULL OUTER JOIN` | Keep all rows from both tables; fill non-matches with `NaN` (union). |
| **Cross** | `how='cross'` | `CROSS JOIN` | Cartesian product of both tables (every left row paired with every right row). |

### 6.2 Key Matching Options
- Identical column names: `on='user_id'` or `on=['tenant_id', 'user_id']`.
- Different column names: `left_on='buyer_id', right_on='customer_id'`.
- Merging on Index: `left_index=True, right_on='user_id'` or `left_index=True, right_index=True`.
- Avoiding collision with `suffixes=('_left', '_right')` (defaults to `('_x', '_y')`).

---

## 7. Defensive Merging: The `validate` Parameter

In real-world data engineering, silent cardinality bugs cause millions of dollars in duplicate billing or distorted analytics. 
For example, if you expect an order to have at most one customer record, but the customer table has duplicates, a standard `how='left'` join silently **duplicates order rows**!

To defend against this, always specify `validate`:

```python
pd.merge(orders, customers, on='customer_id', how='left', validate='many_to_one')
```

Valid options:
- `'1:1'` or `'one_to_one'`: Checks if merge keys are unique in both left and right datasets.
- `'1:m'` or `'one_to_many'`: Checks if merge keys are unique in left dataset.
- `'m:1'` or `'many_to_one'`: Checks if merge keys are unique in right dataset (most common when joining transaction log to dimension table).
- `'m:m'` or `'many_to_many'`: Default behavior (no check performed).

If the constraint is violated, Pandas immediately raises a `MergeError`:
```
MergeError: Merge keys are not unique in right dataset; not a many-to-one merge
```

---

## 8. Anti-Joins & Audit Trails with `indicator=True`

An **Anti-Join** returns rows from the left table that have **no match** in the right table (e.g. finding customers who have never placed an order).

Pandas does not have a native `how='anti'`, but provides `indicator=True`:

```python
# Performs merge and creates a special column named '_merge'
merged = pd.merge(customers, orders, on='customer_id', how='left', indicator=True)

# Inspect the merge indicator column values:
# '_merge' contains Categorical: ['left_only', 'right_only', 'both']

# Anti-Join: Customers without orders
customers_no_orders = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
```

Audit Full Discrepancies with Outer Join:
```python
reconciliation = pd.merge(ledger_bank, ledger_internal, on='txn_id', how='outer', indicator=True)
unmatched_bank = reconciliation[reconciliation['_merge'] == 'left_only']
unmatched_internal = reconciliation[reconciliation['_merge'] == 'right_only']
```

---

## 9. Combining DataFrames: `pd.concat` & `.join()`

### 9.1 `pd.concat()`
Used for stacking DataFrames vertically (row-wise, `axis=0`) or binding horizontally (column-wise, `axis=1`).

```python
# Vertical stacking (e.g. combining monthly CSV files)
full_year = pd.concat([q1_df, q2_df, q3_df, q4_df], axis=0, ignore_index=True)

# Vertical stacking with source identification keys (creates MultiIndex):
combined = pd.concat({'Q1': q1_df, 'Q2': q2_df}, names=['Quarter', 'RowID'])

# Horizontal stacking:
features = pd.concat([numeric_df, text_df], axis=1)
```

### 9.2 `DataFrame.join()`
A convenience method for joining primarily on the **index**.

```python
# Joins on index by default:
df1.join(df2, how='left', lsuffix='_df1', rsuffix='_df2')

# Join df2's index against df1's column 'key':
df1.join(df2.set_index('key'), on='key')
```

---

## 10. Why Avoid `.apply()`? The Cython Fast Path

A common anti-pattern among beginners is using `df.groupby(...).apply(...)` for every aggregation and transformation.

### Why `.apply()` is Slow:
1. **Python Interpreter Overhead**: Calls a pure Python function row-by-row or group-by-group, defeating NumPy/C vectorization.
2. **Object Boxing & Unboxing**: Converts internal C structures into Python `Series` or `DataFrame` objects on every iteration.
3. **Memory Allocations**: Generates intermediate Python objects that trigger frequent garbage collection cycles.

### Vectorized Alternatives:

| Instead of `.apply(...)` | Use Vectorized / Fast Alternative |
| :--- | :--- |
| `g.apply(lambda x: x['a'].sum() / x['b'].sum())` | Compute sums separately and divide: `g['a'].sum() / g['b'].sum()` |
| `g.apply(lambda x: (x - x.mean()) / x.std())` | `(df['x'] - g['x'].transform('mean')) / g['x'].transform('std')` |
| `g.apply(lambda x: x.sort_values('date').iloc[-1])` | `df.sort_values('date').groupby('id').last()` or `.tail(1)` |
| `g.apply(lambda x: x.nlargest(3, 'val'))` | `df.groupby('id')['val'].nlargest(3)` |
| `g.apply(lambda x: x['sales'] * 1.1)` | Simple column multiplication: `df['sales'] * 1.1` |

---

## 11. The 6 Critical Gotchas & Best Practices

1. **`dropna=True` in `groupby` Drops Missing Keys**:
   By default, `groupby(..., dropna=True)` silently ignores rows where the group key is `NaN`! If missing keys are important, always specify `dropna=False`.
2. **Duplicate Keys Cause Cartesian Explosions in Merges**:
   If left has 1,000 rows with key `X` and right has 1,000 rows with key `X`, an inner merge generates $1,000 \times 1,000 = 1,000,000$ rows! Always use `validate='m:1'` or `validate='1:m'` unless many-to-many is explicitly intended.
3. **MultiIndex Columns after `.agg()`**:
   Using `df.groupby('dept').agg({'salary': ['mean', 'max']})` creates hierarchical column levels: `('salary', 'mean')`. Prefer Named Aggregation (`salary_mean=('salary', 'mean')`) to keep columns flat.
4. **Index Alignment in Transformations**:
   When computing group operations with `.transform()`, the resulting series is aligned by index to the original DataFrame. Ensure your index has no duplicate labels if re-assigning.
5. **Categorical Dtypes in Crosstabs & Pivots**:
   If grouping columns are of `category` dtype, `pivot_table` and `crosstab` will include rows/columns for categories with 0 observations by default (`observed=False`). Pass `observed=True` to exclude empty categories.
6. **In-place Column Overwriting with Merge**:
   Never assume `merge` replaces matching columns. If both DataFrames share column names not in `on`, Pandas renames them with suffixes `_x` and `_y`.

---

## 12. 20-Problem Aggregation & Joins Drill Guide

In this module, you have two companion scripts:
- [`aggregation_joins_mastery.py`](file:///c:/Users/ASUS/Desktop/ds-internship-2026/W2/d2/aggregation_joins_mastery.py): Full interactive reference script demonstrating all concepts with live datasets and timing benchmarks.
- [`drill_20_problems.py`](file:///c:/Users/ASUS/Desktop/ds-internship-2026/W2/d2/drill_20_problems.py): 20 solved enterprise drills testing named aggregations, transformations, pivots, defensive joins, anti-joins, and concatenations.
- [`drill_blank_practice.py`](file:///c:/Users/ASUS/Desktop/ds-internship-2026/W2/d2/drill_blank_practice.py): Self-paced practice template with assertions.
