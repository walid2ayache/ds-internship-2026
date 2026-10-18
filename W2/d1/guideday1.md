# Day 1: Pandas Core — Comprehensive Guide & Reference (`guideday1`)

Welcome to **Day 1** of Week 2! Today is dedicated to mastering the bedrock of data manipulation in Python: **Pandas Core**.

---

## Table of Contents
1. [Core Data Structures: Series vs DataFrame](#1-core-data-structures-series-vs-dataframe)
2. [Data Inspection & Metadata](#2-data-inspection--metadata)
3. [Indexing & Selection: `loc` vs `iloc`](#3-indexing--selection-loc-vs-iloc)
4. [Filtering Data: Boolean Indexing & `.query()`](#4-filtering-data-boolean-indexing--query)
5. [Sorting: `.sort_values()` & `.sort_index()`](#5-sorting-sort_values--sort_index)
6. [Data Types & Type Conversion (`dtypes`, `astype`)](#6-data-types--type-conversion-dtypes-astype)
7. [Kaggle Pandas Micro-Course Breakdown](#7-kaggle-pandas-micro-course-breakdown)
8. [The 5 Critical Gotchas & Best Practices](#8-the-5-critical-gotchas--best-practices)
9. [20-Problem Selection & Filtering Drill Guide](#9-20-problem-selection--filtering-drill-guide)

---

## 1. Core Data Structures: Series vs DataFrame

Pandas is built on top of NumPy, but adds labeled axes (indices and column headers), heterogeneous data support, and alignment operations.

### 1.1 `pd.Series` (1-Dimensional)
A Series is a 1D labeled array capable of holding any data type.
- **Index**: Labels assigned to each row (defaults to `0, 1, 2, ...`).
- **Values**: Underlying array of data (`.values` or `.to_numpy()`).
- **Name**: An optional identifier for the Series (useful when converting to DataFrame).

```python
import pandas as pd
import numpy as np

s = pd.Series([10, 25, 40, np.nan], index=['a', 'b', 'c', 'd'], name='points')
print(s.index)   # Index(['a', 'b', 'c', 'd'], dtype='object')
print(s.values)  # array([10., 25., 40., nan])
print(s.dtype)   # float64
```

### 1.2 `pd.DataFrame` (2-Dimensional)
A 2D tabular data structure with labeled axes (`index` for rows, `columns` for columns). Each column is a `pd.Series`.

```python
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 22],
    'salary': [70000, 85000, 92000, 58000],
    'department': ['HR', 'IT', 'IT', 'Marketing']
}
df = pd.DataFrame(data, index=['e1', 'e2', 'e3', 'e4'])
```

Key structural attributes:
- `df.shape` -> `(rows, cols)` tuple
- `df.columns` -> column labels
- `df.index` -> row index labels
- `df.ndim` -> always `2` for DataFrame

---

## 2. Data Inspection & Metadata

Before doing any analysis or transformation, inspect the data's shape, types, and summary statistics.

| Method | What It Tells You | Typical Usage |
| :--- | :--- | :--- |
| `df.info()` | Memory usage, non-null counts, dtypes | Detect missing values & unexpected object types |
| `df.describe()` | Count, mean, std, min, quartiles, max | Distribution of numeric columns |
| `df.describe(include='all')` | Includes top, freq, unique for categorical columns | Full overview across all column types |
| `df['col'].value_counts()` | Frequency of each unique value in a column | Categorical distribution |
| `df['col'].value_counts(normalize=True)` | Relative frequency (proportions / percentages) | Share / percentage breakdown |
| `df['col'].nunique()` | Count of distinct unique values | Cardinality check |
| `df['col'].unique()` | Array of distinct unique values | Find distinct categories |

### Pro-Tip: Value Counts Options
```python
# Include missing values in count:
df['department'].value_counts(dropna=False)

# Bin continuous values into intervals automatically:
df['salary'].value_counts(bins=3)
```

---

## 3. Indexing & Selection: `loc` vs `iloc`

This is the most fundamental concept in Pandas.

| Feature | `loc` (Label-based) | `iloc` (Integer position-based) |
| :--- | :--- | :--- |
| **Lookup Mode** | By **Index label** & **Column name** | By **0-based integer position** |
| **Slicing End** | **INCLUSIVE** of end label (`'a':'c'` includes `'c'`) | **EXCLUSIVE** of end stop (`0:3` gets `0, 1, 2`) |
| **Row Indexer** | `df.loc['e1']` or `df.loc[['e1', 'e3']]` | `df.iloc[0]` or `df.iloc[[0, 2]]` |
| **Row & Col** | `df.loc['e1':'e3', ['name', 'salary']]` | `df.iloc[0:3, [0, 2]]` |
| **Boolean Mask**| Directly supported: `df.loc[df['age'] > 25]` | Supported if passed as numpy boolean array |

### Visual Comparison Example:
```python
# Row indices: 'e1', 'e2', 'e3', 'e4'
# Col names: 'name', 'age', 'salary', 'department'

# loc includes BOTH start and stop labels:
df.loc['e1':'e3', 'name':'salary']
# Returns rows e1, e2, e3 and columns name, age, salary.

# iloc follows standard Python slice rules [start:stop):
df.iloc[0:2, 0:2]
# Returns row 0 and 1, col 0 and 1.
```

### Single item lookup:
- `df.at['e1', 'salary']` — Faster scalar label lookup
- `df.iat[0, 2]` — Faster scalar position lookup

---

## 4. Filtering Data: Boolean Indexing & `.query()`

### 4.1 Boolean Indexing Operators
In Pandas, Python's logical keywords `and`, `or`, `not` **do not work** for element-wise comparisons on Series. You **must** use bitwise operators and wrap each condition in parentheses:
- `&` : Element-wise AND
- `|` : Element-wise OR
- `~` : Element-wise NOT (Invert)

```python
# Correct (Parentheses are mandatory due to operator precedence):
filtered = df[(df['age'] > 24) & (df['department'] == 'IT')]

# Multiple choices: isin
filtered = df[df['department'].isin(['IT', 'HR'])]

# Numeric ranges: between (inclusive by default)
filtered = df[df['salary'].between(60000, 90000)]

# String filtering:
filtered = df[df['name'].str.startswith('A')]
filtered = df[df['department'].str.contains('market', case=False, na=False)]
```

### 4.2 The `.query()` Method
`.query()` evaluates a boolean expression as a string. It can make code much cleaner and easier to read.
- Refer to columns directly without `df['col']`.
- If a column name has spaces or special characters, wrap it in backticks: `` `Order Date` ``.
- Refer to variables from the outer Python scope using the `@` prefix.

```python
target_dept = 'IT'
min_salary = 80000

# Using @ to inject local variables:
result = df.query("department == @target_dept and salary >= @min_salary")

# Combining conditions:
result = df.query("age < 30 or salary > 90000")
```

---

## 5. Sorting: `.sort_values()` & `.sort_index()`

### 5.1 `.sort_values()`
Sort by one or more columns:
```python
# Sort by single column ascending (default):
df.sort_values(by='salary')

# Sort descending:
df.sort_values(by='salary', ascending=False)

# Multi-column sort: department ascending, salary descending
df.sort_values(by=['department', 'salary'], ascending=[True, False])

# Control where NaNs go: 'first' or 'last'
df.sort_values(by='salary', na_position='first')
```

### 5.2 `.sort_index()`
Sort by row labels or column headers:
```python
df.sort_index(ascending=True)         # Rows by index label
df.sort_index(axis=1, ascending=True) # Columns alphabetically
```

---

## 6. Data Types & Type Conversion (`dtypes`, `astype`)

Pandas assigns a data type (`dtype`) to each Series.

| Common Dtype | Python / NumPy Equivalent | Notes |
| :--- | :--- | :--- |
| `int64` / `Int64` | Integers | `Int64` (capitalized) is Pandas' nullable integer |
| `float64` | Floats | Supports `np.nan` |
| `object` | Strings or arbitrary Python objects | Can be memory inefficient |
| `string` / `string[pyarrow]` | Dedicated string type | Safer string operations |
| `category` | Categorical / Factors | Massive memory savings for low-cardinality strings |
| `datetime64[ns]` | Timestamp | Enables `.dt` accessors (`.dt.year`, `.dt.day_name()`) |
| `bool` / `boolean` | True / False | `boolean` supports `<NA>` |

### Type Conversion with `.astype()`
```python
# Convert single column:
df['age'] = df['age'].astype('int32')

# Convert multiple columns at once:
df = df.astype({
    'age': 'int32',
    'department': 'category'
})

# Convert strings to datetime:
df['order_date'] = pd.to_datetime(df['order_date'])

# Convert dirty strings to numeric with error coercion (creates NaN for invalid text):
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
```

---

## 7. Kaggle Pandas Micro-Course Breakdown

The official Kaggle Pandas micro-course consists of 6 core lessons:

### Lesson 1: Creating, Reading and Writing
- `pd.DataFrame({'A': [1, 2], 'B': [3, 4]})`
- `pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='Scores')`
- `pd.read_csv('file.csv', index_col=0)`
- `df.to_csv('output.csv', index=False)`

### Lesson 2: Indexing, Selecting & Assigning
- Native access: `df.col` or `df['col']`
- `df.iloc[0:5, 0]` vs `df.loc[0:5, 'country']`
- Setting values: `df['critic'] = 'everyone'` or `df.loc[condition, 'status'] = 'Active'`

### Lesson 3: Summary Functions and Maps
- Functions: `df.points.describe()`, `df.points.mean()`, `df.taster_name.unique()`
- Maps:
  - `df.points.map(lambda p: p - mean_points)` (for Series)
  - `df.apply(custom_row_func, axis='columns')` (for DataFrame row-wise)

### Lesson 4: Grouping and Sorting
- `df.groupby('points').price.min()`
- Multi-aggregations: `df.groupby('country').price.agg([len, min, max])`
- MultiIndex & `reset_index()`
- Sorting: `df.sort_values(by=['country', 'points'], ascending=[True, False])`

### Lesson 5: Data Types and Missing Values
- Checking types: `df.price.dtype`, `df.dtypes`
- Converting: `df.points.astype('float64')`
- Finding missing: `df[pd.isnull(df.country)]` or `df[df.country.isna()]`
- Imputing: `df.region_2.fillna('Unknown')`
- Replacing: `df.taster_twitter_handle.replace("@kerinokeefe", "@kerino")`

### Lesson 6: Renaming and Combining
- Renaming axes: `df.rename(columns={'points': 'score'})`
- Renaming index: `df.rename(index={0: 'firstEntry', 1: 'secondEntry'})`
- Renaming axis names: `df.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')`
- Combining:
  - `pd.concat([df1, df2])` (vertical or horizontal stack)
  - `df1.join(df2, lsuffix='_CAN', rsuffix='_UK')` (index-based)
  - `df1.merge(df2, on='key', how='inner'|'left'|'right'|'outer')`

---

## 8. The 5 Critical Gotchas & Best Practices

1. **`SettingWithCopyWarning`**:
   - Avoid chained indexing: `df[df['age'] > 30]['salary'] = 100000` (DO NOT DO THIS).
   - Use `.loc` instead: `df.loc[df['age'] > 30, 'salary'] = 100000`.
   - If filtering a subset to modify independently, explicitly call `.copy()`:
     `subset = df[df['age'] > 30].copy()`.

2. **Bitwise Operator Precedence**:
   - `df['age'] > 20 & df['age'] < 40` evaluates as `df['age'] > (20 & df['age']) < 40`!
   - Always wrap in parentheses: `(df['age'] > 20) & (df['age'] < 40)`.

3. **`loc` vs `iloc` Endpoint**:
   - `df.loc[0:2]` includes row with index label `2`.
   - `df.iloc[0:2]` only includes rows at positions `0` and `1`.

4. **Missing Values with Comparison**:
   - `np.nan == np.nan` evaluates to `False`.
   - Always use `.isna()` / `.notna()` to check for missing data.

5. **Memory Optimization**:
   - Convert repeating string columns with low cardinality to `'category'`.
   - Downcast large integers (`int64` to `int32` or `int16`) when working with large datasets.

---

## 9. 20-Problem Selection & Filtering Drill Guide

Inside the `d1/` folder, you will find:
1. `pandas_core_mastery.py` — Complete runnable demonstration script covering all topics.
2. `drill_20_problems.py` — The complete 20 problems with full solutions, detailed commentary, and assertions.
3. `drill_blank_practice.py` — A blank template where you can solve all 20 drills from scratch yourself!
4. `kaggle_pandas_summary.md` — Complete reference and notes for the Kaggle Pandas course exercises.

### Summary of the 20 Drills

| Drill # | Topic / Concept | Task Description |
| :--- | :--- | :--- |
| **01** | Series Creation & Summary | Create a Series from a dictionary, compute `.sum()`, find `.idxmax()` |
| **02** | Column Selection | Select multiple columns (`[['col1', 'col2']]`) preserving 2D DataFrame shape |
| **03** | Label Slicing (`.loc`) | Slice rows and columns by label; verify inclusive endpoint behavior |
| **04** | Position Slicing (`.iloc`) | Slice rows and columns by integer index; verify exclusive stop behavior |
| **05** | Fast Scalar Lookups | Extract single cell values using `.at` (label) and `.iat` (position) |
| **06** | Boolean Indexing | Single condition filtering (`df['salary'] > 85000`) |
| **07** | Compound Filtering (`&`) | AND condition with parentheses (`(dept == 'Eng') & (is_remote == True)`) |
| **08** | Compound Filtering (`|`, `~`) | OR condition with NOT inversion (`isin(['Mkt', 'Fin']) & ~(city == 'Chicago')`) |
| **09** | Value Membership (`.isin`) | Filter rows matching a list of categories (`df['city'].isin(...)`) |
| **10** | Range Filtering (`.between`) | Filter numerical range (`df['score'].between(85.0, 95.0)`) |
| **11** | Missing Data (`.isna`/`.notna`) | Identify rows with missing and non-missing bonus values |
| **12** | String Search (`.str.contains`) | Case-insensitive substring search (`df['name'].str.contains('an', case=False)`) |
| **13** | String Prefix (`.str.startswith`) | Filter categories with specific prefixes (`df['department'].str.startswith('Eng')`) |
| **14** | Querying with `.query()` | String-based querying (`df.query("is_remote == True and years_exp >= 5")`) |
| **15** | Environment Vars in `.query()` | Inject local Python variables via `@var` syntax in query strings |
| **16** | Single-Column Sorting | Sort by column descending (`.sort_values(by='salary', ascending=False)`) |
| **17** | Multi-Column Sorting | Multi-level sort with mixed orders (`ascending=[True, False]`) |
| **18** | Index Sorting | Sort DataFrame by its index labels (`.sort_index(ascending=False)`) |
| **19** | Frequency Proportions | Proportional distributions with `.value_counts(normalize=True)` |
| **20** | Safe Mutation | Avoid `SettingWithCopyWarning` via explicit `.copy()` and `.loc[:, col]` |

### Running the Solutions
To run all 20 solved drills with automated test assertions:
```bash
python drill_20_problems.py
```

### Practicing from Scratch
To test your knowledge, open `drill_blank_practice.py` and implement each function, then run:
```bash
python drill_blank_practice.py
```
Each solved drill will transition from `[PENDING]` to `[PASS]`.

