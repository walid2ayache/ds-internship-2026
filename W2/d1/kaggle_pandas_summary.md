# Kaggle Pandas Micro-Course: Complete Reference & Study Notes

This document provides a comprehensive, structured reference guide to the 6 lessons in the **Kaggle Pandas Micro-Course**. It includes key syntax, code patterns, common pitfalls, and reference solutions inspired by the classic **Wine Magazine Reviews** dataset exercises.

---

## Curriculum Roadmap

```
Kaggle Pandas Micro-Course
├── Lesson 1: Creating, Reading and Writing
├── Lesson 2: Indexing, Selecting & Assigning
├── Lesson 3: Summary Functions and Maps
├── Lesson 4: Grouping and Sorting
├── Lesson 5: Data Types and Missing Values
└── Lesson 6: Renaming and Combining
```

---

## Lesson 1: Creating, Reading and Writing

### 1.1 Creating DataFrames & Series Manually

#### `pd.DataFrame`
A DataFrame is a dictionary whose keys are column names and values are lists of column entries.

```python
import pandas as pd

# Standard dictionary of lists:
fruits = pd.DataFrame({
    'Apples': [30, 45],
    'Bananas': [21, 15]
}, index=['2017 Sales', '2018 Sales'])
```

#### `pd.Series`
A Series is a sequence of data values. Think of it as a single column of a DataFrame.

```python
# Series with custom index and name
quantities = pd.Series([4, 1, 2], index=['flour', 'milk', 'eggs'], name='Dinner Ingredients')
```

### 1.2 Reading & Writing Files

#### Reading CSVs
The most common mistake when reading CSVs is forgetting `index_col=0` when the CSV already has an unnamed index column.

```python
# Load dataset with first column as row index:
reviews = pd.read_csv('winemag-data-130k-v2.csv', index_col=0)

# Check shape immediately:
print(reviews.shape)  # e.g., (129971, 13)
```

#### Saving to CSV
Always pass `index=False` unless you intentionally want the numeric DataFrame index saved as an extra column.

```python
# Save without duplicating integer index:
reviews.to_csv('cleaned_reviews.csv', index=False)
```

---

## Lesson 2: Indexing, Selecting & Assigning

Pandas supports two primary indexers: `iloc` (position-based) and `loc` (label-based).

### 2.1 Native Python vs Pandas Access
```python
# Direct attribute or key access:
reviews.country
reviews['country']
reviews['country'][0]  # First element
```

### 2.2 `iloc` (Integer-Position Based)
- Follows standard Python 0-indexed slicing: `[start:stop)` (end is **excluded**).
- Works with integers only.

```python
# First row across all columns:
reviews.iloc[0]

# First 10 rows of column at index 0:
reviews.iloc[0:10, 0]

# Specific rows and columns by integer index:
reviews.iloc[[1, 2, 3, 5, 8], 0]

# Negative indexing works like Python lists:
reviews.iloc[-5:]  # Last 5 rows
```

### 2.3 `loc` (Label Based)
- Uses row index labels and column header strings.
- Slicing `start:stop` is **INCLUSIVE** of `stop`!

```python
# First row (label 0) for column 'country':
reviews.loc[0, 'country']

# Slicing with inclusive endpoint:
reviews.loc[0:9, 'country']  # Returns 10 items (0 through 9)

# Multiple columns:
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
```

### 2.4 Boolean Indexing with `loc`
Combine multiple criteria using bitwise operators (`&`, `|`, `~`). Each condition **must** be enclosed in parentheses.

```python
# Italian wines with score at least 90:
italian_wines = reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]

# Wines from Italy OR France with score >= 95:
top_wines = reviews.loc[
    reviews.country.isin(['Italy', 'France']) & (reviews.points >= 95)
]

# Wines with non-null price:
priced_wines = reviews.loc[reviews.price.notna()]
```

### 2.5 Assigning Values
```python
# Constant assignment:
reviews['critic'] = 'everyone'

# Iterable assignment:
reviews['index_backwards'] = range(len(reviews), 0, -1)
```

---

## Lesson 3: Summary Functions and Maps

### 3.1 Common Summary Functions
```python
# Statistical summary:
reviews.points.describe()

# Scalar summaries:
reviews.points.mean()
reviews.points.median()

# Unique inspection:
reviews.taster_name.unique()
reviews.taster_name.nunique()

# Frequency of occurrences:
reviews.taster_name.value_counts()
```

### 3.2 Maps & Vectorization

#### `.map()` (Series method)
Takes each single value in a Series and transforms it:

```python
# Centering points around the mean:
mean_points = reviews.points.mean()
centered_points = reviews.points.map(lambda p: p - mean_points)
```

#### Direct Vectorized Operations (Preferred & Much Faster!)
In almost all simple arithmetic cases, native vectorization is drastically faster than `.map()`:

```python
# Vectorized arithmetic (orders of magnitude faster than lambda):
centered_points = reviews.points - reviews.points.mean()

# Combining string columns vectorized:
reviews.country + " - " + reviews.variety
```

#### `.apply()` (DataFrame method)
Transforms entire rows or columns:

```python
def remap_price(row):
    row.price = row.price * 1.08  # Apply 8% tax
    return row

# axis='columns' means apply function across each row:
taxed_reviews = reviews.apply(remap_price, axis='columns')
```

### 3.3 Kaggle Exercise Highlights
- **Best Bargain Wine**:
  ```python
  bargain_idx = (reviews.points / reviews.price).idxmax()
  bargain_wine = reviews.loc[bargain_idx, 'title']
  ```
- **Word Counting in Text**:
  ```python
  n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
  n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
  descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
  ```

---

## Lesson 4: Grouping and Sorting

### 4.1 `.groupby()` Operations
`groupby` replicates SQL `GROUP BY` logic:

```python
# Who wrote the most reviews?
reviews_by_taster = reviews.groupby('taster_twitter_handle').size()

# Cheapest wine per point value:
cheapest_per_point = reviews.groupby('points').price.min()

# Best wine rating per variety:
best_rating_per_variety = reviews.groupby('variety').points.max()
```

### 4.2 Multi-Aggregations with `.agg()`
Pass a list of summary functions to `.agg()`:

```python
# Min and Max price per variety:
price_extremes = reviews.groupby('variety').price.agg([min, max])
```

### 4.3 Multi-Index & `reset_index()`
Grouping by multiple columns generates a `MultiIndex`:

```python
multi_group = reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])

# Convert MultiIndex back to standard flat columns:
flat_group = multi_group.reset_index()
```

### 4.4 Sorting: `.sort_values()` & `.sort_index()`
```python
# Sort by single column:
reviews.sort_values(by='price', ascending=False)

# Multi-column sort with mixed directions:
reviews.sort_values(by=['country', 'points'], ascending=[True, False])

# Sort by index:
reviews.sort_index()

# Control NaN placement:
reviews.sort_values(by='price', na_position='first')
```

---

## Lesson 5: Data Types and Missing Values

### 5.1 Checking Dtypes & Converting
```python
# Check type of a column:
reviews.price.dtype

# Check all column types:
reviews.dtypes

# Convert column dtype:
reviews.points = reviews.points.astype('float64')
```

### 5.2 Identifying Missing Values
```python
# Boolean mask of missing entries:
reviews[reviews.country.isna()]

# Count missing entries per column:
reviews.isna().sum()
```

### 5.3 Handling & Imputing Missing Values
```python
# Impute NaNs with a replacement string or value:
reviews.region_1.fillna("Unknown")

# Replace specific values:
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
```

---

## Lesson 6: Renaming and Combining

### 6.1 Renaming Columns & Index
```python
# Renaming columns:
reviews.rename(columns={'points': 'score'})

# Renaming row index labels:
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# Renaming axis titles:
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```

### 6.2 Combining Data: Concat, Join, and Merge

| Method | Best Used For | Notes |
| :--- | :--- | :--- |
| `pd.concat([df1, df2])` | Stacking tables vertically (`axis=0`) or horizontally (`axis=1`) | Columns/indices align automatically |
| `df1.join(df2, lsuffix, rsuffix)` | Fast combination on matching indices | Requires unique suffixes if column names collide |
| `df1.merge(df2, on='key', how='inner')` | SQL-style join on common key columns | Supports `'inner'`, `'left'`, `'right'`, `'outer'` |

#### Examples:
```python
# Vertical concatenation of tables with identical columns:
canadian_youtube = pd.read_csv("CAvideos.csv")
british_youtube = pd.read_csv("GBvideos.csv")
combined_videos = pd.concat([canadian_youtube, british_youtube])

# Index-based join:
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
joined = left.join(right, lsuffix='_CAN', rsuffix='_UK')

# SQL-style merge on shared ID column:
merged = df_orders.merge(df_customers, on='customer_id', how='left')
```

---

## Summary Cheat Sheet

| Task | Preferred Pandas Method |
| :--- | :--- |
| Position lookup | `df.iloc[row_idx, col_idx]` |
| Label lookup | `df.loc[row_label, col_label]` |
| Fast scalar lookup | `df.at[label]` / `df.iat[pos]` |
| Compound filtering | `df[(df['a'] > 1) & (df['b'] == 2)]` |
| Text search | `df['col'].str.contains('text', case=False, na=False)` |
| Clean string query | `df.query('a > 1 and b == @var')` |
| Value frequency | `df['col'].value_counts(normalize=True)` |
| Grouped aggregations | `df.groupby('category')['metric'].agg(['mean', 'std'])` |
| Fill missing | `df['col'].fillna(df['col'].median())` |
| Replace values | `df['col'].replace({'old': 'new'})` |
| Safe slice mutation | `subset = df[mask].copy()` then `subset.loc[:, 'col'] = val` |
