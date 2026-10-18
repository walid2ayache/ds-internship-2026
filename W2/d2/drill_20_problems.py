"""
Day 2: 20-Problem Aggregation & Joins Drill Solutions
=====================================================
This file contains 20 curated drills on Pandas Aggregations and Joins:
- GroupBy with Named Aggregation (modern syntax)
- Multi-column grouping and flat summaries
- Window and broadcast calculations with .transform()
- Missing value imputation using group statistics (.transform)
- Group-level filtering with .filter()
- Cross-tabulation & frequency tables with pd.crosstab
- Reshaping multi-dimensional metrics with pd.pivot_table (margins & fill_values)
- Relational joins: pd.merge (inner, left, outer, cross)
- Key alignment with left_on / right_on & index-based merging (df.join)
- Defensive merging using the 'validate' parameter (1:1, 1:m, m:1)
- Anti-joins and gap identification using indicator=True
- Vertical and horizontal concatenation with pd.concat
- End-to-end vectorized business pipeline avoiding .apply()

Run this file with:
    python drill_20_problems.py
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


# ==============================================================================
# SAMPLE DATASET GENERATOR
# ==============================================================================
def create_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Creates a fresh set of relational enterprise DataFrames for the drills."""
    
    customers = pd.DataFrame({
        'customer_id': ['C101', 'C102', 'C103', 'C104', 'C105', 'C106'],
        'customer_name': ['Acme Corp', 'Globex Inc', 'Soylent Corp', 'Initech', 'Umbrella Corp', 'Stark Industries'],
        'segment': ['Enterprise', 'SMB', 'Enterprise', 'SMB', 'Consumer', 'Enterprise'],
        'city': ['New York', 'Austin', 'New York', 'Chicago', 'San Francisco', 'New York']
    })

    products = pd.DataFrame({
        'product_id': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06'],
        'product_name': ['Laptop Pro', 'Wireless Mouse', 'Mechanical Keyboard', '4K Monitor', 'USB Hub', 'Desk Mat'],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Accessories'],
        'sub_category': ['Computers', 'Peripherals', 'Peripherals', 'Monitors', 'Peripherals', 'Office'],
        'unit_price': [1200.0, 35.0, 110.0, 450.0, 25.0, 20.0],
        'unit_cost': [800.0, 15.0, 60.0, 280.0, 10.0, 8.0]
    })

    orders = pd.DataFrame({
        'order_id': ['ORD-01', 'ORD-02', 'ORD-03', 'ORD-04', 'ORD-05', 'ORD-06', 'ORD-07', 'ORD-08', 'ORD-09', 'ORD-10'],
        'customer_id': ['C101', 'C102', 'C101', 'C103', 'C104', 'C101', 'C103', 'C102', 'C104', 'C101'],
        'product_id': ['P01', 'P02', 'P03', 'P01', 'P04', 'P04', 'P02', 'P05', 'P03', 'P06'],
        'sales_rep': ['Alice', 'Bob', 'Alice', 'Charlie', 'Diana', 'Alice', 'Charlie', 'Bob', 'Diana', 'Alice'],
        'quantity': [2, 10, 3, 1, 4, 1, 5, 8, 2, 6],
        'discount_rate': [0.10, 0.05, 0.00, 0.15, 0.10, 0.00, 0.00, 0.05, 0.10, 0.00],
        'region': ['East', 'South', 'East', 'East', 'Midwest', 'East', 'East', 'South', 'Midwest', 'East']
    })

    sales_reps = pd.DataFrame({
        'rep_id': ['SR01', 'SR02', 'SR03', 'SR04', 'SR05'],
        'rep_name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Evan'],  # Evan has no orders
        'base_salary': [75000, 68000, 72000, 70000, 65000],
        'region': ['East', 'South', 'East', 'Midwest', 'West']
    })

    returns = pd.DataFrame({
        'order_id': ['ORD-03', 'ORD-07', 'ORD-99'],  # ORD-99 has no matching order
        'return_reason': ['Defective Item', 'Customer Error', 'Damaged in Shipping']
    })

    return {
        'customers': customers,
        'products': products,
        'orders': orders,
        'sales_reps': sales_reps,
        'returns': returns
    }


# ==============================================================================
# DRILL 1: Named Aggregation on GroupBy
# ==============================================================================
def drill_01(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 1:
    Group `orders` by `region` and compute named aggregations:
    - 'total_quantity': sum of 'quantity'
    - 'avg_discount': mean of 'discount_rate' rounded to 4 decimals
    - 'order_count': count of 'order_id'
    Keep 'region' as a regular column (as_index=False).
    """
    result = orders.groupby('region', as_index=False).agg(
        total_quantity=('quantity', 'sum'),
        avg_discount=('discount_rate', 'mean'),
        order_count=('order_id', 'count')
    )
    result['avg_discount'] = result['avg_discount'].round(4)
    
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['region', 'total_quantity', 'avg_discount', 'order_count']
    assert len(result) == 3
    # Check East: orders 1, 3, 4, 6, 7, 10 -> qty: 2 + 3 + 1 + 1 + 5 + 6 = 18
    east_row = result[result['region'] == 'East'].iloc[0]
    assert east_row['total_quantity'] == 18
    assert east_row['order_count'] == 6
    return result


# ==============================================================================
# DRILL 2: Multi-Column GroupBy & Sorting
# ==============================================================================
def drill_02(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 2:
    Group `orders` by BOTH `region` and `sales_rep` (as_index=False).
    Compute:
    - 'total_qty': sum of 'quantity'
    - 'max_qty': max of 'quantity'
    Sort the resulting DataFrame by 'total_qty' in descending order.
    Reset the index so it is 0, 1, 2, ...
    """
    result = orders.groupby(['region', 'sales_rep'], as_index=False).agg(
        total_qty=('quantity', 'sum'),
        max_qty=('quantity', 'max')
    ).sort_values(by='total_qty', ascending=False, ignore_index=True)

    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]['total_qty'] >= result.iloc[1]['total_qty']
    assert 'region' in result.columns and 'sales_rep' in result.columns
    return result


# ==============================================================================
# DRILL 3: Vectorized Range Aggregation (Zero Apply)
# ==============================================================================
def drill_03(products: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 3:
    For each 'category' in `products`, compute:
    - 'min_price': minimum 'unit_price'
    - 'max_price': maximum 'unit_price'
    - 'price_spread': max_price - min_price
    Do this using named aggregation or vectorized column operations. Do NOT use .apply().
    """
    agg_df = products.groupby('category', as_index=False).agg(
        min_price=('unit_price', 'min'),
        max_price=('unit_price', 'max')
    )
    agg_df['price_spread'] = agg_df['max_price'] - agg_df['min_price']

    assert isinstance(agg_df, pd.DataFrame)
    assert 'price_spread' in agg_df.columns
    # Electronics: min 450, max 1200 -> spread 750
    elec = agg_df[agg_df['category'] == 'Electronics'].iloc[0]
    assert elec['price_spread'] == 750.0
    return agg_df


# ==============================================================================
# DRILL 4: Broadcast Group Share with .transform()
# ==============================================================================
def drill_04(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 4:
    Using .transform(), calculate each order's percentage share of total quantity
    within its respective 'region'.
    Add this as a new column: 'pct_of_region_qty' rounded to 2 decimal places.
    Example formula: (order_quantity / region_total_quantity) * 100.
    """
    df = orders.copy()
    region_totals = df.groupby('region')['quantity'].transform('sum')
    df['pct_of_region_qty'] = ((df['quantity'] / region_totals) * 100).round(2)

    assert len(df) == len(orders)
    # Check that sum of pct_of_region_qty for each region equals ~100.0
    for _, group in df.groupby('region'):
        assert np.isclose(group['pct_of_region_qty'].sum(), 100.0, atol=0.1)
    return df


# ==============================================================================
# DRILL 5: Group Standardization (Z-score) with .transform()
# ==============================================================================
def drill_05(products: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 5:
    Compute the z-score of 'unit_price' within each 'category' using .transform():
    z = (unit_price - category_mean) / category_std
    Add it as column 'price_zscore'.
    Avoid .apply().
    """
    df = products.copy()
    cat_mean = df.groupby('category')['unit_price'].transform('mean')
    cat_std = df.groupby('category')['unit_price'].transform('std')
    df['price_zscore'] = (df['unit_price'] - cat_mean) / cat_std

    assert len(df) == len(products)
    assert 'price_zscore' in df.columns
    # Check that for any category with >= 2 items, the mean z-score is 0
    assert np.isclose(df.groupby('category')['price_zscore'].mean().dropna().iloc[0], 0.0, atol=1e-6)
    return df


# ==============================================================================
# DRILL 6: Group-Level Missing Value Imputation with .transform()
# ==============================================================================
def drill_06() -> pd.DataFrame:
    """
    Problem 6:
    Given a DataFrame with missing 'sales' values, impute missing values using
    the median 'sales' of the respective 'department'.
    Do NOT use apply. Use .transform('median') and .fillna().
    """
    data = pd.DataFrame({
        'department': ['IT', 'IT', 'IT', 'HR', 'HR', 'HR', 'Sales', 'Sales'],
        'sales': [500.0, np.nan, 700.0, 200.0, np.nan, 220.0, np.nan, 900.0]
    })
    
    dept_median = data.groupby('department')['sales'].transform('median')
    data['sales_imputed'] = data['sales'].fillna(dept_median)

    # In IT, median of [500, 700] is 600.0
    assert data.loc[1, 'sales_imputed'] == 600.0
    # In HR, median of [200, 220] is 210.0
    assert data.loc[4, 'sales_imputed'] == 210.0
    # In Sales, median of [900] is 900.0
    assert data.loc[6, 'sales_imputed'] == 900.0
    assert data['sales_imputed'].isna().sum() == 0
    return data


# ==============================================================================
# DRILL 7: Group-Level Filtering with .filter()
# ==============================================================================
def drill_07(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 7:
    Filter `orders` to retain ONLY rows belonging to sales reps who have
    accumulated a TOTAL ordered quantity of at least 8 units across all their orders.
    """
    filtered = orders.groupby('sales_rep').filter(lambda g: g['quantity'].sum() >= 8)

    assert isinstance(filtered, pd.DataFrame)
    reps_kept = set(filtered['sales_rep'].unique())
    # Alice: 2 + 3 + 1 + 6 = 12 (>= 8) -> Kept
    # Bob: 10 + 8 = 18 (>= 8) -> Kept
    # Charlie: 1 + 5 = 6 (< 8) -> Dropped
    # Diana: 4 + 2 = 6 (< 8) -> Dropped
    assert reps_kept == {'Alice', 'Bob'}
    return filtered


# ==============================================================================
# DRILL 8: Pivot Table (Multi-Dimension Aggregation)
# ==============================================================================
def drill_08(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 8:
    Construct a pivot table from `orders` with:
    - index: 'region'
    - columns: 'sales_rep'
    - values: 'quantity'
    - aggfunc: 'sum'
    - fill_value: 0
    """
    pivot = orders.pivot_table(
        index='region',
        columns='sales_rep',
        values='quantity',
        aggfunc='sum',
        fill_value=0
    )
    
    assert isinstance(pivot, pd.DataFrame)
    assert pivot.loc['East', 'Alice'] == 12
    assert pivot.loc['South', 'Bob'] == 18
    assert pivot.loc['Midwest', 'Diana'] == 6
    assert (pivot.values >= 0).all()
    return pivot


# ==============================================================================
# DRILL 9: Pivot Table with Subtotals (Margins)
# ==============================================================================
def drill_09(products: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 9:
    Create a pivot table of `products` with:
    - index: 'category'
    - columns: 'sub_category'
    - values: 'unit_price'
    - aggfunc: 'mean'
    - fill_value: 0.0
    - margins: True
    - margins_name: 'All_Categories'
    """
    pivot = products.pivot_table(
        index='category',
        columns='sub_category',
        values='unit_price',
        aggfunc='mean',
        fill_value=0.0,
        margins=True,
        margins_name='All_Categories'
    )

    assert 'All_Categories' in pivot.index
    assert 'All_Categories' in pivot.columns
    return pivot


# ==============================================================================
# DRILL 10: Cross-Tabulation with Proportions (pd.crosstab)
# ==============================================================================
def drill_10(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 10:
    Generate a contingency table of customer 'city' (index) vs 'segment' (columns).
    Normalize across rows ('index') such that each row sums to 1.0 (100%).
    Multiply by 100 and round to 1 decimal place.
    """
    ct = pd.crosstab(
        index=customers['city'],
        columns=customers['segment'],
        normalize='index'
    ) * 100
    ct = ct.round(1)

    assert isinstance(ct, pd.DataFrame)
    # Check that each row sum is 100.0
    row_sums = ct.sum(axis=1)
    assert np.allclose(row_sums, 100.0, atol=0.2)
    return ct


# ==============================================================================
# DRILL 11: Inner Merge on Single Key
# ==============================================================================
def drill_11(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 11:
    Perform an INNER join between `orders` and `products` on 'product_id'.
    Calculate a new column:
    'gross_revenue' = quantity * unit_price
    Return the merged DataFrame containing only:
    ['order_id', 'product_name', 'category', 'quantity', 'unit_price', 'gross_revenue'].
    """
    merged = pd.merge(orders, products, on='product_id', how='inner')
    merged['gross_revenue'] = merged['quantity'] * merged['unit_price']
    
    cols = ['order_id', 'product_name', 'category', 'quantity', 'unit_price', 'gross_revenue']
    result = merged[cols]

    assert len(result) == len(orders)
    assert list(result.columns) == cols
    # ORD-01 is Laptop Pro (1200.0), qty = 2 -> 2400.0
    row1 = result[result['order_id'] == 'ORD-01'].iloc[0]
    assert row1['gross_revenue'] == 2400.0
    return result


# ==============================================================================
# DRILL 12: Left Merge & Null Handling
# ==============================================================================
def drill_12(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 12:
    Perform a LEFT join of `customers` with `orders` on 'customer_id'.
    Group by 'customer_name' (as_index=False) and compute:
    - 'total_units_ordered': sum of 'quantity' (fill missing/NaN with 0)
    Cast 'total_units_ordered' to int.
    """
    merged = pd.merge(customers, orders, on='customer_id', how='left')
    result = merged.groupby('customer_name', as_index=False).agg(
        total_units_ordered=('quantity', 'sum')
    )
    result['total_units_ordered'] = result['total_units_ordered'].fillna(0).astype(int)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(customers)
    # Stark Industries (C106) and Umbrella Corp (C105) have no orders -> 0
    stark = result[result['customer_name'] == 'Stark Industries'].iloc[0]
    assert stark['total_units_ordered'] == 0
    return result


# ==============================================================================
# DRILL 13: Anti-Join with indicator=True
# ==============================================================================
def drill_13(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 13:
    Identify customers who have NEVER placed any order (Anti-Join).
    Use pd.merge with how='left' and indicator=True.
    Filter for rows where _merge == 'left_only'.
    Return a DataFrame of only ['customer_id', 'customer_name', 'segment'] with reset index.
    """
    merged = pd.merge(customers, orders, on='customer_id', how='left', indicator=True)
    anti_joined = merged[merged['_merge'] == 'left_only'][['customer_id', 'customer_name', 'segment']].drop_duplicates().reset_index(drop=True)

    assert isinstance(anti_joined, pd.DataFrame)
    assert set(anti_joined['customer_id']) == {'C105', 'C106'}
    return anti_joined


# ==============================================================================
# DRILL 14: Merge with Mismatched Key Column Names
# ==============================================================================
def drill_14(sales_reps: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 14:
    Merge `sales_reps` (left) with `orders` (right) where:
    left key is 'rep_name' and right key is 'sales_rep'.
    Perform a LEFT join.
    Drop the duplicate key column 'sales_rep' from the result.
    """
    merged = pd.merge(
        sales_reps,
        orders,
        left_on='rep_name',
        right_on='sales_rep',
        how='left'
    ).drop(columns=['sales_rep'])

    assert 'sales_rep' not in merged.columns
    assert 'rep_name' in merged.columns
    # Evan (SR05) has no orders, so order_id should be NaN for Evan
    evan_rows = merged[merged['rep_name'] == 'Evan']
    assert len(evan_rows) == 1
    assert pd.isna(evan_rows.iloc[0]['order_id'])
    return merged


# ==============================================================================
# DRILL 15: Defensive Validation with validate Parameter
# ==============================================================================
def drill_15(orders: pd.DataFrame, products: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Problem 15:
    1. Merge `orders` with `products` on 'product_id' with validate='many_to_one'.
       Ensure it succeeds without error.
    2. Attempt to merge `orders` with `products` with validate='one_to_one'.
       Catch the pd.errors.MergeError and return caught_error = True.
    Return: (valid_merged_df, caught_error)
    """
    # 1. Valid many_to_one merge
    valid_df = pd.merge(orders, products, on='product_id', how='inner', validate='many_to_one')

    # 2. Invalid one_to_one merge (orders has multiple duplicate product_ids!)
    caught_error = False
    try:
        pd.merge(orders, products, on='product_id', how='inner', validate='one_to_one')
    except pd.errors.MergeError:
        caught_error = True

    assert isinstance(valid_df, pd.DataFrame)
    assert caught_error is True
    return valid_df, caught_error


# ==============================================================================
# DRILL 16: Index-Based Joining with DataFrame.join()
# ==============================================================================
def drill_16(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 16:
    Set 'customer_id' as index for `customers`.
    Compute order counts per customer from `orders` as a Series with index 'customer_id'
    and name 'order_count'.
    Join this Series into `customers` using customers.join(...).
    Fill missing order_count with 0 and convert to int.
    """
    cust_idx = customers.set_index('customer_id')
    counts = orders['customer_id'].value_counts().rename('order_count')
    
    joined = cust_idx.join(counts, how='left')
    joined['order_count'] = joined['order_count'].fillna(0).astype(int)

    assert isinstance(joined, pd.DataFrame)
    assert joined.index.name == 'customer_id'
    assert joined.loc['C101', 'order_count'] == 4
    assert joined.loc['C105', 'order_count'] == 0
    return joined


# ==============================================================================
# DRILL 17: Cross Join (Cartesian Product)
# ==============================================================================
def drill_17(sales_reps: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 17:
    Generate all possible pairings of sales reps and product categories.
    Get unique 'rep_name' from sales_reps and unique 'category' from products.
    Perform a cross join (how='cross') to get every combination.
    """
    reps = pd.DataFrame({'rep_name': sales_reps['rep_name'].unique()})
    cats = pd.DataFrame({'category': products['category'].unique()})
    
    cross = pd.merge(reps, cats, how='cross')

    assert isinstance(cross, pd.DataFrame)
    assert len(cross) == len(reps) * len(cats)
    assert list(cross.columns) == ['rep_name', 'category']
    return cross


# ==============================================================================
# DRILL 18: Vertical Concatenation (pd.concat axis=0)
# ==============================================================================
def drill_18() -> pd.DataFrame:
    """
    Problem 18:
    Given two seasonal batches of sales logs with slightly different columns:
    - batch1: columns ['order_id', 'amount', 'tax']
    - batch2: columns ['order_id', 'amount', 'discount']
    Concatenate them vertically (axis=0) with ignore_index=True.
    Fill all missing values with 0.0.
    """
    batch1 = pd.DataFrame({
        'order_id': ['B1-01', 'B1-02'],
        'amount': [100.0, 200.0],
        'tax': [8.0, 16.0]
    })
    batch2 = pd.DataFrame({
        'order_id': ['B2-01', 'B2-02'],
        'amount': [150.0, 300.0],
        'discount': [15.0, 30.0]
    })

    combined = pd.concat([batch1, batch2], axis=0, ignore_index=True).fillna(0.0)

    assert isinstance(combined, pd.DataFrame)
    assert len(combined) == 4
    assert set(combined.columns) == {'order_id', 'amount', 'tax', 'discount'}
    assert combined.loc[0, 'discount'] == 0.0
    assert combined.loc[2, 'tax'] == 0.0
    return combined


# ==============================================================================
# DRILL 19: Horizontal Concatenation with Index Alignment (axis=1)
# ==============================================================================
def drill_19() -> pd.DataFrame:
    """
    Problem 19:
    Given two DataFrames indexed by employee ID:
    - metrics: columns ['performance', 'attendance']
    - compensation: columns ['salary', 'bonus']
    Concatenate them horizontally (axis=1).
    Ensure the result contains all 4 features properly aligned by employee ID.
    """
    metrics = pd.DataFrame({
        'performance': [92.0, 85.0, 98.0],
        'attendance': [0.99, 0.95, 0.98]
    }, index=['E01', 'E02', 'E03'])

    compensation = pd.DataFrame({
        'salary': [85000, 72000, 115000],
        'bonus': [10000, 5000, 20000]
    }, index=['E01', 'E02', 'E03'])

    result = pd.concat([metrics, compensation], axis=1)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 4)
    assert list(result.columns) == ['performance', 'attendance', 'salary', 'bonus']
    assert result.loc['E03', 'bonus'] == 20000
    return result


# ==============================================================================
# DRILL 20: Complex Vectorized Analytics Pipeline (Zero Apply)
# ==============================================================================
def drill_20(orders: pd.DataFrame, products: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 20:
    Build an end-to-end profit and category performance analytics pipeline:
    1. Merge `orders` with `products` on 'product_id' (validate='m:1').
    2. Merge the result with `customers` on 'customer_id' (validate='m:1').
    3. Compute item revenue & net profit:
       - effective_price = unit_price * (1.0 - discount_rate)
       - revenue = quantity * effective_price
       - cost = quantity * unit_cost
       - profit = revenue - cost
    4. Group by ['category', 'segment'] (as_index=False) and compute:
       - 'total_revenue': sum of revenue
       - 'total_profit': sum of profit
       - 'avg_margin_pct': mean of (profit / revenue * 100)
    5. Using .transform(), calculate 'pct_of_category_revenue':
       each segment's share (%) of total category revenue.
    6. Sort by 'category' ascending, 'total_revenue' descending.
    Strictly NO .apply() allowed!
    """
    # 1 & 2. Defensive Multi-table Merge
    m1 = pd.merge(orders, products, on='product_id', how='inner', validate='m:1')
    df = pd.merge(m1, customers, on='customer_id', how='inner', validate='m:1')

    # 3. Vectorized Financial Metrics
    effective_price = df['unit_price'] * (1.0 - df['discount_rate'])
    df['revenue'] = df['quantity'] * effective_price
    df['cost'] = df['quantity'] * df['unit_cost']
    df['profit'] = df['revenue'] - df['cost']
    df['margin_pct'] = (df['profit'] / df['revenue']) * 100

    # 4. GroupBy Named Aggregation
    summary = df.groupby(['category', 'segment'], as_index=False).agg(
        total_revenue=('revenue', 'sum'),
        total_profit=('profit', 'sum'),
        avg_margin_pct=('margin_pct', 'mean')
    )

    # 5. Broadcast Category Share using .transform()
    cat_revenue_total = summary.groupby('category')['total_revenue'].transform('sum')
    summary['pct_of_category_revenue'] = (summary['total_revenue'] / cat_revenue_total) * 100

    # 6. Sorting
    summary = summary.sort_values(
        by=['category', 'total_revenue'],
        ascending=[True, False],
        ignore_index=True
    )

    assert isinstance(summary, pd.DataFrame)
    assert 'pct_of_category_revenue' in summary.columns
    # Check that category shares sum to 100% per category
    for _, group in summary.groupby('category'):
        assert np.isclose(group['pct_of_category_revenue'].sum(), 100.0, atol=1e-3)
    return summary


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================
def run_all_drills():
    print("=" * 65)
    print(" Running Day 2: 20-Problem Aggregation & Joins Drills")
    print("=" * 65)
    
    datasets = create_sample_datasets()
    orders = datasets['orders']
    products = datasets['products']
    customers = datasets['customers']
    sales_reps = datasets['sales_reps']

    drills = [
        ("Drill 01: Named Aggregation on GroupBy", lambda: drill_01(orders)),
        ("Drill 02: Multi-Column GroupBy & Sorting", lambda: drill_02(orders)),
        ("Drill 03: Vectorized Range Aggregation (Zero Apply)", lambda: drill_03(products)),
        ("Drill 04: Broadcast Group Share with .transform()", lambda: drill_04(orders)),
        ("Drill 05: Group Standardization (Z-score) with .transform()", lambda: drill_05(products)),
        ("Drill 06: Group-Level Missing Value Imputation", lambda: drill_06()),
        ("Drill 07: Group-Level Filtering with .filter()", lambda: drill_07(orders)),
        ("Drill 08: Pivot Table (Multi-Dimension Aggregation)", lambda: drill_08(orders)),
        ("Drill 09: Pivot Table with Subtotals (Margins)", lambda: drill_09(products)),
        ("Drill 10: Cross-Tabulation with Proportions (pd.crosstab)", lambda: drill_10(customers)),
        ("Drill 11: Inner Merge on Single Key", lambda: drill_11(orders, products)),
        ("Drill 12: Left Merge & Null Handling", lambda: drill_12(customers, orders)),
        ("Drill 13: Anti-Join with indicator=True", lambda: drill_13(customers, orders)),
        ("Drill 14: Merge with Mismatched Key Column Names", lambda: drill_14(sales_reps, orders)),
        ("Drill 15: Defensive Validation with validate Parameter", lambda: drill_15(orders, products)),
        ("Drill 16: Index-Based Joining with DataFrame.join()", lambda: drill_16(customers, orders)),
        ("Drill 17: Cross Join (Cartesian Product)", lambda: drill_17(sales_reps, products)),
        ("Drill 18: Vertical Concatenation (pd.concat axis=0)", lambda: drill_18()),
        ("Drill 19: Horizontal Concatenation with Index Alignment (axis=1)", lambda: drill_19()),
        ("Drill 20: Complex Vectorized Analytics Pipeline (Zero Apply)", lambda: drill_20(orders, products, customers)),
    ]

    passed = 0
    for name, drill_fn in drills:
        try:
            drill_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            raise e

    print("=" * 65)
    print(f" ALL {passed}/{len(drills)} DRILLS PASSED SUCCESSFULLY! ")
    print("=" * 65)


if __name__ == '__main__':
    run_all_drills()
