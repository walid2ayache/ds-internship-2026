"""
Day 2: Aggregation & Joins Mastery
==================================
This script provides complete, runnable demonstrations of all core concepts for Day 2:
- GroupBy with Named Aggregation (modern syntax)
- Group-level transformations (.transform) for broadcast metrics & imputation
- Group-level filtering (.filter)
- Reshaping: pd.pivot_table and pd.crosstab with normalization & margins
- Relational Joins: pd.merge (inner, left, right, outer, cross)
- Key validation in merges (validate='1:m', 'm:1', '1:1')
- Anti-joins and merge reconciliation with indicator=True
- pd.concat (row-wise axis=0, column-wise axis=1) and df.join
- Performance benchmark: Vectorized .transform vs .apply()

Run this script with:
    python aggregation_joins_mastery.py
"""

import time
import numpy as np
import pandas as pd


def header(title: str):
    print("\n" + "=" * 65)
    print(f" {title} ")
    print("=" * 65)


# ==============================================================================
# 1. GROUPBY & NAMED AGGREGATION
# ==============================================================================
header("1. GroupBy & Named Aggregation (.agg)")

sales_data = pd.DataFrame({
    'department': ['Tech', 'Tech', 'Tech', 'Apparel', 'Apparel', 'Home', 'Home', 'Home'],
    'region': ['East', 'West', 'East', 'East', 'West', 'East', 'West', 'West'],
    'rep_name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Evan', 'Fiona', 'George', 'Hannah'],
    'units_sold': [12, 19, 14, 30, 25, 8, 15, 22],
    'revenue': [12000, 19000, 14500, 3000, 2500, 1600, 3000, 4400],
    'customer_rating': [4.8, 4.2, 4.9, 3.9, 4.5, 4.1, 4.7, 4.0]
})

print("Raw Sales DataFrame:")
print(sales_data)

# Named Aggregation: specify clean new column names directly as kwargs!
dept_summary = sales_data.groupby('department', as_index=False).agg(
    total_revenue=('revenue', 'sum'),
    avg_units=('units_sold', 'mean'),
    max_rating=('customer_rating', 'max'),
    sales_rep_count=('rep_name', 'count')
)

print("\n--- Summary per Department (Named Aggregation) ---")
print(dept_summary)

# Multi-level GroupBy with as_index=True
dept_reg_summary = sales_data.groupby(['department', 'region']).agg(
    rev_sum=('revenue', 'sum'),
    rev_mean=('revenue', 'mean')
)
print("\n--- Multi-Column GroupBy (Department, Region) ---")
print(dept_reg_summary)


# ==============================================================================
# 2. GROUP TRANSFORMS (.transform)
# ==============================================================================
header("2. Group Transformations (.transform) — Broadcast to Original Shape")

# A. Share of Group Total (Percentage of department revenue per rep)
sales_data['dept_total_rev'] = sales_data.groupby('department')['revenue'].transform('sum')
sales_data['pct_of_dept_rev'] = (sales_data['revenue'] / sales_data['dept_total_rev']) * 100

# B. Group Standardization (Z-score within department)
dept_mean = sales_data.groupby('department')['revenue'].transform('mean')
dept_std = sales_data.groupby('department')['revenue'].transform('std')
sales_data['rev_zscore_dept'] = (sales_data['revenue'] - dept_mean) / dept_std

print("DataFrame after .transform operations:")
print(sales_data[['department', 'rep_name', 'revenue', 'pct_of_dept_rev', 'rev_zscore_dept']])

# C. Group Imputation with Median
messy_salaries = pd.DataFrame({
    'team': ['Dev', 'Dev', 'Dev', 'Sales', 'Sales', 'Sales'],
    'salary': [95000, np.nan, 105000, 60000, 75000, np.nan]
})
print("\nBefore Imputation:")
print(messy_salaries)

messy_salaries['salary_imputed'] = messy_salaries['salary'].fillna(
    messy_salaries.groupby('team')['salary'].transform('median')
)
print("\nAfter Group-Median Imputation:")
print(messy_salaries)


# ==============================================================================
# 3. GROUP FILTERING (.filter)
# ==============================================================================
header("3. Group-Level Filtering (.filter)")

# Keep only departments with total revenue > 10,000
high_rev_depts = sales_data.groupby('department').filter(lambda g: g['revenue'].sum() > 10000)
print(f"Original row count: {len(sales_data)}, Filtered row count: {len(high_rev_depts)}")
print("Departments kept:", high_rev_depts['department'].unique())


# ==============================================================================
# 4. RESHAPING: PIVOT TABLE & CROSSTAB
# ==============================================================================
header("4. Reshaping: pivot_table & crosstab")

# Pivot table: Department x Region revenue sum with subtotal margins
pivot_rev = sales_data.pivot_table(
    index='department',
    columns='region',
    values='revenue',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='Grand Total'
)
print("--- Pivot Table (Department x Region Total Revenue) ---")
print(pivot_rev)

# Crosstab: Headcount distribution & row percentages
ct_counts = pd.crosstab(
    index=sales_data['department'],
    columns=sales_data['region'],
    margins=True
)
print("\n--- Crosstab (Frequencies) ---")
print(ct_counts)

ct_pct = pd.crosstab(
    index=sales_data['department'],
    columns=sales_data['region'],
    normalize='index'
) * 100
print("\n--- Crosstab (Row Proportions %) ---")
print(ct_pct.round(1))


# ==============================================================================
# 5. RELATIONAL JOINS WITH pd.merge()
# ==============================================================================
header("5. Relational Joins with pd.merge()")

customers = pd.DataFrame({
    'cust_id': ['C01', 'C02', 'C03', 'C04'],
    'name': ['Alpha Corp', 'Beta LLC', 'Gamma Inc', 'Delta Co'],
    'country': ['USA', 'UK', 'USA', 'Germany']
})

orders = pd.DataFrame({
    'order_id': ['O1001', 'O1002', 'O1003', 'O1004', 'O1005'],
    'cust_id': ['C01', 'C01', 'C02', 'C04', 'C99'],  # C99 has no customer record
    'amount': [500.0, 1200.0, 350.0, 2200.0, 150.0]
})

print("Customers table:")
print(customers)
print("\nOrders table:")
print(orders)

# Inner Join: Only matching IDs
inner_m = pd.merge(orders, customers, on='cust_id', how='inner')
print(f"\n--- Inner Join (Matches only: {len(inner_m)} rows) ---")
print(inner_m[['order_id', 'cust_id', 'name', 'amount']])

# Left Join: All orders preserved
left_m = pd.merge(orders, customers, on='cust_id', how='left')
print(f"\n--- Left Join (Preserves all orders: {len(left_m)} rows) ---")
print(left_m[['order_id', 'cust_id', 'name', 'amount']])


# ==============================================================================
# 6. DEFENSIVE MERGING (validate) & ANTI-JOINS (indicator)
# ==============================================================================
header("6. Defensive Merging (validate) & Anti-Joins (indicator)")

# Defensive validation: 'many_to_one' ensures each customer appears once in customers
validated_merge = pd.merge(
    orders, customers,
    on='cust_id',
    how='left',
    validate='many_to_one'
)
print("Merge passed validate='many_to_one' successfully!")

# Anti-Join using indicator=True: Find customers who NEVER placed an order
cust_orders_audit = pd.merge(
    customers, orders,
    on='cust_id',
    how='left',
    indicator=True
)
print("\nMerge with indicator=True:")
print(cust_orders_audit[['cust_id', 'name', 'order_id', '_merge']])

unmatched_customers = cust_orders_audit[cust_orders_audit['_merge'] == 'left_only']
print("\nAnti-Join Result: Customers with ZERO orders:")
print(unmatched_customers[['cust_id', 'name']])


# ==============================================================================
# 7. CONCATENATION & JOIN
# ==============================================================================
header("7. Concatenation (pd.concat) & df.join")

# Row-wise concatenation (Stacking monthly data)
jan_sales = pd.DataFrame({'product': ['Widget A', 'Widget B'], 'sales': [100, 200]})
feb_sales = pd.DataFrame({'product': ['Widget A', 'Widget C'], 'sales': [150, 300]})

q1_stacked = pd.concat([jan_sales, feb_sales], axis=0, ignore_index=True)
print("--- Stacked DataFrames (axis=0) ---")
print(q1_stacked)

# Column-wise concatenation (axis=1)
user_profiles = pd.DataFrame({'age': [25, 34, 42]}, index=['U1', 'U2', 'U3'])
user_scores = pd.DataFrame({'credit_score': [720, 680, 810]}, index=['U1', 'U2', 'U3'])
merged_users = pd.concat([user_profiles, user_scores], axis=1)
print("\n--- Column-wise Concatenation (axis=1) ---")
print(merged_users)


# ==============================================================================
# 8. PERFORMANCE BENCHMARK: Vectorized .transform vs .apply()
# ==============================================================================
header("8. Performance Benchmark: Vectorized .transform vs .apply()")

N = 200_000
benchmark_df = pd.DataFrame({
    'category_id': np.random.randint(1, 100, size=N),
    'val': np.random.randn(N)
})

# Method 1: apply with custom lambda (SLOW)
t0 = time.perf_counter()
res_apply = benchmark_df.groupby('category_id')['val'].apply(lambda x: x - x.mean()).values
t_apply = time.perf_counter() - t0

# Method 2: transform with fast vectorized Cython engine (FAST)
t0 = time.perf_counter()
res_transform = (benchmark_df['val'] - benchmark_df.groupby('category_id')['val'].transform('mean')).values
t_transform = time.perf_counter() - t0

speedup = t_apply / t_transform if t_transform > 0 else 0
print(f"Benchmark on {N:,} rows:")
print(f"  .apply(lambda ...) time:     {t_apply:.4f} seconds")
print(f"  .transform('mean') time:    {t_transform:.4f} seconds")
print(f"  -> Vectorized Speedup:       {speedup:.1f}x faster!")

print("\n" + "=" * 65)
print(" Day 2 Mastery Complete! Proceed to drill_20_problems.py ")
print("=" * 65)


if __name__ == "__main__":
    pass
