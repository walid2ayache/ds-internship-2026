"""
Day 2: 20-Problem Aggregation & Joins Practice Template (Blank)
================================================================
Use this template to test your Pandas aggregation, transformation, reshaping,
merging, validation, and concatenation skills from scratch.

Instructions:
1. Replace each `raise NotImplementedError(...)` with your own implementation.
2. Run this script using:
       python drill_blank_practice.py
3. Watch each drill turn from [PENDING] to [PASS] as you solve them!
4. If you get stuck, check `drill_20_problems.py` for solutions and explanations.
5. Rule: Strictly avoid using `.apply()` across all problems unless specifically requested!
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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 01")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 02")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 03")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 04")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 05")


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
    
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 06")


# ==============================================================================
# DRILL 7: Group-Level Filtering with .filter()
# ==============================================================================
def drill_07(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Problem 7:
    Filter `orders` to retain ONLY rows belonging to sales reps who have
    accumulated a TOTAL ordered quantity of at least 8 units across all their orders.
    """
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 07")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 08")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 09")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 10")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 11")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 12")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 13")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 14")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 15")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 16")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 17")


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

    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 18")


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

    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 19")


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
    # TODO: Your implementation here
    raise NotImplementedError("Implement Drill 20")


# ==============================================================================
# PRACTICE TEST RUNNER
# ==============================================================================
def run_practice_tests():
    print("=" * 65)
    print(" Day 2: Aggregation & Joins Practice Test Runner")
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
    pending = 0
    failed = 0

    for name, drill_fn in drills:
        try:
            drill_fn()
            print(f"[PASS]    {name}")
            passed += 1
        except NotImplementedError:
            print(f"[PENDING] {name}")
            pending += 1
        except Exception as e:
            print(f"[FAIL]    {name}: {e}")
            failed += 1

    print("=" * 65)
    print(f" Summary: {passed} PASSED | {pending} PENDING | {failed} FAILED")
    print("=" * 65)


if __name__ == '__main__':
    run_practice_tests()
