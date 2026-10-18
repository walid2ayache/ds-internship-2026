"""
Day 1: Pandas Core Mastery
==========================
This script demonstrates all core concepts required for Day 1:
- Series & DataFrame creation
- Indexing & Slicing: loc vs iloc
- Boolean filtering & bitwise operations
- .query() method with local variable injection
- Sorting values and indices
- Data types, astype conversions, downcasting
- Descriptive summaries: info, describe, value_counts
"""

import pandas as pd
import numpy as np

def header(title: str):
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)

# ==========================================
# 1. SERIES & DATAFRAME CREATION
# ==========================================
header("1. Series & DataFrame Basics")

# Series with custom index and name
temperatures = pd.Series([22.5, 24.1, 19.8, np.nan, 28.3],
                         index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                         name='Celsius')
print("Series:")
print(temperatures)
print("Index:", temperatures.index)
print("Underlying array:", temperatures.to_numpy())

# DataFrame from dictionary of lists
data = {
    'employee_id': ['EMP101', 'EMP102', 'EMP103', 'EMP104', 'EMP105', 'EMP106'],
    'full_name': ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince', 'Evan Wright', 'Fiona Gallagher'],
    'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Finance', 'Engineering'],
    'years_exp': [3, 7, 2, 9, 5, 12],
    'salary': [72000, 85000, 68000, 93000, 81000, 135000],
    'is_remote': [True, False, True, False, True, False]
}

df = pd.DataFrame(data)
# Set employee_id as index
df.set_index('employee_id', inplace=True)
print("\nDataFrame:")
print(df)


# ==========================================
# 2. LOC VS ILOC
# ==========================================
header("2. Indexing & Slicing: loc vs iloc")

# loc: Label-based (inclusive of endpoint)
print("--- df.loc['EMP102':'EMP104', ['full_name', 'salary']] (End label included!) ---")
print(df.loc['EMP102':'EMP104', ['full_name', 'salary']])

# iloc: Integer position-based (exclusive of endpoint, like Python slices)
print("\n--- df.iloc[1:4, [0, 3]] (Index 4 is excluded!) ---")
print(df.iloc[1:4, [0, 3]])

# Single scalar lookup with at and iat (much faster than loc/iloc)
print("\nScalar lookup at['EMP101', 'salary']:", df.at['EMP101', 'salary'])
print("Scalar lookup iat[0, 3]:", df.iat[0, 3])


# ==========================================
# 3. BOOLEAN FILTERING
# ==========================================
header("3. Boolean Filtering & Masking")

# Single condition
mask_high_salary = df['salary'] > 80000
print("High earners (> $80,000):")
print(df[mask_high_salary][['full_name', 'salary']])

# Compound condition: & (AND), | (OR), ~ (NOT)
print("\nEngineering employees with > 2 years experience who work remotely:")
mask_compound = (df['department'] == 'Engineering') & (df['years_exp'] > 2) & (df['is_remote'] == True)
print(df[mask_compound][['full_name', 'department', 'years_exp', 'is_remote']])

# .isin() method
print("\nEmployees in HR or Finance:")
print(df[df['department'].isin(['HR', 'Finance'])][['full_name', 'department']])

# .between() method (inclusive=both by default)
print("\nEmployees with salary between $70,000 and $90,000:")
print(df[df['salary'].between(70000, 90000)][['full_name', 'salary']])

# String filtering via .str accessor
print("\nEmployees with 'Brown' or 'Prince' in their name:")
print(df[df['full_name'].str.contains('Brown|Prince', regex=True)][['full_name', 'department']])


# ==========================================
# 4. THE .query() METHOD
# ==========================================
header("4. The .query() Method")

# Clean, readable SQL-like string querying
target_dept = 'Engineering'
threshold_salary = 70000

# Using @ to reference variables from the environment
query_res = df.query("department == @target_dept and salary > @threshold_salary and is_remote == True")
print(f"df.query(\"department == @target_dept and salary > @threshold_salary and is_remote == True\"):")
print(query_res[['full_name', 'department', 'salary', 'is_remote']])


# ==========================================
# 5. SORTING: .sort_values() & .sort_index()
# ==========================================
header("5. Sorting: .sort_values() & .sort_index()")

# Sort by single column descending
print("Sorted by salary descending:")
print(df.sort_values(by='salary', ascending=False)[['full_name', 'salary']])

# Multi-column sort: department ascending, then years_exp descending
print("\nSorted by Department (asc) and Years Exp (desc):")
print(df.sort_values(by=['department', 'years_exp'], ascending=[True, False])[
    ['full_name', 'department', 'years_exp', 'salary']
])

# Sort by index
print("\nSorted by index alphabetically descending:")
print(df.sort_index(ascending=False))


# ==========================================
# 6. DTYPES & TYPE CONVERSION (.astype)
# ==========================================
header("6. Data Types & .astype()")

print("Current dtypes:")
print(df.dtypes)

# Convert department to 'category' and years_exp to int8 for memory efficiency
df_optimized = df.copy()
df_optimized['department'] = df_optimized['department'].astype('category')
df_optimized['years_exp'] = df_optimized['years_exp'].astype('int8')

print("\nOptimized dtypes:")
print(df_optimized.dtypes)
print(f"Memory reduction in department: {df['department'].memory_usage()} bytes -> {df_optimized['department'].memory_usage()} bytes")


# ==========================================
# 7. SUMMARY FUNCTIONS: info, describe, value_counts
# ==========================================
header("7. Inspection: info, describe, value_counts")

print("--- df.info() ---")
df.info()

print("\n--- df.describe() (Numeric) ---")
print(df.describe())

print("\n--- df.describe(include='all') (Mixed types) ---")
print(df.describe(include='all'))

print("\n--- df['department'].value_counts() ---")
print(df['department'].value_counts())

print("\n--- df['department'].value_counts(normalize=True) (Proportions) ---")
print(df['department'].value_counts(normalize=True).round(3))

print("\n--- df['is_remote'].value_counts() ---")
print(df['is_remote'].value_counts())

if __name__ == '__main__':
    print("\n[+] Pandas Core Mastery demonstration completed successfully!")
