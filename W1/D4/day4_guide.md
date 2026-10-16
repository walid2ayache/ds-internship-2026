# 🗓️ Day 4 — NumPy Fundamentals — Full Learning Guide

---

## Part 1: ndarray — The Core of NumPy

### 1.1 · What is NumPy?

NumPy = **Num**erical **Py**thon. It's the foundation of the entire Python data-science stack (Pandas, scikit-learn, PyTorch — all built on NumPy).

**Why not just use Python lists?**

| Feature | Python `list` | NumPy `ndarray` |
|---------|--------------|-----------------|
| Speed | Slow (interpreted loops) | Fast (C under the hood) |
| Memory | Each element is a full Python object | Compact, contiguous block of memory |
| Operations | Element-by-element requires loops | **Vectorized** — operates on entire arrays at once |
| Types | Mixed types allowed | **Homogeneous** — all elements same type |

```python
import numpy as np

# Python list — slow for math
py_list = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in py_list]        # need a loop

# NumPy array — fast, no loop needed
arr = np.array([1, 2, 3, 4, 5])
doubled = arr * 2                          # array([2, 4, 6, 8, 10])
```

> [!IMPORTANT]
> **Rule for today: NO `for` loops for numerical operations.** If you're looping over array elements, you're doing it wrong. Use vectorized operations instead.

---

### 1.2 · Creating Arrays

```python
import numpy as np

# From a Python list
a = np.array([1, 2, 3])                    # 1D array
b = np.array([[1, 2, 3], [4, 5, 6]])       # 2D array (matrix)

# Common constructors
np.zeros((3, 4))       # 3×4 matrix of 0.0
np.ones((2, 3))        # 2×3 matrix of 1.0
np.full((2, 2), 7)     # 2×2 matrix filled with 7
np.eye(3)              # 3×3 identity matrix

# Ranges
np.arange(0, 10, 2)    # array([0, 2, 4, 6, 8])  — like range(), but returns ndarray
np.linspace(0, 1, 5)   # array([0., 0.25, 0.5, 0.75, 1.0])  — 5 evenly spaced points

# Random
np.random.rand(3, 4)         # 3×4 matrix of uniform random [0, 1)
np.random.randn(3, 4)        # 3×4 matrix of standard normal (mean=0, std=1)
np.random.randint(0, 10, (3, 4))  # 3×4 matrix of random ints in [0, 10)
```

> [!TIP]
> Use `np.random.seed(42)` at the top of your script to make random results **reproducible** — you'll get the same "random" numbers every time.

---

### 1.3 · `dtype` — Data Types

Every ndarray has a `dtype` — the type of ALL its elements.

```python
a = np.array([1, 2, 3])
print(a.dtype)          # int64 (or int32 on some systems)

b = np.array([1.0, 2.0, 3.0])
print(b.dtype)          # float64

c = np.array([1, 2, 3], dtype=np.float32)   # force a specific type
print(c.dtype)          # float32
print(c)                # array([1., 2., 3.], dtype=float32)
```

**Common dtypes:**
| dtype | Description | Bytes |
|-------|-------------|-------|
| `np.int32` | 32-bit integer | 4 |
| `np.int64` | 64-bit integer | 8 |
| `np.float32` | 32-bit float (single precision) | 4 |
| `np.float64` | 64-bit float (double precision) | 8 |
| `np.bool_` | Boolean | 1 |
| `np.str_` | String | varies |

**Converting dtype:**
```python
a = np.array([1, 2, 3])
b = a.astype(np.float64)    # creates a NEW array with different dtype
print(b)                     # array([1., 2., 3.])
```

> [!WARNING]
> `astype()` always returns a **new** array. It does NOT modify in place.

---

### 1.4 · `shape`, `ndim`, `size` — Understanding Dimensions

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

a.shape     # (2, 3)  — 2 rows, 3 columns
a.ndim      # 2       — number of dimensions (axes)
a.size      # 6       — total number of elements (2 × 3)
```

**How to read `shape`:**
```
shape = (2, 3, 4)
         │  │  │
         │  │  └─ axis 2: 4 elements deep
         │  └──── axis 1: 3 columns
         └─────── axis 0: 2 rows
```

> [!NOTE]
> Axis numbering starts from the **outermost** dimension. For a 2D array: axis 0 = rows (vertical), axis 1 = columns (horizontal). This is crucial for `sum`, `mean`, etc.

---

### 1.5 · `reshape` — Changing the Shape

`reshape` gives you a **new view** of the same data with a different shape. The total number of elements must stay the same.

```python
a = np.arange(12)       # array([ 0,  1,  2, ..., 11])  — shape (12,)

b = a.reshape(3, 4)     # reshape to 3 rows × 4 columns
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

c = a.reshape(2, 2, 3)  # 3D: 2 blocks × 2 rows × 3 columns

# Use -1 to let NumPy calculate one dimension automatically
d = a.reshape(3, -1)    # -1 means "figure it out" → (3, 4)
e = a.reshape(-1, 6)    # → (2, 6)
```

**Flatten (back to 1D):**
```python
b = np.array([[1, 2], [3, 4]])

b.flatten()       # array([1, 2, 3, 4])  — returns a COPY
b.ravel()         # array([1, 2, 3, 4])  — returns a VIEW (faster, shares memory)
b.reshape(-1)     # array([1, 2, 3, 4])  — same as ravel
```

> [!CAUTION]
> `reshape` fails if the total element count doesn't match: `np.arange(10).reshape(3, 4)` → **error** (10 ≠ 12).

---

### ✏️ Exercise 1 — Array Basics

Create `task1.py`. **No `for` loops allowed!**

1. Create a 1D array of integers from 1 to 20 (inclusive)
2. Reshape it to a 4×5 matrix
3. Print the `shape`, `ndim`, `size`, and `dtype`
4. Convert the matrix to `float32` and verify the dtype changed
5. Create a 3×3 identity matrix
6. Create a 5×5 matrix of random integers between 1 and 100

---

## Part 2: Indexing, Slicing & Boolean Masks

### 2.1 · Basic Indexing (works like Python lists)

```python
a = np.array([10, 20, 30, 40, 50])

a[0]        # 10       — first element
a[-1]       # 50       — last element
a[1:4]      # array([20, 30, 40])  — slice (start:stop, stop excluded)
a[:3]       # array([10, 20, 30])  — first 3
a[::2]      # array([10, 30, 50])  — every other element
```

### 2.2 · 2D Indexing — `[row, col]`

```python
m = np.array([[1,  2,  3,  4],
              [5,  6,  7,  8],
              [9, 10, 11, 12]])

m[0, 0]       # 1       — row 0, col 0
m[2, 3]       # 12      — row 2, col 3
m[1]          # array([5, 6, 7, 8])  — entire row 1
m[:, 2]       # array([3, 7, 11])    — entire column 2
m[0:2, 1:3]   # sub-matrix: rows 0-1, cols 1-2
              # array([[2, 3],
              #        [6, 7]])
```

**Visual guide for `m[0:2, 1:3]`:**
```
     col0  [col1  col2]  col3
row0 [  1,  [ 2,    3],    4 ]
row1 [  5,  [ 6,    7],    8 ]
row2    9,   10,   11,    12
```

> [!IMPORTANT]
> NumPy slices return **views**, not copies. Modifying a slice modifies the original array!
> ```python
> a = np.array([1, 2, 3, 4, 5])
> b = a[1:4]      # b is a VIEW of a
> b[0] = 99       # this also changes a!
> print(a)        # array([ 1, 99,  3,  4,  5])
> ```
> Use `.copy()` if you need an independent copy: `b = a[1:4].copy()`

---

### 2.3 · Fancy Indexing — Using Arrays as Indices

```python
a = np.array([10, 20, 30, 40, 50])

# Index with a list/array of positions
indices = [0, 2, 4]
a[indices]       # array([10, 30, 50])

# Index with an array of ints
idx = np.array([3, 1, 4, 1])
a[idx]           # array([40, 20, 50, 20])

# 2D fancy indexing
m = np.array([[1, 2], [3, 4], [5, 6]])
rows = [0, 2]
cols = [1, 0]
m[rows, cols]    # array([2, 5])  — elements at (0,1) and (2,0)
```

> [!NOTE]
> Fancy indexing (with arrays/lists) returns a **copy**, unlike slicing which returns a view. This is a key difference!

---

### 2.4 · Boolean Masks — The Most Powerful Indexing

Boolean masks let you **filter arrays by condition** — no loops needed!

```python
a = np.array([10, 25, 30, 5, 42, 18])

# Step 1: create a boolean mask
mask = a > 20
print(mask)      # array([False,  True,  True, False,  True, False])

# Step 2: use the mask to filter
a[mask]          # array([25, 30, 42])  — only elements where mask is True

# Or do it in one line:
a[a > 20]        # array([25, 30, 42])

# Combine conditions with & (and), | (or), ~ (not)
# ⚠️ MUST use parentheses around each condition!
a[(a > 10) & (a < 40)]    # array([25, 30, 18])
a[(a < 10) | (a > 40)]    # array([5, 42])
a[~(a > 20)]               # array([10, 5, 18])  — NOT greater than 20
```

> [!WARNING]
> Use `&`, `|`, `~` for element-wise boolean operations on arrays — **NOT** `and`, `or`, `not`! Those are Python keywords that don't work element-wise on arrays.
> ```python
> # ❌ WRONG — will crash
> a[(a > 10) and (a < 40)]
>
> # ✅ CORRECT
> a[(a > 10) & (a < 40)]
> ```

**Modifying with masks:**
```python
a = np.array([10, 25, 30, 5, 42, 18])

# Set all values > 30 to 30 (clamping)
a[a > 30] = 30
print(a)     # array([10, 25, 30,  5, 30, 18])
```

---

### ✏️ Exercise 2 — Indexing & Masks

Add to `task1.py` (or create `task2.py`). **No `for` loops!**

1. Create a 6×6 matrix of random integers between 1 and 50
2. Extract the second row
3. Extract the last column
4. Extract the 3×3 sub-matrix from the center (rows 1–3, cols 1–3 — remember: 0-indexed, exclusive end)
5. Use a boolean mask to find all elements greater than 25
6. Replace all even numbers in the matrix with -1
7. Count how many elements are greater than 30 (hint: `np.sum()` on a boolean mask)

---

## Part 3: Broadcasting & Vectorized Operations

### 3.1 · Arithmetic is Element-wise

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

a + b       # array([11, 22, 33])
a * b       # array([10, 40, 90])
a ** 2      # array([1, 4, 9])
b / a       # array([10., 10., 10.])
```

This also works with 2D arrays:
```python
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[10, 20], [30, 40]])

m1 + m2     # array([[11, 22], [33, 44]])
m1 * m2     # array([[10, 40], [90, 160]])  — element-wise, NOT matrix multiply!
```

> [!NOTE]
> `*` is **element-wise** multiplication. For matrix multiplication, use `@` or `np.dot()`:
> ```python
> m1 @ m2          # matrix multiplication
> np.dot(m1, m2)   # same thing
> ```

---

### 3.2 · Broadcasting — The Magic Rule

Broadcasting is how NumPy handles operations between arrays of **different shapes**. Instead of throwing an error, NumPy "stretches" the smaller array to match.

**The simple rule:** Two dimensions are compatible when:
1. They are **equal**, OR
2. One of them is **1** (it gets stretched)

```python
# Scalar broadcast — the scalar is "stretched" to match the array
a = np.array([1, 2, 3])
a + 10      # array([11, 12, 13])   — 10 broadcasts to [10, 10, 10]
a * 2       # array([2, 4, 6])      — 2 broadcasts to [2, 2, 2]

# 1D + 2D broadcast
m = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)
v = np.array([10, 20, 30])     # shape (3,)

m + v
# array([[11, 22, 33],
#        [14, 25, 36]])
# v is broadcast across each row of m
```

**Visual explanation:**
```
m (2, 3):       v (3,) → broadcasts to (2, 3):
[[1, 2, 3],     [[10, 20, 30],
 [4, 5, 6]]  +   [10, 20, 30]]
```

**Column broadcast (need reshape!):**
```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)
col = np.array([100, 200])      # shape (2,)

# m + col  →  ERROR! shapes (2,3) and (2,) don't broadcast along the right axis

# Fix: reshape col to (2, 1) so it broadcasts across columns
col = col.reshape(-1, 1)        # shape (2, 1)
m + col
# array([[101, 102, 103],
#        [204, 205, 206]])
```

**Broadcasting alignment (right-to-left):**
```
Shape A:     (2, 3)
Shape B:        (3,)   ← padded to (1, 3) → broadcasts to (2, 3) ✅

Shape A:     (2, 3)
Shape B:     (2, 1)   → broadcasts to (2, 3) ✅

Shape A:     (2, 3)
Shape B:        (2,)  ← padded to (1, 2) → (2, 3) vs (1, 2) MISMATCH ❌
```

> [!TIP]
> When in doubt, use `.reshape()` to make shapes explicit. `arr.reshape(-1, 1)` makes a column vector, `arr.reshape(1, -1)` makes a row vector.

---

### 3.3 · `axis` — The Direction of Operations

Many NumPy functions take an `axis` parameter to specify **which direction** to operate along.

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)

# No axis — operates on ALL elements (flattened)
np.sum(m)         # 21

# axis=0 — collapse ROWS (operate down each column)
np.sum(m, axis=0)   # array([5, 7, 9])   — sum of each column

# axis=1 — collapse COLUMNS (operate across each row)
np.sum(m, axis=1)   # array([6, 15])     — sum of each row
```

**Memory trick for `axis`:**
```
axis=0: the result loses the row    dimension → shape (2,3) → (3,)   — "sum DOWN"
axis=1: the result loses the column dimension → shape (2,3) → (2,)   — "sum ACROSS"
```

Think of it as: **`axis=N` collapses (removes) dimension N**.

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])

np.mean(m, axis=0)    # array([2.5, 3.5, 4.5])  — mean of each column
np.mean(m, axis=1)    # array([2., 5.])           — mean of each row

np.max(m, axis=0)     # array([4, 5, 6])          — max of each column
np.min(m, axis=1)     # array([1, 4])              — min of each row
```

---

### 3.4 · Aggregation Functions — `sum`, `mean`, `std`, `min`, `max`

```python
a = np.array([10, 20, 30, 40, 50])

np.sum(a)       # 150      (or a.sum())
np.mean(a)      # 30.0     (or a.mean())
np.std(a)       # 14.14    (or a.std())  — standard deviation
np.var(a)       # 200.0    (or a.var())  — variance
np.min(a)       # 10       (or a.min())
np.max(a)       # 50       (or a.max())
np.median(a)    # 30.0     (no method form — only np.median())
np.prod(a)      # 12000000 — product of all elements
```

**Cumulative versions:**
```python
a = np.array([1, 2, 3, 4, 5])

np.cumsum(a)    # array([ 1,  3,  6, 10, 15])  — running total
np.cumprod(a)   # array([  1,   2,   6,  24, 120])  — running product
```

---

### 3.5 · `argmax`, `argmin` — Finding Positions

These return the **index** (position) of the max/min value, not the value itself.

```python
a = np.array([10, 50, 30, 20, 40])

np.argmax(a)    # 1   — index of max value (50 is at index 1)
np.argmin(a)    # 3   — index of min value (20 is at index 3)

# Get the actual value:
a[np.argmax(a)] # 50

# With 2D arrays:
m = np.array([[1, 9, 3],
              [7, 2, 8]])

np.argmax(m)            # 1   — index in FLATTENED array (9 is at flat index 1)
np.argmax(m, axis=0)    # array([1, 0, 1])  — row index of max in each column
np.argmax(m, axis=1)    # array([1, 2])      — col index of max in each row

# Convert flat index to 2D coordinates:
np.unravel_index(np.argmax(m), m.shape)   # (0, 1)  — row 0, col 1
```

---

### 3.6 · `np.where` — Conditional Selection

`np.where(condition, x, y)` returns `x` where condition is True, `y` where it's False. Think of it as a **vectorized if-else**.

```python
a = np.array([1, -2, 3, -4, 5])

# Replace negatives with 0
np.where(a > 0, a, 0)       # array([1, 0, 3, 0, 5])

# Classify as "pos" or "neg"
np.where(a > 0, "pos", "neg")   # array(['pos', 'neg', 'pos', 'neg', 'pos'])

# np.where with only condition — returns indices where True
np.where(a > 0)     # (array([0, 2, 4]),)  — indices of positive elements
```

**More useful patterns:**
```python
scores = np.array([85, 42, 73, 91, 58, 66])

# Assign grades
grades = np.where(scores >= 70, "pass", "fail")
# array(['pass', 'fail', 'pass', 'pass', 'fail', 'fail'])

# Clip values to a range (though np.clip is simpler for this)
clipped = np.where(scores > 80, 80, scores)    # cap at 80
clipped = np.where(clipped < 50, 50, clipped)  # floor at 50
# Simpler: np.clip(scores, 50, 80)
```

---

### ✏️ Exercise 3 — Broadcasting & Aggregation

Create `task2.py`. **No `for` loops!**

1. Create a 5×4 matrix of random integers between 0 and 100 (use `np.random.seed(42)` first)
2. Compute the mean of each column and each row
3. Subtract the column means from each element (center the data — broadcasting!)
4. Find the index of the maximum value in the entire matrix (use `argmax` + `unravel_index`)
5. Use `np.where` to create a new matrix where values below 50 are replaced with 0 and values ≥ 50 are replaced with 1
6. Compute the standard deviation of each row

---

## Part 4: Useful NumPy Operations

### 4.1 · Sorting

```python
a = np.array([30, 10, 50, 20, 40])

np.sort(a)        # array([10, 20, 30, 40, 50])  — returns sorted COPY
a.sort()          # sorts IN PLACE (modifies a)

# Sort indices (useful for sorting one array by another)
idx = np.argsort(a)     # array([1, 3, 0, 4, 2])  — indices that would sort the array
a[idx]                  # array([10, 20, 30, 40, 50])

# 2D sorting
m = np.array([[3, 1, 2],
              [6, 4, 5]])
np.sort(m, axis=0)    # sort each column
np.sort(m, axis=1)    # sort each row
```

---

### 4.2 · Stacking & Splitting

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stacking
np.concatenate([a, b])              # array([1, 2, 3, 4, 5, 6])
np.vstack([a, b])                   # array([[1, 2, 3], [4, 5, 6]])  — vertical stack
np.hstack([a.reshape(-1,1), b.reshape(-1,1)])  # horizontal stack (need 2D)

# For 2D arrays:
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

np.vstack([m1, m2])    # shape (4, 2) — stack vertically
np.hstack([m1, m2])    # shape (2, 4) — stack horizontally

# Splitting
a = np.arange(12).reshape(4, 3)
np.vsplit(a, 2)    # split into 2 arrays along rows
np.hsplit(a, 3)    # split into 3 arrays along columns
```

---

### 4.3 · `np.clip`, `np.unique`, `np.count_nonzero`

```python
a = np.array([2, 8, 15, 1, 25, 6])

# Clip — constrain values to [min, max]
np.clip(a, 5, 20)    # array([ 5,  8, 15,  5, 20,  6])

# Unique values
b = np.array([3, 1, 2, 3, 1, 2, 4])
np.unique(b)                        # array([1, 2, 3, 4])
values, counts = np.unique(b, return_counts=True)
# values: array([1, 2, 3, 4])
# counts: array([2, 2, 2, 1])

# Count non-zero elements (also works with boolean arrays!)
np.count_nonzero(a > 5)     # 3  — same as np.sum(a > 5)
```

---

### 4.4 · Vectorize: Replacing Loops

Here's the core philosophy: every `for` loop over array elements can (almost always) be replaced with a vectorized operation.

**❌ Loop way (slow):**
```python
# Normalize an array to [0, 1]
a = np.array([10, 20, 30, 40, 50])
result = np.empty_like(a, dtype=float)
for i in range(len(a)):
    result[i] = (a[i] - a.min()) / (a.max() - a.min())
```

**✅ Vectorized way (fast):**
```python
a = np.array([10, 20, 30, 40, 50])
result = (a - a.min()) / (a.max() - a.min())
# array([0., 0.25, 0.5, 0.75, 1.0])
```

**More vectorization patterns:**

```python
# ❌ Loop: Euclidean distance
def dist_loop(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total ** 0.5

# ✅ Vectorized
def dist_vec(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# ❌ Loop: count values in range
count = 0
for x in a:
    if 20 <= x <= 40:
        count += 1

# ✅ Vectorized
count = np.sum((a >= 20) & (a <= 40))
```

> [!TIP]
> **Speed comparison:** Vectorized NumPy is typically **50–100× faster** than Python loops for large arrays. The bigger the array, the bigger the speedup.

---

### ✏️ Exercise 4 — Putting It All Together

Create `task3.py`. **No `for` loops!**

1. **Normalize:** Create an array of 10 random integers (0–100). Normalize them to the range [0, 1] using: `(x - min) / (max - min)`
2. **Temperature conversion:** Create an array of temperatures in Celsius: `[0, 10, 20, 30, 40]`. Convert to Fahrenheit: `F = C × 9/5 + 32`. Convert back to Celsius to verify.
3. **Statistics:** Create a 100×4 matrix of random numbers (normal distribution). Compute mean, std, min, max of each column.
4. **Boolean masking:** From the matrix above, count how many values in each column are within 1 standard deviation of the column mean.
5. **Sorting:** Create an array of 20 random integers. Find the 5 largest values (hint: sort, then slice).

---

## Part 5: Drill — 25 NumPy Exercises

These exercises build muscle memory. **All solutions must be vectorized — NO `for` loops.**

Create `drills.py` and solve each one:

```python
import numpy as np

# ─── Creation & Inspection ───────────────────────────────────

# 1. Create a vector of size 10 filled with zeros, but the 5th element is 1
# Expected: [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]

# 2. Create a 3×3 matrix with values from 1 to 9

# 3. Find the indices of non-zero elements in [1, 2, 0, 0, 4, 0]

# 4. Create a 10×10 array with random values and find the min and max

# 5. Create a random vector of size 30 and find the mean


# ─── Reshaping & Stacking ────────────────────────────────────

# 6. Create a 5×5 matrix with row values ranging from 0 to 4
# Hint: use np.arange + broadcasting or np.tile

# 7. Reverse the rows of a 2D array (first row becomes last)

# 8. Reverse the columns of a 2D array (first col becomes last)

# 9. Create a 8×8 checkerboard pattern of 0s and 1s
# Hint: use slicing

# 10. Flatten a 3D array of shape (2, 3, 4) into 1D


# ─── Indexing & Boolean Masks ────────────────────────────────

# 11. Given array [1, 2, 3, 4, 5, 6, 7, 8, 9], replace all odd values with -1

# 12. Given a 5×5 random integer matrix, set all elements on the border to 0
# (first/last row, first/last column)

# 13. Given two arrays, find the common values (intersection)
# a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
# b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])

# 14. Clamp all values in a random array to be between 0.2 and 0.8

# 15. Extract all integers from a random array between 3 and 8 (exclusive)


# ─── Broadcasting & Math ─────────────────────────────────────

# 16. Subtract the mean of each row from a 5×5 matrix (row-wise centering)

# 17. Normalize each column of a matrix to have mean=0 and std=1
# Formula: (x - mean) / std

# 18. Compute the Euclidean distance between two random vectors of length 10

# 19. Given a 1D array, create a 2D array where each row is the original
#     shifted by one position to the right (with zeros filling in)
# Input:  [1, 2, 3, 4]
# Output: [[1, 2, 3, 4],
#          [0, 1, 2, 3],
#          [0, 0, 1, 2],
#          [0, 0, 0, 1]]

# 20. Multiply a 5×3 matrix by a 3×2 matrix (matrix multiplication)


# ─── Aggregation & Analysis ──────────────────────────────────

# 21. Find the most frequent value in an array
# Hint: np.unique with return_counts, then argmax

# 22. Given a 2D array, find the row with the highest sum

# 23. Compute the running (cumulative) sum of a 1D array

# 24. Given an array of exam scores [85, 42, 73, 91, 58, 66, 77, 49, 88, 95],
#     compute: mean, median, std, number passing (≥60), percentage passing

# 25. Create two random 1000-element arrays and verify that
#     np.sum(a * b) ≈ np.dot(a, b)  (they should be nearly equal)
```

---

## 🗺️ Quick Reference Cheat Sheet

| Topic | Key Takeaway |
|-------|-------------|
| `np.array()` | Create an ndarray from a list. All elements same type |
| `dtype` | Type of elements: `int64`, `float32`, `bool_`. Convert with `.astype()` |
| `shape` / `reshape` | `shape` = dimensions tuple. `reshape(-1, 3)` = let NumPy figure out rows |
| Indexing | `a[row, col]`, `a[:, 2]` = all rows col 2, `a[1:3, :]` = rows 1-2 |
| Boolean mask | `a[a > 5]` — filters without loops. Use `&`, `|`, `~` (with parentheses!) |
| Broadcasting | Scalar auto-expands. `(2,3) + (3,)` works. `(2,3) + (2,1)` works |
| `axis=0` | Operate **down** rows → result has one value per column |
| `axis=1` | Operate **across** columns → result has one value per row |
| `sum/mean/std` | `np.sum(a)`, `np.mean(a, axis=0)`, `a.std()` — with or without axis |
| `argmax/argmin` | Returns the **index** of the max/min, not the value |
| `np.where` | `np.where(cond, x, y)` — vectorized if-else |
| `np.sort/argsort` | Sort values / get indices that would sort |
| `np.clip` | `np.clip(a, lo, hi)` — constrain values to range |
| `np.unique` | Unique values + optional counts |
| Vectorize | **No `for` loops!** Use array operations: `a * 2`, `a[a > 0]`, `np.where` |
