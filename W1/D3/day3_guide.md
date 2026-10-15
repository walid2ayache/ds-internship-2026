# 🗓️ Day 3 — Files, Errors & OOP — Full Learning Guide

---

## Part 1: Files — `with open`, CSV, JSON, `pathlib`

### 1.1 · `with open` — The Safe Way to Handle Files

**The problem:** When you open a file, you MUST close it. If your code crashes before `f.close()`, the file stays locked.

**The solution:** `with` statement (context manager) — it **auto-closes** the file, even if an error happens.

```python
# ❌ Old way — risky
f = open("data.txt", "r")
content = f.read()
f.close()  # what if an error happens before this line?

# ✅ Safe way — with statement
with open("data.txt", "r") as f:
    content = f.read()
# file is automatically closed here, even if an error occurs
```

**File modes:**
| Mode | Meaning | Creates file? | Overwrites? |
|------|---------|--------------|-------------|
| `"r"` | Read (default) | ❌ | — |
| `"w"` | Write | ✅ | ✅ Yes, destroys old content |
| `"a"` | Append | ✅ | ❌ Adds to end |
| `"r+"` | Read + Write | ❌ | Partial |

**Key methods:**
```python
with open("file.txt", "r") as f:
    f.read()        # entire file as one string
    f.readline()    # one line at a time
    f.readlines()   # list of all lines

with open("file.txt", "w") as f:
    f.write("hello\n")       # write a string
    f.writelines(["a\n", "b\n"])  # write a list of strings
```

> [!TIP]
> You can loop over a file directly:
> ```python
> with open("file.txt") as f:
>     for line in f:        # memory-efficient, reads one line at a time
>         print(line.strip())  # .strip() removes the trailing \n
> ```

---

### 1.2 · CSV Files

CSV = Comma-Separated Values. Each line is a row, commas separate columns.

```
name,age,city
Alice,30,Paris
Bob,25,London
```

**Reading CSV:**
```python
import csv

with open("people.csv", "r") as f:
    reader = csv.reader(f)        # returns an iterator of lists
    header = next(reader)         # ['name', 'age', 'city']
    for row in reader:
        print(row)                # ['Alice', '30', 'Paris']
```

**Reading CSV as dictionaries (most useful!):**
```python
import csv

with open("people.csv", "r") as f:
    reader = csv.DictReader(f)    # each row becomes a dict
    for row in reader:
        print(row["name"], row["age"])  # Alice 30
```

**Writing CSV:**
```python
import csv

data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
]

with open("output.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age"])
    writer.writeheader()       # writes: name,age
    writer.writerows(data)     # writes all rows
```

> [!IMPORTANT]
> Always use `newline=""` when opening CSV files for writing on Windows, otherwise you get blank lines between rows.

---

### 1.3 · JSON Files

JSON = JavaScript Object Notation. Looks almost like Python dicts/lists.

```json
{"name": "Alice", "age": 30, "hobbies": ["reading", "coding"]}
```

**The `json` module has 4 key functions:**

| Function | What it does |
|----------|-------------|
| `json.load(f)` | Read JSON from a **file** → Python object |
| `json.loads(s)` | Read JSON from a **string** → Python object |
| `json.dump(obj, f)` | Write Python object → **file** as JSON |
| `json.dumps(obj)` | Write Python object → **string** as JSON |

> [!TIP]
> Memory trick: the `s` stands for **s**tring. `loads` = load from **s**tring, `dumps` = dump to **s**tring.

```python
import json

# Reading JSON
with open("data.json", "r") as f:
    data = json.load(f)       # now data is a Python dict or list

# Writing JSON
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)  # indent=2 makes it pretty-printed
```

---

### 1.4 · `pathlib` — Modern File Paths

`pathlib` replaces ugly string manipulation with an object-oriented approach.

```python
from pathlib import Path

# Creating paths
p = Path("data") / "files" / "input.csv"   # data/files/input.csv
# the / operator joins paths — no more os.path.join()!

# Useful properties
p.name          # "input.csv"
p.stem          # "input"       (filename without extension)
p.suffix        # ".csv"
p.parent        # Path("data/files")
p.exists()      # True/False
p.is_file()     # True/False
p.is_dir()      # True/False

# Reading/writing (shortcut — no need for open()!)
text = p.read_text()
p.write_text("hello world")

# Listing files in a directory
for file in Path("data").glob("*.csv"):    # all CSV files
    print(file)

for file in Path("data").rglob("*.csv"):   # recursive (subdirs too)
    print(file)

# Creating directories
Path("output").mkdir(exist_ok=True)        # no error if already exists
Path("a/b/c").mkdir(parents=True, exist_ok=True)  # create parent dirs too
```

---

### ✏️ Exercise 1 — Try It Yourself

Create a file called `task1.py`. Do the following:

1. Create a sample CSV file called `sample.csv` with columns: `name, age, city` and at least 3 rows
2. Read it with `csv.DictReader`
3. Print each person's name and city
4. Convert the data to a list of dicts and save it as `sample.json` using `json.dump`
5. Use `pathlib.Path` to check if `sample.json` exists after writing, and print a confirmation

---

## Part 2: Error Handling — `try/except/finally`, Custom Exceptions, `logging`

### 2.1 · Why Handle Errors?

Without error handling, one bad input crashes your entire program. Error handling lets you **catch problems gracefully**.

### 2.2 · `try / except / else / finally`

```python
try:
    # code that MIGHT fail
    result = 10 / 0
except ZeroDivisionError:
    # runs ONLY if that specific error happens
    print("Can't divide by zero!")
except (ValueError, TypeError) as e:
    # catch multiple types; 'e' holds the error details
    print(f"Bad value: {e}")
except Exception as e:
    # catch-all for any other error (use sparingly!)
    print(f"Something went wrong: {e}")
else:
    # runs ONLY if NO error happened (optional)
    print(f"Result is {result}")
finally:
    # ALWAYS runs — error or not (optional)
    # great for cleanup (closing connections, etc.)
    print("Done.")
```

**The flow:**
```
try → (error?) → except → finally
try → (no error?) → else → finally
```

> [!WARNING]
> Never use bare `except:` without specifying an exception type. It catches **everything** including `KeyboardInterrupt` and `SystemExit`, making your program impossible to stop.

**Common built-in exceptions:**
| Exception | When it happens |
|-----------|----------------|
| `ValueError` | Wrong value: `int("abc")` |
| `TypeError` | Wrong type: `"5" + 5` |
| `KeyError` | Missing dict key: `d["missing"]` |
| `IndexError` | List index out of range |
| `FileNotFoundError` | File doesn't exist |
| `ZeroDivisionError` | Division by zero |

---

### 2.3 · Custom Exceptions

You can create your own exception types to represent **domain-specific errors**:

```python
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class AgeError(ValidationError):
    """Raised when age is invalid."""
    def __init__(self, age, message="Age must be between 0 and 150"):
        self.age = age
        self.message = message
        super().__init__(self.message)    # call parent's __init__

# Using it:
def validate_age(age):
    if age < 0 or age > 150:
        raise AgeError(age, f"Invalid age: {age}")
    return age

try:
    validate_age(200)
except AgeError as e:
    print(e.message)   # "Invalid age: 200"
    print(e.age)        # 200
```

> [!NOTE]
> Custom exceptions should inherit from `Exception` (not `BaseException`). You can build a hierarchy: `ValidationError` → `AgeError`, `EmailError`, etc.

---

### 2.4 · `logging` — Professional Error Reporting

`print()` is for users. `logging` is for developers. Logging gives you **severity levels**, **timestamps**, and can write to files.

```python
import logging

# Basic setup
logging.basicConfig(
    level=logging.DEBUG,              # show all messages DEBUG and above
    format="%(asctime)s - %(levelname)s - %(message)s",
    # Optional: log to a file
    # filename="app.log",
)

# 5 severity levels (low → high):
logging.debug("Variable x = 42")         # for detailed debugging
logging.info("Processing started")       # normal operation updates
logging.warning("Disk space low")        # something unexpected but not fatal
logging.error("Failed to read file")     # something failed
logging.critical("Database is down!")     # system is broken
```

**Level hierarchy:**
```
DEBUG < INFO < WARNING < ERROR < CRITICAL
```
If you set `level=logging.WARNING`, only WARNING, ERROR, and CRITICAL messages show up. DEBUG and INFO are silenced.

**Using logging in try/except:**
```python
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    data = int("not_a_number")
except ValueError:
    logging.error("Could not convert to int", exc_info=True)
    # exc_info=True includes the full traceback in the log!
```

---

### ✏️ Exercise 2 — Try It Yourself

Create `task2.py`:

1. Set up `logging` with level `DEBUG` and a format that shows time + level + message
2. Create a custom exception `InvalidRowError(Exception)` with a `row_number` attribute
3. Write a function `validate_row(row, row_num)` that:
   - Checks if `name` is not empty
   - Checks if `age` can be converted to `int` and is between 0–150
   - Raises `InvalidRowError` if validation fails
4. Read your `sample.csv`, validate each row, and log the results (info for valid, error for invalid)

---

## Part 3: OOP — Classes, `@property`, `@dataclass`, Dunder Methods

### 3.1 · Classes — The Blueprint

A class is a **blueprint** for creating objects. Objects bundle **data** (attributes) and **behavior** (methods) together.

```python
class Person:
    # class variable — shared by ALL instances
    species = "Homo sapiens"

    def __init__(self, name, age):
        """Constructor — runs when you create a new Person."""
        # instance variables — unique to each object
        self.name = name
        self.age = age

    def greet(self):
        """Instance method — needs 'self' to access the object's data."""
        return f"Hi, I'm {self.name}, {self.age} years old."

# Creating objects (instances)
alice = Person("Alice", 30)
bob = Person("Bob", 25)

print(alice.greet())       # Hi, I'm Alice, 30 years old.
print(alice.species)       # Homo sapiens
print(Person.species)      # Homo sapiens (access via class too)
```

**`self` explained:** `self` is the **current object**. When you call `alice.greet()`, Python secretly passes `alice` as `self`. It's how methods know which object's data to use.

---

### 3.2 · `@property` — Controlled Access

Properties let you **compute values on the fly** or **add validation** when getting/setting attributes.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius       # _ prefix = "private by convention"

    @property
    def radius(self):
        """Getter — runs when you ACCESS circle.radius"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Setter — runs when you SET circle.radius = something"""
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self):
        """Read-only computed property (no setter defined)"""
        import math
        return math.pi * self._radius ** 2

c = Circle(5)
print(c.radius)    # 5  — calls the getter
print(c.area)      # 78.54  — computed on the fly
c.radius = 10      # calls the setter
c.radius = -1      # raises ValueError!
```

> [!TIP]
> Use `@property` when you want attribute-like syntax (`obj.value`) but need validation or computation behind the scenes.

---

### 3.3 · Dunder (Magic) Methods

"Dunder" = **d**ouble **under**score. These special methods let your objects work with Python's built-in operators and functions.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        """Developer-friendly representation (for debugging)"""
        return f"Vector({self.x}, {self.y})"

    def __str__(self):
        """User-friendly string (used by print())"""
        return f"({self.x}, {self.y})"

    def __len__(self):
        """Called by len()"""
        return 2

    def __add__(self, other):
        """Called by the + operator"""
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        """Called by the == operator"""
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        """Called by < operator — also enables sorting!"""
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1)          # (1, 2)        — __str__
print(repr(v1))    # Vector(1, 2)  — __repr__
print(v1 + v2)     # (4, 6)        — __add__
print(v1 == v2)    # False         — __eq__
```

**Most common dunders:**
| Method | Triggered by | Purpose |
|--------|-------------|---------|
| `__init__` | `MyClass()` | Constructor |
| `__repr__` | `repr(obj)` | Debug string |
| `__str__` | `print(obj)`, `str(obj)` | User-friendly string |
| `__len__` | `len(obj)` | Length |
| `__add__` | `a + b` | Addition |
| `__eq__` | `a == b` | Equality |
| `__lt__` | `a < b` | Less than (enables `sorted()`) |
| `__getitem__` | `obj[key]` | Indexing / subscript |
| `__contains__` | `x in obj` | Membership test |

---

### 3.4 · `@dataclass` — Classes Without the Boilerplate

For classes that are mainly about **storing data**, `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` for you:

```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int
    city: str = "Unknown"    # default value

# Auto-generated __init__:
p = Person("Alice", 30, "Paris")
# Auto-generated __repr__:
print(p)   # Person(name='Alice', age=30, city='Paris')
# Auto-generated __eq__:
print(Person("Alice", 30) == Person("Alice", 30))  # True
```

**Advanced dataclass features:**
```python
from dataclasses import dataclass, field

@dataclass(order=True)      # auto-generates <, >, <=, >= based on fields
class Student:
    name: str
    grade: float
    courses: list = field(default_factory=list)  # mutable default!

    # You can still add custom methods:
    @property
    def is_passing(self):
        return self.grade >= 50.0

s = Student("Alice", 85.5, ["Math", "Physics"])
print(s.is_passing)  # True
```

> [!WARNING]
> Never use mutable defaults directly: `courses: list = []` is a BUG (all instances share the same list!). Always use `field(default_factory=list)`.

---

### ✏️ Exercise 3 — Try It Yourself

Create `task3.py`:

1. Create a `@dataclass` called `Record` with fields: `name: str`, `age: int`, `city: str`
2. Add a `@property` called `is_valid` that returns `True` if age is 0–150 and name is not empty
3. Add `__str__` that returns a nice formatted string
4. Create a few `Record` instances and print them
5. Store them in a list and use `sorted()` (you'll need `__lt__`)

---

## Part 4: The Build — CLI CSV Cleaner (`argparse`)

Now we combine everything into one real project!

### 4.1 · `argparse` — Command-Line Arguments

`argparse` lets users pass options to your script from the terminal:

```python
import argparse

parser = argparse.ArgumentParser(description="Clean a CSV file")
parser.add_argument("input", help="Path to input CSV file")
parser.add_argument("output", help="Path to output JSON file")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

args = parser.parse_args()
print(args.input)     # whatever the user typed as first argument
print(args.output)    # second argument
print(args.verbose)   # True if --verbose was used, False otherwise
```

**Usage from terminal:**
```bash
python clean_csv.py data.csv output.json --verbose
```

**Argument types:**
```python
# Positional (required, order matters)
parser.add_argument("filename")

# Optional (prefixed with -- or -)
parser.add_argument("--count", type=int, default=10)
parser.add_argument("--format", choices=["json", "csv"], default="json")
parser.add_argument("--verbose", "-v", action="store_true")  # flag, no value needed
```

---

### 4.2 · The Build Plan

Your final script `clean_csv.py` should:

```
python clean_csv.py dirty_data.csv clean_output.json --verbose
```

**Architecture:**
```
1. Parse CLI arguments (argparse)
2. Set up logging (verbose → DEBUG, normal → WARNING)
3. Read CSV file (csv.DictReader + pathlib)
4. Validate & clean each row:
   - Strip whitespace from all fields
   - Validate age (must be int, 0-150)
   - Skip invalid rows (log them as warnings)
5. Write clean data to JSON (json.dump)
6. Print summary: X rows cleaned, Y rows skipped
```

**Data structures:**
```python
@dataclass
class Record:
    name: str
    age: int
    city: str

class InvalidRowError(Exception):
    def __init__(self, row_num, reason):
        self.row_num = row_num
        self.reason = reason
        super().__init__(f"Row {row_num}: {reason}")
```

**Step-by-step skeleton** (fill in the logic yourself!):

```python
#!/usr/bin/env python3
"""CLI tool to clean a CSV file and output valid records as JSON."""

import argparse
import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path


# --- Data Model ---
@dataclass
class Record:
    """Represents a cleaned data record."""
    # TODO: define fields (name, age, city)
    # TODO: add @property is_valid
    pass


# --- Custom Exception ---
class InvalidRowError(Exception):
    """Raised when a CSV row fails validation."""
    # TODO: add row_num and reason attributes
    pass


# --- Core Logic ---
def clean_row(row: dict, row_num: int) -> Record:
    """Validate and clean a single CSV row. Raises InvalidRowError on failure."""
    # TODO: strip whitespace
    # TODO: validate name not empty
    # TODO: validate + convert age
    # TODO: return a Record
    pass


def process_csv(input_path: Path) -> tuple[list[Record], list[str]]:
    """Read CSV, clean each row, return (clean_records, errors)."""
    clean = []
    errors = []
    # TODO: open file, use DictReader
    # TODO: loop rows, try clean_row, catch InvalidRowError
    return clean, errors


def write_json(records: list[Record], output_path: Path) -> None:
    """Write cleaned records to a JSON file."""
    # TODO: use json.dump with asdict() to convert dataclasses
    pass


# --- CLI ---
def main():
    # TODO: set up argparse
    # TODO: set up logging based on --verbose
    # TODO: check input file exists (pathlib)
    # TODO: call process_csv
    # TODO: call write_json
    # TODO: print summary
    pass


if __name__ == "__main__":
    main()
```

---

### ✏️ Exercise 4 — Build It

1. First, create a `dirty_data.csv` test file:
```csv
name,age,city
Alice , 30 , Paris
,25,London
Bob,abc,Berlin
Charlie, 200, Tokyo
Diana,28,Rome
  Eve  , 22 ,  Madrid  
```
(Notice: extra spaces, empty name, non-numeric age, age out of range)

2. Fill in the skeleton above to create `clean_csv.py`
3. Run it: `python clean_csv.py dirty_data.csv output.json --verbose`
4. Expected output should log which rows were skipped and why, then save the valid ones as JSON

---

## 🗺️ Quick Reference Cheat Sheet

| Topic | Key Takeaway |
|-------|-------------|
| `with open()` | Always use `with` — auto-closes files |
| `csv.DictReader` | Reads CSV rows as dictionaries |
| `json.load/dump` | File ↔ Python. `loads/dumps` for strings |
| `pathlib.Path` | Modern paths: `Path("a") / "b"`, `.exists()`, `.read_text()` |
| `try/except` | Catch specific exceptions, not bare `except:` |
| Custom exceptions | Inherit from `Exception`, add useful attributes |
| `logging` | 5 levels: DEBUG → CRITICAL. Use instead of `print()` for debugging |
| `class` | Blueprint: `__init__` sets up data, methods define behavior |
| `@property` | Computed/validated attributes with attribute-like syntax |
| `@dataclass` | Auto-generates `__init__`, `__repr__`, `__eq__` |
| Dunder methods | `__str__`, `__repr__`, `__add__`, `__eq__`, `__lt__`, etc. |
| `argparse` | Professional CLI argument parsing |
