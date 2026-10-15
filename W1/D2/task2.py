"""
=============================================================================
WEEK 1 - DAY 2: TASK 2 - FUNCTIONAL PYTHON, COMPREHENSIONS & STRING MASTERY
=============================================================================
Topics Covered:
  1. List Comprehensions (Transform, Filter, If-Else, Nested/Flattening)
  2. Dictionary Comprehensions (Transform, Filter, Inverting, Mapping)
  3. Set Comprehensions (Deduplication & Extraction)
  4. zip() (Parallel Iteration, Dict Creation, Unzipping, strict=True)
  5. enumerate() (Indexing Without range(len()), Custom Start, Label Encoding)
  6. map() (Lazy Evaluation, Multi-Iterable Mapping, map vs. Comprehension)
  7. filter() (Filtering with Predicates, Truthiness Filter, filter vs. Comprehension)
  8. Strings & Essential String Methods (strip, replace, startswith, find, count)
  9. Advanced f-strings (Float Precision, Commas, Percentages, Alignment, {var=})
 10. Slicing Mechanics ([start:stop:step], Negative Indices, Reversal)
 11. split() and join() (Tokenization, Delimiters, rsplit, Rebuilding Strings)
 12. Real-World Data Science Capstone (End-to-End Log & Feature Pipeline)
=============================================================================
"""


# =============================================================================
# 1. LIST COMPREHENSIONS
# =============================================================================
# Syntax:
#   Basic:            [expression for item in iterable]
#   With Filter:      [expression for item in iterable if condition]
#   With If-Else:     [expr_true if condition else expr_false for item in iterable]
#   Nested (Flatten): [item for sublist in matrix for item in sublist]

def demo_list_comprehensions():
    print("\n--- 1. List Comprehensions ---")
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # 1. Basic Transformation: Squares of all numbers
    squares = [x ** 2 for x in numbers]
    print("Squares:               ", squares)
    
    # 2. Filtering: Only even numbers
    evens = [x for x in numbers if x % 2 == 0]
    print("Evens only:            ", evens)
    
    # 3. If-Else (Ternary): "even" or "odd" labels
    # Note: When using if-else, it goes BEFORE the 'for' keyword!
    labels = ["EVEN" if x % 2 == 0 else "ODD" for x in numbers]
    print("Even/Odd labels:       ", labels)
    
    # 4. Nested Comprehension: Flattening a 2D matrix
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    # Read as: for row in matrix: for val in row: yield val
    flattened = [val for row in matrix for val in row]
    print("Flattened 2D matrix:   ", flattened)
    
    # 5. Conditional Nested: Filter elements within nested structure
    flattened_evens = [val for row in matrix for val in row if val % 2 == 0]
    print("Flattened evens only:  ", flattened_evens)


# =============================================================================
# 2. DICTIONARY COMPREHENSIONS
# =============================================================================
# Syntax:
#   Basic:     {key_expr: value_expr for item in iterable}
#   Filtered:  {key_expr: value_expr for k, v in dict.items() if condition}

def demo_dict_comprehensions():
    print("\n--- 2. Dictionary Comprehensions ---")
    
    features = ["age", "salary", "experience", "credit_score"]
    
    # 1. Basic: Create a mapping of feature -> length of name
    name_lengths = {feat: len(feat) for feat in features}
    print("Feature name lengths:      ", name_lengths)
    
    # 2. Filtering a Dictionary: Keep only high-performing models
    model_scores = {
        "LogisticRegression": 0.78,
        "DecisionTree": 0.72,
        "RandomForest": 0.91,
        "XGBoost": 0.94,
        "SVM": 0.83
    }
    top_models = {name: score for name, score in model_scores.items() if score >= 0.85}
    print("Top models (score >= 0.85):", top_models)
    
    # 3. Inverting a Dictionary: Swap keys and values (useful for ID lookups)
    id_to_class = {0: "setosa", 1: "versicolor", 2: "virginica"}
    class_to_id = {v: k for k, v in id_to_class.items()}
    print("Inverted class mapping:    ", class_to_id)
    
    # 4. Data Normalization: Scale scores to percentages
    normalized_scores = {name: f"{score * 100:.1f}%" for name, score in model_scores.items()}
    print("Normalized model scores:   ", normalized_scores)


# =============================================================================
# 3. SET COMPREHENSIONS
# =============================================================================
# Syntax:
#   {expression for item in iterable if condition}
# Key Property: Sets automatically remove duplicate elements.

def demo_set_comprehensions():
    print("\n--- 3. Set Comprehensions ---")
    
    file_names = [
        "data_2026.csv", "model.pkl", "README.md", "train.csv",
        "labels.CSV", "pipeline.pkl", "test.Csv", "config.yaml"
    ]
    
    # Extract unique, normalized file extensions
    extensions = {file.split(".")[-1].lower() for file in file_names}
    print("Raw files:          ", file_names)
    print("Unique extensions:  ", extensions)
    
    # Extract unique lengths of words
    sentences = "Data science is fun and data analysis is powerful"
    unique_word_lengths = {len(word) for word in sentences.lower().split()}
    print("Unique word lengths:", sorted(unique_word_lengths))


# =============================================================================
# 4. THE zip() FUNCTION
# =============================================================================
# Syntax:
#   zip(iter1, iter2, ...)
# Pairs elements by index into tuples. Stops at shortest iterable unless strict=True.

def demo_zip():
    print("\n--- 4. The zip() Function ---")
    
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    sample_values = [5.1, 3.5, 1.4, 0.2]
    importance    = [0.25, 0.10, 0.45, 0.20]
    
    # 1. Parallel Iteration
    print("[Parallel Iteration with zip]")
    for name, val, imp in zip(feature_names, sample_values, importance):
        print(f"  Feature: {name:<12} | Value: {val:<4} | Importance: {imp:.2f}")
        
    # 2. Creating a Dictionary directly from two lists
    feature_dict = dict(zip(feature_names, sample_values))
    print("\nDict created with dict(zip()):", feature_dict)
    
    # 3. Unzipping paired data using the * (splat) operator
    pairs = [("Alice", 95), ("Bob", 88), ("Charlie", 92), ("Dana", 79)]
    names, scores = zip(*pairs)
    print("\nUnzipping pairs:")
    print("  Names: ", names)
    print("  Scores:", scores)
    
    # 4. Python 3.10+ strict=True parameter (Prevents silent data loss on mismatched lengths)
    try:
        short_list = [1, 2]
        long_list = ["a", "b", "c", "d"]
        print("\nAttempting zip with strict=True on uneven lists:")
        list(zip(short_list, long_list, strict=True))
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")


# =============================================================================
# 5. THE enumerate() FUNCTION
# =============================================================================
# Syntax:
#   enumerate(iterable, start=0)
# Returns an iterator yielding (index, item) pairs cleanly without range(len()).

def demo_enumerate():
    print("\n--- 5. The enumerate() Function ---")
    
    classes = ["Cat", "Dog", "Bird", "Fish"]
    
    # 1. The Pythonic Way vs Anti-pattern
    # ❌ Anti-pattern: for i in range(len(classes)): print(i, classes[i])
    # ✅ Pythonic way:
    print("[Clean iteration with enumerate]")
    for index, label in enumerate(classes):
        print(f"  Index {index} -> {label}")
        
    # 2. Custom Starting Index (e.g. human-friendly 1-based ranking)
    print("\n[Ranking with start=1]")
    top_models = ["XGBoost", "LightGBM", "CatBoost"]
    for rank, model in enumerate(top_models, start=1):
        print(f"  Rank #{rank}: {model}")
        
    # 3. Building an integer Label Encoder (Common in Machine Learning)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for idx, label in enumerate(classes)}
    print("\nLabel-to-Index Encoder:", label_to_idx)
    print("Index-to-Label Decoder:", idx_to_label)


# =============================================================================
# 6. THE map() FUNCTION
# =============================================================================
# Syntax:
#   map(function, iterable, ...)
# Applies a function to every item in an iterable lazily (yields on demand).

def demo_map():
    print("\n--- 6. The map() Function ---")
    
    temperatures_celsius = [0.0, 15.5, 20.0, 25.0, 32.5, 100.0]
    
    # Helper transformation
    def c_to_f(c):
        return round((c * 9 / 5) + 32, 2)
    
    # 1. map() returns a lazy iterator
    map_obj = map(c_to_f, temperatures_celsius)
    print(f"map object is lazy: {map_obj} (Type: {type(map_obj)})")
    
    # Convert to list to consume the values
    temperatures_fahrenheit = list(map_obj)
    print("Converted to Fahrenheit:", temperatures_fahrenheit)
    
    # 2. Mapping with lambda functions
    raw_strings = ["   Age: 25 ", " Salary: 50000\n", "\tCity: Algiers  "]
    cleaned_strings = list(map(lambda s: s.strip().lower(), raw_strings))
    print("Cleaned strings:        ", cleaned_strings)
    
    # 3. Multi-Iterable map: map takes corresponding elements from each iterable
    bases = [2, 3, 4, 5]
    exponents = [1, 2, 3, 4]
    powers = list(map(pow, bases, exponents))  # 2^1, 3^2, 4^3, 5^4
    print("Multi-iterable map (pow):", powers)


# =============================================================================
# 7. THE filter() FUNCTION
# =============================================================================
# Syntax:
#   filter(predicate_function, iterable)
# Yields only items for which predicate_function(item) evaluates to True.

def demo_filter():
    print("\n--- 7. The filter() Function ---")
    
    records = [
        {"id": 1, "name": "Alice", "score": 88, "active": True},
        {"id": 2, "name": "Bob",   "score": 45, "active": False},
        {"id": 3, "name": "Carol", "score": 92, "active": True},
        {"id": 4, "name": "Dave",  "score": 58, "active": True},
        {"id": 5, "name": "Eve",   "score": 77, "active": False},
    ]
    
    # 1. Filtering with custom function
    def is_passing(student):
        return student["score"] >= 60 and student["active"]
    
    passing_students = list(filter(is_passing, records))
    print(f"Passing active students ({len(passing_students)} found):")
    for s in passing_students:
        print(f"  - {s['name']} (Score: {s['score']})")
        
    # 2. Truthiness Filter using filter(None, iterable)
    # When function is None, filter keeps only TRUTHY elements (drops None, 0, "", [], False)
    dirty_data = ["Algiers", "", None, "Oran", 0, "Constantine", [], False, "Annaba"]
    clean_cities = list(filter(None, dirty_data))
    print("\nRaw dirty data:            ", dirty_data)
    print("filter(None, ...) cleaned: ", clean_cities)


# =============================================================================
# 8. STRINGS & ESSENTIAL STRING METHODS
# =============================================================================
def demo_string_methods():
    print("\n--- 8. Strings & Essential String Methods ---")
    
    raw_text = "   >>> Data Science & Machine Learning Internship 2026! <<<   \n"
    
    # 1. Stripping whitespace and specific characters
    stripped_ws = raw_text.strip()
    stripped_custom = stripped_ws.strip(" ><!")
    print(f"Raw repr          : {repr(raw_text)}")
    print(f"strip()           : {repr(stripped_ws)}")
    print(f"strip(' ><!')     : {repr(stripped_custom)}")
    
    # 2. Case normalization
    query = "PyThOn PrOgRaMmInG"
    print(f"\nOriginal          : {query}")
    print(f"lower()           : {query.lower()}")
    print(f"upper()           : {query.upper()}")
    print(f"title()           : {query.title()}")
    
    # 3. Substring inspection & replacement
    filename = "customer_churn_model_v2.tar.gz"
    print(f"\nChecking: {filename}")
    print(f"startswith('customer'): {filename.startswith('customer')}")
    print(f"endswith('.gz')       : {filename.endswith('.gz')}")
    print(f"find('model')         : index {filename.find('model')}")
    print(f"count('_')            : {filename.count('_')} underscores")
    print(f"replace('v2', 'final'): {filename.replace('v2', 'final')}")


# =============================================================================
# 9. ADVANCED F-STRINGS
# =============================================================================
def demo_f_strings():
    print("\n--- 9. Advanced f-strings ---")
    
    model_name = "GradientBoostingClassifier"
    accuracy = 0.945821
    sample_count = 1250000
    loss = 0.000314159
    
    # 1. Number Formatting: Decimals, Percentage, Commas, Scientific Notation
    print(f"Float precision (.2f) : {accuracy:.2f}")
    print(f"Percentage (.1%)      : {accuracy:.1%}")
    print(f"Thousands comma (:,)  : {sample_count:,} samples")
    print(f"Scientific (.2e)      : {loss:.2e}")
    
    # 2. Text Alignment & Column Padding
    #   < left-align | > right-align | ^ center-align
    print("\n[Aligned Table Output]")
    print(f"{'Model':<28} | {'Accuracy':>10} | {'Status':^12}")
    print("-" * 56)
    print(f"{model_name:<28} | {accuracy * 100:>9.2f}% | {'PASSED':^12}")
    print(f"{'LogisticRegression':<28} | {0.8123 * 100:>9.2f}% | {'WARNING':^12}")
    
    # 3. Self-Documenting Debugging Expressions with = (Python 3.8+)
    x = 42
    y = 7
    print("\n[Debugging with {var=}]")
    print(f"{x=}, {y=}, {x*y=}, {x/y=:.2f}")


# =============================================================================
# 10. SLICING MECHANICS
# =============================================================================
# Syntax:
#   sequence[start : stop : step]
#   - start: inclusive (defaults to 0)
#   - stop:  exclusive (defaults to len)
#   - step:  stride / direction (defaults to 1)

def demo_slicing():
    print("\n--- 10. Slicing Mechanics ---")
    
    text = "DataScience2026"
    print(f"Original text: '{text}' (Length: {len(text)})")
    
    # 1. Basic Slices
    print(f"text[0:4]       (First 4 chars) : '{text[0:4]}'")
    print(f"text[:4]        (Omitted start) : '{text[:4]}'")
    print(f"text[4:11]      ('Science')     : '{text[4:11]}'")
    print(f"text[11:]       (From 11 to end): '{text[11:]}'")
    
    # 2. Negative Indices (Counting from right)
    print(f"text[-4:]       (Last 4 chars)  : '{text[-4:]}'")
    print(f"text[:-4]       (Drop last 4)   : '{text[:-4]}'")
    print(f"text[-8:-4]     ('ence')        : '{text[-8:-4]}'")
    
    # 3. Strides (Step parameter)
    print(f"text[::2]       (Every 2nd char): '{text[::2]}'")
    
    # 4. Reversing Sequences with Step -1 (Idiomatic Python trick)
    reversed_text = text[::-1]
    print(f"text[::-1]      (Reversed text) : '{reversed_text}'")
    
    # 5. List Slicing & Shallow Copy
    raw_readings = [10.2, 11.5, 12.1, 14.8, 15.0, 16.3, 18.9]
    top_3 = raw_readings[-3:]                 # last 3 items
    copy_list = raw_readings[:]               # clean shallow copy
    downsampled = raw_readings[::2]           # downsample by 2x
    print(f"\nList slicing:")
    print(f"  Top 3 highest : {top_3}")
    print(f"  Downsampled 2x: {downsampled}")


# =============================================================================
# 11. split() AND join()
# =============================================================================
def demo_split_and_join():
    print("\n--- 11. split() and join() ---")
    
    # 1. Default split() splits on ANY consecutive whitespace (spaces, tabs, newlines)
    messy_line = "  batch_size    64   lr  0.001   optimizer  adam   \n"
    tokens = messy_line.split()
    print("Default whitespace split():", tokens)
    
    # 2. Delimiter split()
    csv_row = "101,Walid,Algiers,Data Scientist,95000"
    fields = csv_row.split(",")
    print("CSV comma split()         :", fields)
    
    # 3. maxsplit and rsplit: Limit the number of splits
    log_entry = "2026-03-01T10:45:12 [ERROR] Database connection timed out after 3000ms"
    timestamp, log_level, message = log_entry.split(" ", 2)
    print("\nSplitting with maxsplit=2:")
    print(f"  timestamp: {timestamp}")
    print(f"  log_level: {log_level}")
    print(f"  message  : {message}")
    
    # 4. join(): The Pythonic string builder
    # Syntax: 'delimiter'.join(iterable_of_strings)
    tech_stack = ["Python", "Pandas", "Scikit-Learn", "PyTorch", "Docker"]
    
    joined_csv = ",".join(tech_stack)
    joined_bullets = "\n  * ".join(["Core Skills:"] + tech_stack)
    
    print("\njoin() demonstrations:")
    print("  CSV format :", joined_csv)
    print("  Bullet list:\n  *", joined_bullets[len("Core Skills:\n  * "):])
    
    # 💡 Performance Tip: NEVER concatenate strings in a loop with `+`!
    # ❌ Bad:  s = ""; for word in words: s += word
    # ✅ Good: s = "".join(words)  (O(N) instead of O(N^2))


# =============================================================================
# 12. REAL-WORLD DATA SCIENCE CAPSTONE
# =============================================================================
# Combines Comprehensions, zip, enumerate, map, filter, strings, f-strings,
# slicing, and join/split into an End-to-End Server Log Parser & Feature Pipeline.

def demo_capstone():
    print("\n--- 12. Real-World Data Science Capstone ---")
    
    # Raw unstructured web server access log lines
    raw_logs = [
        "2026-03-01T12:00:01Z | USER:101 | IP:192.168.1.15 | PATH:/api/v1/predict | STATUS:200 | LATENCY:45ms",
        "2026-03-01T12:00:02Z | USER:102 | IP:10.0.0.84     | PATH:/api/v1/train   | STATUS:500 | LATENCY:1250ms",
        "2026-03-01T12:00:03Z | USER:101 | IP:192.168.1.15 | PATH:/api/v1/health  | STATUS:200 | LATENCY:12ms",
        "# MALFORMED COMMENT LINE - IGNORE",
        "2026-03-01T12:00:04Z | USER:103 | IP:172.16.0.4    | PATH:/api/v1/predict | STATUS:200 | LATENCY:52ms",
        "2026-03-01T12:00:05Z | USER:104 | IP:10.0.0.99     | PATH:/login          | STATUS:404 | LATENCY:21ms",
        "",  # Empty line
    ]
    
    print("[Step 1: Filtering non-log and malformed lines]")
    # Use filter() to drop empty lines and comments using string methods
    clean_raw_logs = list(filter(
        lambda line: line.strip() != "" and not line.startswith("#"),
        raw_logs
    ))
    print(f"  Valid log lines: {len(clean_raw_logs)} of {len(raw_logs)}")
    
    print("\n[Step 2: Parsing, Slicing & Dict Comprehension]")
    def parse_log_line(line):
        # Split on pipeline delimiter and strip each segment
        parts = [p.strip() for p in line.split("|")]
        
        # Slicing: timestamp is first item
        timestamp = parts[0]
        # Dict comprehension for key:value pairs (USER:101, IP:..., etc.)
        record = {
            item.split(":")[0].strip().lower(): item.split(":", 1)[1].strip()
            for item in parts[1:]
        }
        
        # String slice & type conversion: "45ms" -> 45
        latency_str = record.get("latency", "0ms")
        record["latency_ms"] = int(latency_str[:-2]) if latency_str.endswith("ms") else int(latency_str)
        record["status"] = int(record.get("status", 0))
        record["timestamp"] = timestamp
        return record

    parsed_records = list(map(parse_log_line, clean_raw_logs))
    
    print("\n[Step 3: Set Comprehension to extract unique endpoints]")
    unique_paths = {rec["path"] for rec in parsed_records}
    print(f"  Unique paths accessed: {', '.join(sorted(unique_paths))}")
    
    print("\n[Step 4: zip() and enumerate() to generate an audit report]")
    # Zip together extracted fields
    users = [r["user"] for r in parsed_records]
    paths = [r["path"] for r in parsed_records]
    latencies = [r["latency_ms"] for r in parsed_records]
    statuses = [r["status"] for r in parsed_records]
    
    print("\n  " + "=" * 65)
    print(f"  {'#':<3} | {'User':<6} | {'Status':<6} | {'Latency':>10} | {'Endpoint':<25}")
    print("  " + "-" * 65)
    
    for idx, (user, path, lat, stat) in enumerate(zip(users, paths, latencies, statuses), start=1):
        # f-string formatting with conditional tag and alignment
        tag = "OK" if stat == 200 else f"ERR:{stat}"
        print(f"  {idx:<3} | {user:<6} | {tag:<6} | {lat:>8}ms | {path:<25}")
    print("  " + "=" * 65)
    
    print("\n[Step 5: Exporting as Clean CSV using join()]")
    header = "idx,user,status,latency_ms,endpoint"
    rows = [
        f"{idx},{user},{stat},{lat},{path}"
        for idx, (user, path, lat, stat) in enumerate(zip(users, paths, latencies, statuses), start=1)
    ]
    csv_payload = "\n".join([header] + rows)
    print(csv_payload)


# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" PYTHON MASTERY: COMPREHENSIONS, ZIP, MAP, FILTER, STRINGS & SLICING")
    print("=" * 70)
    
    demo_list_comprehensions()
    demo_dict_comprehensions()
    demo_set_comprehensions()
    demo_zip()
    demo_enumerate()
    demo_map()
    demo_filter()
    demo_string_methods()
    demo_f_strings()
    demo_slicing()
    demo_split_and_join()
    demo_capstone()
    
    print("\n" + "=" * 70)
    print(" ALL TASK 2 DEMOS EXECUTED SUCCESSFULLY! ")
    print("=" * 70)
