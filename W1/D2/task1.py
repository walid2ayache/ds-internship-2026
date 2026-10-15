"""
=============================================================================
WEEK 1 - DAY 2: TASK 1 - COMPREHENSIVE GUIDE TO PYTHON FUNCTIONS
=============================================================================
Topics Covered:
  1. Positional & Keyword Arguments
  2. Default Arguments & The Mutable Default Trap
  3. *args (Variable Positional Arguments & Unpacking)
  4. **kwargs (Variable Keyword Arguments & Unpacking)
  5. The Golden Order of Function Parameters
  6. Return Values (Tuples, Early Returns, Implicit None)
  7. Scope & The LEGB Rule (Local, Enclosing, Global, Built-in)
  8. Real-World Data Science Capstone Example
=============================================================================
"""


# =============================================================================
# 1. POSITIONAL VS. KEYWORD ARGUMENTS
# =============================================================================
def calculate_bmi(weight_kg, height_m):
    """
    Calculates Body Mass Index (BMI).
    weight_kg and height_m are positional parameters.
    """
    return round(weight_kg / (height_m ** 2), 2)


def demo_positional_and_keyword():
    print("\n--- 1. Positional vs Keyword Arguments ---")
    
    # Called positionally: argument order MUST match parameter order
    bmi_1 = calculate_bmi(70, 1.75)
    print(f"Positional call -> calculate_bmi(70, 1.75): {bmi_1}")
    
    # Called by keyword: order does NOT matter because names are explicit
    bmi_2 = calculate_bmi(height_m=1.75, weight_kg=70)
    print(f"Keyword call    -> calculate_bmi(height_m=1.75, weight_kg=70): {bmi_2}")
    
    # Mixing both: Positional arguments MUST come before keyword arguments!
    bmi_3 = calculate_bmi(70, height_m=1.75)
    print(f"Mixed call      -> calculate_bmi(70, height_m=1.75): {bmi_3}")
    
    # SyntaxError rule: calculate_bmi(weight_kg=70, 1.75) is INVALID in Python!


# =============================================================================
# 2. DEFAULT ARGUMENTS & THE DANGEROUS MUTABLE DEFAULT PITFALL
# =============================================================================
def greet(name, greeting="Hello", punctuation="!"):
    """
    Parameters with default values make arguments optional.
    Default parameters must always follow non-default parameters.
    """
    return f"{greeting}, {name}{punctuation}"


# ⚠️ THE CLASSIC PITFALL: Never use mutable objects (like lists or dicts) as defaults!
def buggy_append(item, target_list=[]):
    """
    BUG: The list `[]` is created ONCE when the function is defined,
    NOT each time the function is called! All calls share the same list.
    """
    target_list.append(item)
    return target_list


# ✅ THE IDIOMATIC SOLUTION: Use None as the sentinel default
def safe_append(item, target_list=None):
    """
    SAFE: A fresh list is instantiated inside the function on every call.
    """
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list


def demo_defaults_and_pitfalls():
    print("\n--- 2. Default Arguments & Mutable Default Pitfall ---")
    
    # Using defaults
    print(greet("Alice"))                        # uses "Hello" and "!"
    print(greet("Bob", greeting="Welcome"))       # overrides greeting
    print(greet("Charlie", "Hi", "..."))          # overrides both
    
    # Demonstrating the bug with mutable default
    print("\n[Buggy Mutable Default Demonstration]")
    list_a = buggy_append(1)
    print(f"Call 1 (added 1): {list_a}")
    list_b = buggy_append(2)
    print(f"Call 2 (added 2): {list_b}  <-- Notice: 1 is still here! Shared state bug!")
    
    # Demonstrating the fix
    print("\n[Safe Default Demonstration]")
    safe_a = safe_append(1)
    print(f"Call 1 (added 1): {safe_a}")
    safe_b = safe_append(2)
    print(f"Call 2 (added 2): {safe_b}  <-- Each call gets a brand-new list!")


# =============================================================================
# 3. *args: VARIABLE POSITIONAL ARGUMENTS
# =============================================================================
def calculate_mean(*numbers):
    """
    *args gathers any number of positional arguments into a TUPLE named `numbers`.
    """
    print(f"  Received `numbers` as type: {type(numbers)}, contents: {numbers}")
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def demo_args():
    print("\n--- 3. *args (Tuple Packing & Unpacking) ---")
    
    # Packing: Pass any number of arguments directly
    print("Mean of 2 numbers:", calculate_mean(10, 20))
    print("Mean of 5 numbers:", calculate_mean(4, 8, 15, 16, 23, 42))
    print("Mean of 0 numbers:", calculate_mean())
    
    # Unpacking: If you already have a list/tuple, prefix it with `*` to unpack it
    dataset = [85, 90, 78, 92, 88]
    print(f"Unpacking dataset {dataset} using *dataset:")
    mean_score = calculate_mean(*dataset)  # equivalent to calculate_mean(85, 90, 78, 92, 88)
    print("Mean score:", mean_score)


# =============================================================================
# 4. **kwargs: VARIABLE KEYWORD ARGUMENTS
# =============================================================================
def build_model_summary(model_name, **metrics):
    """
    **kwargs gathers any arbitrary keyword arguments into a DICTIONARY named `metrics`.
    """
    print(f"  Received `metrics` as type: {type(metrics)}, contents: {metrics}")
    print(f"Model: {model_name}")
    for metric_name, score in metrics.items():
        print(f"  - {metric_name.upper()}: {score:.4f}")


def demo_kwargs():
    print("\n--- 4. **kwargs (Dictionary Packing & Unpacking) ---")
    
    # Packing: Pass arbitrary key=value pairs
    build_model_summary("Random Forest", accuracy=0.945, f1=0.932, roc_auc=0.981)
    
    # Unpacking: If you have an existing dictionary, prefix it with `**` to unpack it
    evaluation_dict = {
        "precision": 0.892,
        "recall": 0.915,
        "loss": 0.124
    }
    print("\nUnpacking evaluation_dict using **evaluation_dict:")
    build_model_summary("Gradient Boosting", **evaluation_dict)


# =============================================================================
# 5. THE GOLDEN ORDER OF FUNCTION PARAMETERS
# =============================================================================
# Python enforces strict parameter ordering:
# 1. Standard positional parameters
# 2. Default parameters
# 3. *args
# 4. Keyword-only parameters (parameters placed AFTER *args MUST be passed by keyword)
# 5. **kwargs
def train_pipeline(
    model_name,                   # 1. Positional parameter
    epochs=10,                    # 2. Default parameter
    *loss_history,                # 3. *args (consumes extra positional values)
    learning_rate=0.001,          # 4. Keyword-only parameter (with default)
    verbose=False,                # 4. Keyword-only parameter
    **hyperparameters             # 5. **kwargs (consumes extra keyword arguments)
):
    print(f"\n[train_pipeline Configuration]")
    print(f"  model_name     : {model_name}")
    print(f"  epochs         : {epochs}")
    print(f"  loss_history   : {loss_history}")
    print(f"  learning_rate  : {learning_rate}")
    print(f"  verbose        : {verbose}")
    print(f"  hyperparameters: {hyperparameters}")


def demo_parameter_order():
    print("\n--- 5. The Golden Order of Parameters ---")
    train_pipeline(
        "ResNet-50",             # model_name (positional)
        25,                      # epochs (positional override of default)
        0.85, 0.42, 0.21, 0.11,  # *loss_history (captured into tuple)
        learning_rate=0.0005,    # keyword-only (MUST specify name)
        verbose=True,            # keyword-only
        batch_size=64,           # **hyperparameters (into dict)
        optimizer="AdamW"        # **hyperparameters (into dict)
    )


# =============================================================================
# 6. RETURN VALUES: MULTIPLE RETURNS, EARLY RETURN & IMPLICIT NONE
# =============================================================================
def compute_descriptive_stats(numbers):
    """
    Demonstrates:
      1. Early return (Guard Clause)
      2. Returning multiple values packed as a tuple
    """
    # Guard clause: Exit early if input is invalid
    if not numbers:
        return None  # explicit return of None
    
    n = len(numbers)
    mean_val = sum(numbers) / n
    variance = sum((x - mean_val) ** 2 for x in numbers) / (n - 1 if n > 1 else 1)
    std_dev = variance ** 0.5
    min_val, max_val = min(numbers), max(numbers)
    
    # In Python, separating values with commas automatically returns a tuple!
    return mean_val, std_dev, min_val, max_val


def silent_logger(message):
    """
    A function without an explicit `return` statement implicitly returns `None`.
    """
    print(f"[LOG]: {message}")
    # implicit: return None


def demo_return_values():
    print("\n--- 6. Return Statements & Unpacking ---")
    data = [12, 15, 18, 20, 22, 25, 29]
    
    # 1. Multiple Return Values & Tuple Unpacking
    avg, std, minimum, maximum = compute_descriptive_stats(data)
    print(f"Dataset: {data}")
    print(f"Unpacked returns -> Mean: {avg:.2f}, Std: {std:.2f}, Min: {minimum}, Max: {maximum}")
    
    # 2. Early Return handling empty data
    empty_result = compute_descriptive_stats([])
    print(f"Empty data result (guard clause returned): {empty_result}")
    
    # 3. Implicit None Return
    result = silent_logger("Processing batch 1...")
    print(f"Return value of silent_logger: {result} (type: {type(result)})")


# =============================================================================
# 7. SCOPE & THE LEGB RULE (Local, Enclosing, Global, Built-in)
# =============================================================================
# LEGB defines the lookup hierarchy Python follows to resolve any variable name:
#   L: Local      - Names defined inside the current function
#   E: Enclosing  - Names in the outer function of a nested function (closures)
#   G: Global     - Names defined at the module/top level of this file
#   B: Built-in   - Predefined Python names (e.g., len, range, print, sum)

# GLOBAL SCOPE variable
GLOBAL_CONFIG = "production_v1"
counter = 0


def demo_scope_and_legb():
    print("\n--- 7. Variable Scope & The LEGB Rule ---")
    
    # 1. Reading a global variable (Always permitted)
    print(f"Reading global variable from inside function: {GLOBAL_CONFIG}")
    
    # 2. Modifying a global variable requires the `global` keyword
    global counter
    counter += 1
    print(f"Updated global counter: {counter}")
    
    # 3. Enclosing Scope & `nonlocal` in Nested Functions
    def outer_accumulator(initial_value=0):
        total = initial_value  # Enclosing scope variable for inner_add
        
        def inner_add(amount):
            nonlocal total     # Allows modifying variable in the immediate enclosing scope
            total += amount
            return total
        
        return inner_add
    
    # A closure is created: inner_add remembers `total`
    acc = outer_accumulator(100)
    print("\n[Closure & nonlocal Demonstration]")
    print("Add 25:", acc(25))   # total becomes 125
    print("Add 50:", acc(50))   # total becomes 175
    print("Add 10:", acc(10))   # total becomes 185
    
    # 4. Shadowing Built-ins (Anti-pattern to avoid!)
    # If you define `sum = 10`, you shadow Python's built-in sum() function in Local scope!
    print(f"Built-in len() function identity: {len}")


# =============================================================================
# 8. REAL-WORLD DATA SCIENCE PIPELINE (ALL CONCEPTS IN ACTION)
# =============================================================================
def preprocess_features(
    dataset,                           # Positional: list of dict records
    *target_columns,                   # *args: which numeric columns to standardize
    handle_missing="mean",             # Default: how to impute missing values
    scale_type="standard",             # Keyword-only default: 'standard' or 'minmax'
    **metadata                         # **kwargs: extra tracking info (e.g. author, run_id)
):
    """
    Comprehensive Data Science function demonstrating all concepts:
      - Positional dataset
      - *args for variable number of columns to process
      - Default & keyword-only arguments for preprocessing options
      - **kwargs for arbitrary metadata tracking
      - Returning multiple processed artifacts (cleaned_data, summary_report)
    """
    print(f"\nProcessing Run ID: {metadata.get('run_id', 'N/A')} by {metadata.get('author', 'Anonymous')}")
    print(f"Target columns to scale: {target_columns}")
    
    cleaned_data = [row.copy() for row in dataset]
    summary_report = {}
    
    for col in target_columns:
        # Extract valid values (ignoring None)
        valid_vals = [row[col] for row in cleaned_data if row.get(col) is not None]
        
        if not valid_vals:
            continue
            
        col_mean = sum(valid_vals) / len(valid_vals)
        col_std = (sum((x - col_mean)**2 for x in valid_vals) / len(valid_vals)) ** 0.5 or 1.0
        
        # Impute missing values
        for row in cleaned_data:
            if row.get(col) is None:
                row[col] = col_mean if handle_missing == "mean" else 0.0
                
        # Scale values
        for row in cleaned_data:
            if scale_type == "standard":
                row[f"{col}_scaled"] = round((row[col] - col_mean) / col_std, 3)
            elif scale_type == "minmax":
                min_v, max_v = min(valid_vals), max(valid_vals)
                rng = (max_v - min_v) or 1.0
                row[f"{col}_scaled"] = round((row[col] - min_v) / rng, 3)
                
        summary_report[col] = {
            "mean": round(col_mean, 2),
            "std": round(col_std, 2),
            "scale_type": scale_type
        }
        
    return cleaned_data, summary_report


def demo_capstone():
    print("\n--- 8. Real-World Data Science Capstone ---")
    
    raw_data = [
        {"age": 25, "income": 50000, "city": "Algiers"},
        {"age": 30, "income": 65000, "city": "Oran"},
        {"age": None, "income": 80000, "city": "Constantine"},
        {"age": 45, "income": None,  "city": "Annaba"},
    ]
    
    # We pass:
    # 1. raw_data (positional)
    # 2. "age", "income" (*target_columns gathered into tuple)
    # 3. scale_type="standard" (keyword argument)
    # 4. run_id=101, author="Intern" (**metadata gathered into dict)
    processed, stats = preprocess_features(
        raw_data,
        "age", "income",
        scale_type="standard",
        run_id="EXP-2026-001",
        author="Walid"
    )
    
    print("\nProcessed Data Rows:")
    for row in processed:
        print(" ", row)
        
    print("\nScaling Stats Summary:")
    for col, stat in stats.items():
        print(f"  {col}: {stat}")


# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" PYTHON FUNCTIONS MASTERCLASS: ARGS, KWARGS, DEFAULTS, RETURN, SCOPE")
    print("=" * 70)
    
    demo_positional_and_keyword()
    demo_defaults_and_pitfalls()
    demo_args()
    demo_kwargs()
    demo_parameter_order()
    demo_return_values()
    demo_scope_and_legb()
    demo_capstone()
    
    print("\n" + "=" * 70)
    print(" ALL DEMOS EXECUTED SUCCESSFULLY! ")
    print("=" * 70)
