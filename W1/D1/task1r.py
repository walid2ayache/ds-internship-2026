import csv 

import pandas as pd

class dataset:
    def __init__(self, data_input):
        """1. __init__: Initializes the dataset by loading or storing data."""
        if isinstance(data_input, str):
            # Load from CSV file if string path is provided
            self.data = pd.read_csv(data_input)
        elif isinstance(data_input, list):
            # Convert list of dicts to DataFrame
            self.data = pd.DataFrame(data_input)
        elif isinstance(data_input, pd.DataFrame):
            self.data = data_input
        else:
            raise TypeError("Unsupported data format provided.")

    def __len__(self):
        """2. __len__: Returns the total number of records in the dataset."""
        return len(self.data)

    def describe(self):
        """3. describe: Prints and returns summary statistics of the dataset."""
        print(f"Dataset Shape : {self.data.shape[0]} rows x {self.data.shape[1]} columns")
        print(f"Column Names  : {list(self.data.columns)}")
        print("\nStatistical Summary:")
        return self.data.describe()

my_ds = dataset('username-password-recovery-code.csv')
total_rows = len(my_ds)
print(f"Number of rows: {total_rows}")
summary = my_ds.describe()
print(summary)
print("\nFirst 2 rows:")
print(my_ds.data.head(2))
print("\nIdentifiers column:")
print(my_ds.data['Identifier'])


# Create a dataset from a list of dicts:
student_ds = dataset([
    {"name": "Alice", "grade": 95},
    {"name": "Bob", "grade": 82}
])

print("Student count:", len(student_ds))
student_ds.describe()
