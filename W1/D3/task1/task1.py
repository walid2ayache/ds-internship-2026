import csv
import json
from pathlib import Path

# with open("sample.csv", "r") as file:
#     # reader = csv.reader(file)
#     # for line in file: print(line.strip())
#     # header= next(reader)
#     # for row in reader : print(row)
#     reader=csv.DictReader(file)
#     for row in reader: print(row["name"],row["age"])
# data = [
# {"name": "Alice", "age": 30},
# {"name": "Bob", "age": 25}]
# with open("output.csv", "w") as file:
#     writer=csv.DictWriter(file,fieldnames=["name","age"])
#     writer.writeheader()
#     writer.writerows(data)  


# # Reading JSON
# with open("data.json", "r") as f:
#     data = json.load(f)       # now data is a Python dict or list

# # Writing JSON
# with open("output.json", "w") as f:
#     json.dump(data, f, indent=2)  # indent=2 makes it pretty-printed



p=Path("data")/"files"/"input.csv"
# p.name          # "input.csv"
# p.stem          # "input"       (filename without extension)
# p.suffix        # ".csv"
# p.parent        # Path("data/files")
# p.exists()      # True/False
# p.is_file()     # True/False
# p.is_dir()      # True/False

# text=p.read_text()
# p.write_text("hello world")
for file in Path("data/files").glob("*.csv"):
    print(file)  

Path("output").mkdir(exist_ok=True)  
Path("a/b/c").mkdir(parents=True, exist_ok=True) 