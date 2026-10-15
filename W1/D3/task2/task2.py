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



class ValidationError(Exception):
    pass
class AgeArror(ValidationError):
   def __init__(self,age,message="age must be btw 18-99"):
    self.age=age
    self.message=message
    super().__init__(self.message)

def validate_age(age):
    if not 18<=age<=99:
        raise AgeArror(age
            )
    return age 


try: validate_age(16)
except AgeArror as e:
    print(e)
else:
    print("age is valid")


import logging

# Basic setup
# logging.basicConfig(
#     level=logging.DEBUG,              # show all messages DEBUG and above
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     # Optional: log to a file
#     # filename="app.log",
# )

# # 5 severity levels (low → high):
# logging.debug("Variable x = 42")         # for detailed debugging
# logging.info("Processing started")       # normal operation updates
# logging.warning("Disk space low")        # something unexpected but not fatal
# logging.error("Failed to read file")     # something failed
# logging.critical("Database is down!")     # system is broken

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
try:
    data = int("not_a_number")
except ValueError:
    logging.error("Could not convert to int")


