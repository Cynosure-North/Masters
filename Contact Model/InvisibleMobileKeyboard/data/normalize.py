import csv
import random
from pathlib import Path

INPUT_FILE = Path(r"raw\IMK_data.csv")
TRAIN_FILE = Path(r"data_normalized\data_train.csv")
TEST_FILE = Path(r"data_normalized\data_test.csv")
VAL_FILE = Path(r"data_normalized\data_val.csv")

TEST_VAL_RATIO = 0.15
SEED = 42


def normalize_coordinates(values, divisor):
    result = []

    for value in values.split(","):
        normalized = float(value) / divisor
        normalized = max(0.0, min(1.0, normalized))
        result.append(f"{normalized:.12f}")

    return ",".join(result)


with INPUT_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    fieldnames = reader.fieldnames

for row in rows:
    width = float(row["width"])
    height = float(row["height"])

    row["x_list"] = normalize_coordinates(row["x_list"], width)
    row["y_list"] = normalize_coordinates(row["y_list"], height)

random.Random(SEED).shuffle(rows)

test_size = int(len(rows) * TEST_VAL_RATIO)
test_rows = rows[:test_size]
val_rows = rows[test_size:test_size*2]
train_rows = rows[test_size*2:]


def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


write_csv(TRAIN_FILE, train_rows)
write_csv(TEST_FILE, test_rows)
write_csv(VAL_FILE, val_rows)

print(f"Training rows: {len(train_rows)}")
print(f"Test rows: {len(test_rows)}")
print(f"Validation rows: {len(val_rows)}")
print(f"Written: {TRAIN_FILE}")
print(f"Written: {TRAIN_FILE}")
print(f"Written: {VAL_FILE}")