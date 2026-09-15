import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "IMK_data.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

TEST_VAL_RATIO = 0.1
SEED = 0


def normalize_coordinates(values, divisor):
    # Clamp normalized coordinates so malformed measurements cannot leave the [0, 1] range.
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
val_rows = rows[:test_size]
test_rows = rows[test_size:test_size*2]
train_rows = rows[test_size*2:]


def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


write_csv(TRAIN_FILE, train_rows)
write_csv(VALIDATION_FILE, val_rows)
write_csv(TEST_FILE, test_rows)

print(f"Training rows: {len(train_rows)}")
print(f"Validation rows: {len(val_rows)}")
print(f"Test rows: {len(test_rows)}")
print(f"Written: {TRAIN_FILE}, {VALIDATION_FILE}, {TEST_FILE}")