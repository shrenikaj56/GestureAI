import pandas as pd

path = "data/gestures/gesture_dataset.csv"

df = pd.read_csv(path)

# Keep the existing V_SIGN samples
v_sign = df[df["gesture"] == "v_sign"]

# Take the newly collected uppercase samples
new_data = df[
    df["gesture"].isin(
        ["OPEN_PALM", "FIST", "THUMBS_UP", "THUMBS_DOWN"]
    )
].copy()

# Convert labels to the correct lowercase names
new_data["gesture"] = new_data["gesture"].str.lower()

# Create the final 5-class dataset
final_df = pd.concat(
    [new_data, v_sign],
    ignore_index=True
)

# Save
final_df.to_csv(path, index=False)

print("\nFinal dataset:")
print(final_df["gesture"].value_counts())

print("\nTotal samples:", len(final_df))