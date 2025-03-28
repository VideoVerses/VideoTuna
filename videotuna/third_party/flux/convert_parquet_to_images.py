import io
import os

import pandas as pd
from PIL import Image

# Step 1: Load Parquet File
parquet_file_path = "data/train-00000-of-00001-dfb0d9df7ebab67e.parquet"  # Replace with your Parquet file path
output_directory = "data-res"  # Directory to save the images

# Load the Parquet file into a DataFrame
df = pd.read_parquet(parquet_file_path)

# Step 2: Print the column names
print("Columns in the Parquet file:")
print(df.columns)

# Load the Parquet file into a Pandas DataFrame
df = pd.read_parquet(parquet_file_path)


# Step 2: Process DataFrame Rows
for index, row in df.iterrows():
    # Extract the 'text' column as the filename (without extension)
    text_filename = row["text"]

    # Extract image data - assuming the image data is in a column called 'image_data'
    # This data should be in a format suitable to create an image (e.g., 2D numpy array)
    image_data = row["image"]  # Replace with the actual column name for image data
    # print(image_data.items())
    image_bytes = image_data["bytes"]

    # Step 2: Convert the bytes data to an image
    image = Image.open(io.BytesIO(image_bytes))

    # Step 3: Save the image to disk or process it further
    output_path = os.path.join(output_directory, text_filename + ".png")
    image.save(output_path)

    print(f"Saved image: {output_path}")

print("All images have been saved.")
