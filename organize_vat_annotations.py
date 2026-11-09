# code to organize VideoAttentionTarget (VAT) annotations into a single file

import os
import csv

# ===== Configuration =====
input_folder = "/home/icub/lgori/sharingan/data/videoattentiontarget/annotations/train"
output_file = "/home/icub/lgori/sharingan/data/videoattentiontarget/annotations/train_merged_output.txt"

# =========================

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")

                with open(file_path, "r", newline="", encoding="utf-8") as infile:
                    reader = csv.reader(infile)

                    # Write remaining rows
                    for row in reader:
                        # append to the row the relative path of the file (without the input_folder prefix)
                        relative_path = os.path.relpath(file_path, input_folder)
                        # append it to the same row
                        row.append(relative_path)
                        writer.writerow(row)

print(f"\nMerged file created:\n{output_file}")