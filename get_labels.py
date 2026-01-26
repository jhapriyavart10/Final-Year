import kagglehub
import shutil
import os
import glob

def get_labels_file():
    print("Locating dataset...")
    try:
        # This returns the path to the CACHED dataset
        path = kagglehub.dataset_download("nih-chest-xrays/sample")
        print(f"Dataset path: {path}")
        
        # Look for the CSV
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            print("No CSV found in dataset folder.")
            return
            
        src_csv = csv_files[0] # Usually sample_labels.csv
        dst_csv = os.path.join("data", "sample_labels.csv")
        
        shutil.copy(src_csv, dst_csv)
        print(f"Copied {os.path.basename(src_csv)} to {dst_csv}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_labels_file()
