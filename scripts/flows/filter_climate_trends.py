import os
import shutil

import polars as pl

def main():
    trends_path = './data/stance_targets/noun_phrase_kernelreg_trends'
    climate_trends_path = './data/stance_targets/climate_trends'
    os.makedirs(climate_trends_path, exist_ok=True)

    keywords = ['carbon', 'climate', 'energy', 'fossil', 'fuel', 'gas', 'oil', 'coal', 'solar', 'renewable', 'emissions', 'sustainability', 'environment', 'warming', 'greenhouse', 'net-zero', 'pipeline', 'nuclear']

    for filename in os.listdir(trends_path):
        if any(keyword in filename for keyword in keywords):
            src_path = os.path.join(trends_path, filename)
            dst_path = os.path.join(climate_trends_path, filename)
            shutil.copyfile(src_path, dst_path)
            print(f"Copied {filename} to climate_trends")

if __name__ == "__main__":
    main()