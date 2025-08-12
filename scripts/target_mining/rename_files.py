import datetime
import os

def main():
    dir_path = './data/stance_targets/base_targets'
    for filename in os.listdir(dir_path):
        base_name = filename.split('.')[0]
        month = base_name.split('_')[2]
        day = base_name.split('_')[3]
        if len(month) == 1 or len(day) == 1:
            file_path = os.path.join(dir_path, filename)
            year = base_name.split('_')[1]
            date_str = datetime.date(int(year), int(month), int(day))
            new_filename = f"targets_{date_str.strftime('%Y_%m_%d')}.parquet.zstd"
            new_file_path = os.path.join(dir_path, new_filename)
            os.rename(file_path, new_file_path)

if __name__ == "__main__":
    main()