import os

# Hàm xử lý từng file, log path và toàn bộ nội dung file ra file log
def process_file(filepath, log_file=None):
    try:
        # Đọc nội dung file dạng text (nếu là file nhị phân sẽ bỏ qua)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        content = f"Could not read file: {e}"

    info = (
        f"File path: {filepath}\n"
        "----FILE CONTENT START----\n"
        f"{content}\n"
        "----FILE CONTENT END----\n\n"
    )
    print(info)
    if log_file is not None:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(info)

def print_files_and_process(folder_path, log_file=None):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path, log_file=log_file)

if __name__ == "__main__":
    folder_path = "."
    log_file = "process_log.txt"
    print_files_and_process(folder_path, log_file=log_file)
