import sys

def join_lines_to_paragraph(text):
    """
    Nối các dòng trong một chuỗi văn bản thành một đoạn văn bản duy nhất.

    Args:
        text: Chuỗi văn bản đầu vào với nhiều dòng.

    Returns:
        Một chuỗi văn bản duy nhất trên một dòng.
    """
    # Tách chuỗi thành các từ dựa trên khoảng trắng và ký tự xuống dòng
    words = text.split()
    # Nối các từ lại với nhau bằng một khoảng trắng
    return ' '.join(words)

def main():
    """
    Hàm chính để đọc input từ terminal và xử lý.
    """
    print("Nhập văn bản của bạn. Nhấn Enter trên một dòng trống để kết thúc:")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
            break
    
    input_text = "\n".join(lines)
    
    if input_text:
        # Chuyển đổi và in kết quả
        output_text = join_lines_to_paragraph(input_text)
        print("\n--- Kết quả ---")
        print(output_text)
    else:
        print("Không có văn bản nào được nhập.")

if __name__ == "__main__":
    main()