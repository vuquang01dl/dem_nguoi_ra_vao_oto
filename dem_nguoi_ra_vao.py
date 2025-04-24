import cv2
import threading
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load mô hình YOLOv8n
model = YOLO("yolov8n.pt")

# Khởi tạo biến đếm
current_people = 0
max_people = 0

# Hàm cập nhật camera và xử lý ảnh
def update_frame():
    global current_people, max_people

    ret, frame = cap.read()
    if not ret:
        return

    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")

    frame_copy = frame.copy()

    ids = []

    if results[0].boxes.id is not None:
        for box, cls, track_id in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id):
            if int(cls) == 0:  # Chỉ xử lý nhãn person
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame_copy, f"ID: {int(track_id)}", (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                ids.append(int(track_id))

    # Cập nhật số người hiện tại và max
    current_people = len(set(ids))
    max_people = max(max_people, current_people)

    # Hiển thị trên giao diện
    label_current.config(text=f"Số người hiện tại: {current_people}")
    label_max.config(text=f"Số người cao nhất: {max_people}")

    # Hiển thị frame lên Tkinter
    img = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    window.after(10, update_frame)

# Giao diện Tkinter
window = Tk()
window.title("Nhận diện người - YOLOv8")
window.geometry("800x600")

label_current = Label(window, text="Số người hiện tại: 0", font=("Arial", 16))
label_current.pack(pady=10)

label_max = Label(window, text="Số người cao nhất: 0", font=("Arial", 16))
label_max.pack(pady=5)

camera_label = Label(window)
camera_label.pack()

# Mở camera
cap = cv2.VideoCapture(0)

# Chạy cập nhật khung hình
update_frame()

# Bắt đầu giao diện
window.mainloop()

# Giải phóng tài nguyên khi thoát
cap.release()
cv2.destroyAllWindows()
