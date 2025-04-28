import cv2
import threading
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO
from shapely.geometry import Point, box as shapely_box
from tkinter.filedialog import askopenfilename

# Load model YOLOv8n
model = YOLO("yolov8n.pt")

# Giao diện Tkinter
window = Tk()
window.title("Đếm người vào/ra")
window.geometry("900x700")

label_in = Label(window, text="Số người vào: 0", font=("Arial", 16))
label_in.pack(pady=5)

label_out = Label(window, text="Số người ra: 0", font=("Arial", 16))
label_out.pack(pady=5)

camera_label = Label(window)
camera_label.pack()

# Biến theo dõi
people_in = 0
people_out = 0
drawing_rect = False
start_point = None
end_point = None
rect_defined = False
inside_status = {}
video_path = None
cap = None
paused = False  # Thêm biến tạm dừng
running = False  # Để kiểm soát vòng lặp update_frame

# Hàm chọn video
def load_video():
    global cap, video_path, running, paused
    video_path = askopenfilename(title="Chọn video", filetypes=[("Video Files", "*.mp4;*.avi")])
    if video_path:
        if cap:
            cap.release()
        cap = cv2.VideoCapture(video_path)
        running = False
        paused = False

# Hàm vẽ vùng hình chữ nhật
def draw_rectangle(event):
    global drawing_rect, start_point, end_point, rect_defined
    if not drawing_rect:
        start_point = (event.x, event.y)
        drawing_rect = True
    else:
        end_point = (event.x, event.y)
        drawing_rect = False
        rect_defined = True
        print(f"Vẽ vùng đếm: {start_point} -> {end_point}")

camera_label.bind("<Button-1>", draw_rectangle)

# Nút start
def start_counting():
    global people_in, people_out, inside_status, running, paused
    if cap is None:
        print("Chưa chọn video!")
        return
    people_in = 0
    people_out = 0
    inside_status = {}
    paused = False
    if not running:
        running = True
        update_frame()

# Nút tạm dừng / tiếp tục
def toggle_pause():
    global paused
    if cap is None:
        print("Chưa chọn video!")
        return
    paused = not paused
    if not paused:
        update_frame()

# Hàm cập nhật frame
def update_frame():
    global people_in, people_out, inside_status, running

    if not running or paused:
        return

    ret, frame = cap.read()
    if not ret:
        print("Video đã kết thúc")
        running = False
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.track(source=frame_rgb, persist=True, tracker="bytetrack.yaml")

    frame_copy = frame.copy()

    if results[0].boxes.id is not None:
        for bbox, cls, conf, track_id in zip(
            results[0].boxes.xyxy,
            results[0].boxes.cls,
            results[0].boxes.conf,
            results[0].boxes.id,
        ):
            if int(cls) == 0 and conf > 0.4:  # Chỉ lấy người (class 0)
                x1, y1, x2, y2 = map(int, bbox)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame_copy, f"ID: {int(track_id)}", (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame_copy, (cx, cy), 3, (255, 0, 0), -1)

                this_id = int(track_id)
                current_point = Point(cx, cy)

                if rect_defined:
                    rect = shapely_box(
                        min(start_point[0], end_point[0]),
                        min(start_point[1], end_point[1]),
                        max(start_point[0], end_point[0]),
                        max(start_point[1], end_point[1])
                    )

                    is_inside_now = rect.contains(current_point)

                    if this_id in inside_status:
                        was_inside = inside_status[this_id]

                        if not was_inside and is_inside_now:
                            people_in += 1
                        elif was_inside and not is_inside_now:
                            people_out += 1

                    inside_status[this_id] = is_inside_now

    # Vẽ hình chữ nhật
    if rect_defined:
        cv2.rectangle(frame_copy, start_point, end_point, (0, 0, 255), 2)

    # Cập nhật giao diện
    label_in.config(text=f"Số người vào: {people_in}")
    label_out.config(text=f"Số người ra: {people_out}")

    img = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    window.after(10, update_frame)

# Các nút điều khiển
load_button = Button(window, text="Load Video", font=("Arial", 14), command=load_video)
load_button.pack(pady=5)

start_button = Button(window, text="Start", font=("Arial", 14), command=start_counting)
start_button.pack(pady=5)

pause_button = Button(window, text="Tạm dừng / Tiếp tục", font=("Arial", 14), command=toggle_pause)
pause_button.pack(pady=5)

# Khởi chạy GUI
window.mainloop()

# Giải phóng tài nguyên
if cap:
    cap.release()
cv2.destroyAllWindows()
