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
window.geometry("1000x800")

# Khung chứa số liệu
info_frame = Frame(window)
info_frame.pack(pady=10)

label_in = Label(info_frame, text="Số người vào: 0", font=("Arial", 16))
label_in.grid(row=0, column=0, padx=20)

label_out = Label(info_frame, text="Số người ra: 0", font=("Arial", 16))
label_out.grid(row=0, column=1, padx=20)

# Hiển thị video
camera_label = Label(window)
camera_label.pack(pady=10)

# Label thông báo
status_label = Label(window, text="", font=("Arial", 14), fg="blue")
status_label.pack(pady=5)

# Khung chứa các nút
button_frame = Frame(window)
button_frame.pack(pady=10)

load_button = Button(button_frame, text="Load Video", font=("Arial", 14), command=lambda: load_video())
load_button.grid(row=0, column=0, padx=10)

select_in_button = Button(button_frame, text="Chọn ô vào", font=("Arial", 14), command=lambda: select_in_area())
select_in_button.grid(row=0, column=1, padx=10)

select_out_button = Button(button_frame, text="Chọn ô ra", font=("Arial", 14), command=lambda: select_out_area())
select_out_button.grid(row=0, column=2, padx=10)

start_button = Button(button_frame, text="Start", font=("Arial", 14), command=lambda: start_counting())
start_button.grid(row=0, column=3, padx=10)

pause_button = Button(button_frame, text="Tạm dừng / Tiếp tục", font=("Arial", 14), command=lambda: toggle_pause())
pause_button.grid(row=0, column=4, padx=10)

# Biến theo dõi
people_in = 0
people_out = 0
drawing_rect = False
selecting_in = False
selecting_out = False
start_point_in = None
end_point_in = None
start_point_out = None
end_point_out = None
rect_in_defined = False
rect_out_defined = False
inside_status_in = {}
inside_status_out = {}
video_path = None
cap = None
paused = False
running = False

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
        status_label.config(text="Đã chọn video!", fg="green")

# Hàm xử lý click chuột để vẽ vùng vào/ra
def draw_rectangle(event):
    global drawing_rect, start_point_in, end_point_in, start_point_out, end_point_out
    global rect_in_defined, rect_out_defined

    if selecting_in:
        if not drawing_rect:
            start_point_in = (event.x, event.y)
            drawing_rect = True
        else:
            end_point_in = (event.x, event.y)
            drawing_rect = False
            rect_in_defined = True
            status_label.config(text="Đã chọn vị trí ô đỏ (vào) thành công!", fg="red")

    elif selecting_out:
        if not drawing_rect:
            start_point_out = (event.x, event.y)
            drawing_rect = True
        else:
            end_point_out = (event.x, event.y)
            drawing_rect = False
            rect_out_defined = True
            status_label.config(text="Đã chọn vị trí ô xanh (ra) thành công!", fg="green")

camera_label.bind("<Button-1>", draw_rectangle)

# Nút chọn vùng vào
def select_in_area():
    global selecting_in, selecting_out
    selecting_in = True
    selecting_out = False
    status_label.config(text="Chế độ chọn vùng VÀO: Click 2 điểm!", fg="red")

# Nút chọn vùng ra
def select_out_area():
    global selecting_in, selecting_out
    selecting_in = False
    selecting_out = True
    status_label.config(text="Chế độ chọn vùng RA: Click 2 điểm!", fg="green")

# Nút start
def start_counting():
    global people_in, people_out, inside_status_in, inside_status_out, running, paused
    if cap is None:
        status_label.config(text="Chưa chọn video!", fg="red")
        return
    people_in = 0
    people_out = 0
    inside_status_in = {}
    inside_status_out = {}
    paused = False
    if not running:
        running = True
        update_frame()
    status_label.config(text="Đang đếm người...", fg="blue")

# Nút tạm dừng / tiếp tục
def toggle_pause():
    global paused
    if cap is None:
        status_label.config(text="Chưa chọn video!", fg="red")
        return
    paused = not paused
    if paused:
        status_label.config(text="Tạm dừng!", fg="orange")
    else:
        status_label.config(text="Tiếp tục đếm người...", fg="blue")
        update_frame()

# Hàm cập nhật frame
def update_frame():
    global people_in, people_out, inside_status_in, inside_status_out, running

    if not running or paused:
        return

    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Video đã kết thúc", fg="black")
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

                if rect_in_defined:
                    rect_in = shapely_box(
                        min(start_point_in[0], end_point_in[0]),
                        min(start_point_in[1], end_point_in[1]),
                        max(start_point_in[0], end_point_in[0]),
                        max(start_point_in[1], end_point_in[1])
                    )

                    is_inside_in_now = rect_in.contains(current_point)

                    if this_id in inside_status_in:
                        was_inside_in = inside_status_in[this_id]
                        if not was_inside_in and is_inside_in_now:
                            people_in += 1
                    inside_status_in[this_id] = is_inside_in_now

                if rect_out_defined:
                    rect_out = shapely_box(
                        min(start_point_out[0], end_point_out[0]),
                        min(start_point_out[1], end_point_out[1]),
                        max(start_point_out[0], end_point_out[0]),
                        max(start_point_out[1], end_point_out[1])
                    )

                    is_inside_out_now = rect_out.contains(current_point)

                    if this_id in inside_status_out:
                        was_inside_out = inside_status_out[this_id]
                        if was_inside_out and not is_inside_out_now:
                            people_out += 1
                    inside_status_out[this_id] = is_inside_out_now

    if rect_in_defined:
        cv2.rectangle(frame_copy, start_point_in, end_point_in, (0, 0, 255), 2)
    if rect_out_defined:
        cv2.rectangle(frame_copy, start_point_out, end_point_out, (0, 255, 0), 2)

    label_in.config(text=f"Số người vào: {people_in}")
    label_out.config(text=f"Số người ra: {people_out}")

    img = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    window.after(10, update_frame)

# Khởi chạy giao diện
window.mainloop()

if cap:
    cap.release()
cv2.destroyAllWindows()
