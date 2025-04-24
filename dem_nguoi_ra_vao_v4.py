import cv2
import threading
from tkinter import *
from PIL import Image, ImageTk
from ultralytics import YOLO
from shapely.geometry import LineString, Point
from tkinter.filedialog import askopenfilename

# Load model YOLOv8n
model = YOLO("yolov8n.pt")

# Giao diện Tkinter
window = Tk()
window.title("Đếm người qua vạch")
window.geometry("900x700")

label_current = Label(window, text="Số người hiện tại: 0", font=("Arial", 16))
label_current.pack(pady=5)

label_max = Label(window, text="Số người cao nhất: 0", font=("Arial", 16))
label_max.pack(pady=5)

label_count = Label(window, text="Số người đi qua vạch: 0", font=("Arial", 16))
label_count.pack(pady=5)

camera_label = Label(window)
camera_label.pack()

# Biến theo dõi
current_people = 0
max_people = 0
line_count = 0

drawing_line = False
line_start = None
line_end = None
line_defined = False

prev_positions = {}
already_counted = set()

# Chọn video
video_path = askopenfilename(title="Chọn video", filetypes=[("Video Files", "*.mp4;*.avi")])
cap = cv2.VideoCapture(video_path)

# Hàm vẽ vạch
def draw_line(event):
    global drawing_line, line_start, line_end, line_defined
    if not drawing_line:
        line_start = (event.x, event.y)
        drawing_line = True
    else:
        line_end = (event.x, event.y)
        drawing_line = False
        line_defined = True
        print(f"Vạch đã vẽ: {line_start} -> {line_end}")

camera_label.bind("<Button-1>", draw_line)

# Nút start
def start_counting():
    global line_count
    line_count = 0
    update_frame()

start_button = Button(window, text="Start", font=("Arial", 14), command=start_counting)
start_button.pack(pady=10)

def update_frame():
    global current_people, max_people, line_count, prev_positions

    ret, frame = cap.read()
    if not ret:
        print("Video đã kết thúc")
        return

    # Dùng model YOLOv8n + ByteTrack
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")

    frame_copy = frame.copy()
    ids = []

    if results[0].boxes.id is not None:
        for box, cls, conf, track_id in zip(
            results[0].boxes.xyxy,
            results[0].boxes.cls,
            results[0].boxes.conf,
            results[0].boxes.id,
        ):
            if int(cls) == 0 and conf > 0.4:
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame_copy, f"ID: {int(track_id)}", (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame_copy, (cx, cy), 3, (255, 0, 0), -1)

                ids.append(int(track_id))

                if line_defined:
                    this_id = int(track_id)
                    new_point = Point(cx, cy)

                    if this_id in prev_positions:
                        prev_point = Point(prev_positions[this_id])
                        movement = LineString([prev_point, new_point])
                        line_seg = LineString([line_start, line_end])

                        if movement.crosses(line_seg) and this_id not in already_counted:
                            dir = cy - prev_positions[this_id][1]
                            if dir > 0:
                                line_count += 1  # Đi xuống
                            else:
                                line_count -= 1  # Đi lên
                            already_counted.add(this_id)

                    prev_positions[this_id] = (cx, cy)

    current_people = len(set(ids))
    max_people = max(max_people, current_people)

    if line_defined:
        cv2.line(frame_copy, line_start, line_end, (0, 0, 255), 2)

    label_current.config(text=f"Số người hiện tại: {current_people}")
    label_max.config(text=f"Số người cao nhất: {max_people}")
    label_count.config(text=f"Số người đi qua vạch: {line_count}")

    img = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    window.after(10, update_frame)

# Khởi chạy GUI
window.mainloop()

cap.release()
cv2.destroyAllWindows()
