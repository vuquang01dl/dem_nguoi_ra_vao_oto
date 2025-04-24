import cv2
import os
import numpy as np
import onnxruntime
import smtplib
import RPi.GPIO as GPIO
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import serial
import time

# Thiết lập GPIO
RELAY_PIN = 13  # GPIO13 để điều khiển relay
BUTTON_PIN = 4  # GPIO4 để chụp ảnh và gửi email
BUZZER = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.output(BUZZER, GPIO.LOW)
# Khởi tạo Serial (chỉ làm 1 lần ở đầu chương trình chính)
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # Đợi Arduino khởi động

# Đường dẫn dữ liệu
EMBEDDINGS_PATH = r"/home/mypi/Desktop/thangmay/diemdanh_khuonmat-main/test_yolo_arcFce/data/faces-embeddings.npz"
SAVE_DIR = r"/home/mypi/Desktop/thangmay/diemdanh_khuonmat-main/test_yolo_arcFce/data/registered_faces"
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# Load dữ liệu embeddings
def load_database():
    if not os.path.exists(EMBEDDINGS_PATH):
        return [], np.array([])
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return list(data.keys()), np.array(list(data.values()))

# Nhận diện khuôn mặt
def recognize_face(frame, names, embeddings):
    face_embeddings, bboxes = extract_embedding(frame)
    if face_embeddings is None:
        return []
    results = []
    for emb, bbox in zip(face_embeddings, bboxes):
        similarities = cosine_similarity([emb], embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        name = names[best_match_idx] if best_score > 0.6 else "Unknown"
        results.append((name, bbox))
    return results

# Trích xuất embedding
def extract_embedding(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    if not faces:
        return None, None
    return [face.normed_embedding for face in faces], [face.bbox for face in faces]

# Mở cửa nếu khuôn mặt tồn tại trong database
def open_door():
    print("Gửi lệnh mở cửa đến Arduino...")
    arduino.write(b'A')  # Gửi ký tự 'A' dưới dạng byte

# Gửi email khi phát hiện nút nhấn
def send_email_with_attachment(image_path):
    sender_email = "vuquang01dl@gmail.com"
    sender_app_password = "wgmc xccj soba owzz"
    receiver_email = "tahuyhung07032003@gmail.com"
    
    subject = "Cảnh báo: Phát hiện người nhấn nút!"
    body = "Có người nhấn nút tại thang máy. Xem ảnh đính kèm."
    
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
    msg.attach(part)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Giám sát nút nhấn để chụp ảnh
# Giám sát nút nhấn để chụp ảnh và nhận diện khuôn mặt
def monitor_button(cap, names, embeddings):
    image_dir = "/home/mypi/face_data"  # Thay đổi thư mục lưu ảnh
    if not os.path.exists(image_dir):  # Kiểm tra thư mục có tồn tại không
        os.makedirs(image_dir)  # Tạo thư mục nếu không tồn tại
    
    while True:
        ret, frame = cap.read()  # Lấy frame từ camera mỗi lần lặp
        if not ret:
            print("Không thể đọc từ camera.")
            break

        # Nếu nút nhấn được nhấn, tiến hành gửi email và chụp ảnh
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            print("Đã nhấn nút, tiến hành gửi email và chụp ảnh...")
            GPIO.output(BUZZER, GPIO.HIGH)
            print("Đang mở cửa...")
            time.sleep(3)  # Mở cửa trong 10 giây
            print("Đang đóng cửa...")
            GPIO.output(BUZZER, GPIO.LOW)
            # Lưu ảnh vào một đường dẫn tạm thời
            image_path = os.path.join(image_dir, "alert.jpg")
            cv2.imwrite(image_path, frame)
            # Gửi email với ảnh đính kèm
            send_email_with_attachment(image_path)
            print(f"Email đã được gửi với ảnh từ {image_path}")
            time.sleep(2)  # Tránh gửi nhiều email liên tục
            continue  # Tiếp tục vòng lặp để tiếp tục nhận diện khuôn mặt

        # Nếu nút không nhấn, tiến hành nhận diện khuôn mặt
        faces = recognize_face(frame, names, embeddings)
        for name, bbox in faces:
            x1, y1, x2, y2 = map(int, bbox)
            if name != "Unknown":
                open_door()  # Mở cửa khi nhận diện được khuôn mặt hợp lệ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition", frame)  # Hiển thị video từ camera
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break
# Chạy phần nhận diện khuôn mặt và giám sát nút nhấn trong một luồng
def start_recognition():
    names, embeddings = load_database()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    print("Đang nhận diện khuôn mặt...")
    monitor_button(cap, names, embeddings)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        start_recognition()
    except KeyboardInterrupt:
        print("Stopping system...")
        GPIO.cleanup()
