import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk,messagebox
from PIL import Image,ImageTk
import threading
import time
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.camera_active = False
        self.dataset_collection_active = False
        self.recognition_active = False
        self.current_person_name = ""
        self.current_person_id = 0
        self.dataset_dir = "dataset"
        self.models_dir = "models"
        self.current_frame = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.sample_count = 0
        self.model_type = tk.StringVar(value="ANN")  
        self.model_ann = None
        self.model_cnn = None
        self.label_encoder = None 
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.create_widgets()
        self.load_existing_models()
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_panel = ttk.LabelFrame(main_frame, text="Video Feed")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(left_panel)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        dataset_frame = ttk.LabelFrame(right_panel, text="Dataset Collection")
        dataset_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(dataset_frame, text="Person Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.name_entry = ttk.Entry(dataset_frame, width=20)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)
        self.collect_btn = ttk.Button(dataset_frame, text="Start Collection", command=self.toggle_dataset_collection)
        self.collect_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(dataset_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        self.status_label = ttk.Label(dataset_frame, text="Status: Ready")
        self.status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        training_frame = ttk.LabelFrame(right_panel, text="Model Training")
        training_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(training_frame, text="Model Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        model_frame = ttk.Frame(training_frame)
        model_frame.grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(model_frame, text="ANN", variable=self.model_type, value="ANN").pack(side=tk.LEFT)
        ttk.Radiobutton(model_frame, text="CNN", variable=self.model_type, value="CNN").pack(side=tk.LEFT)
        self.train_btn = ttk.Button(training_frame, text="Train Model", command=self.train_model)
        self.train_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        recognition_frame = ttk.LabelFrame(right_panel, text="Face Recognition")
        recognition_frame.pack(fill=tk.X, padx=5, pady=5)
        self.recognize_btn = ttk.Button(recognition_frame, text="Start Recognition", command=self.toggle_recognition)
        self.recognize_btn.pack(fill=tk.X, padx=5, pady=5)
        self.recognition_label = ttk.Label(recognition_frame, text="Recognition: Inactive")
        self.recognition_label.pack(fill=tk.X, padx=5, pady=5)
        camera_frame = ttk.Frame(right_panel)
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        self.camera_btn = ttk.Button(camera_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_btn.pack(fill=tk.X, padx=5, pady=5)
        self.statusbar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    def toggle_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.camera_btn.config(text="Stop Camera")
            self.video_thread = threading.Thread(target=self.update_video)
            self.video_thread.daemon = True
            self.video_thread.start()
        else:
            self.camera_active = False
            self.camera_btn.config(text="Start Camera")
            self.dataset_collection_active = False
            self.recognition_active = False
            self.collect_btn.config(text="Start Collection")
            self.recognize_btn.config(text="Start Recognition")
    def update_video(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.camera_active = False
            self.camera_btn.config(text="Start Camera")
            return
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.current_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if self.dataset_collection_active and len(faces) == 1:
                    self.collect_face(gray, x, y, w, h)
                if self.recognition_active and len(faces) == 1:
                    self.recognize_face(gray, x, y, w, h)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        if self.cap.isOpened():
            self.cap.release()
    def toggle_dataset_collection(self):
        if not self.camera_active:
            messagebox.showinfo("Info", "Please start the camera first")
            return
        if not self.dataset_collection_active:
            name = self.name_entry.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name")
                return
            self.current_person_name = name
            self.current_person_id = len([d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))])
            person_dir = os.path.join(self.dataset_dir, str(self.current_person_id) + "_" + self.current_person_name)
            os.makedirs(person_dir, exist_ok=True)
            self.sample_count = 0
            self.progress_var.set(0)
            self.dataset_collection_active = True
            self.collect_btn.config(text="Stop Collection")
            self.status_label.config(text=f"Status: Collecting samples for {name}")
        else:
            self.dataset_collection_active = False
            self.collect_btn.config(text="Start Collection")
            self.status_label.config(text="Status: Collection completed")
    def collect_face(self, gray, x, y, w, h):
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        person_dir = os.path.join(self.dataset_dir, str(self.current_person_id) + "_" + self.current_person_name)
        filename = os.path.join(person_dir, f"sample_{self.sample_count}.jpg")
        cv2.imwrite(filename, face_resized)
        self.sample_count += 1
        progress = min(int(self.sample_count / 100 * 100), 100)
        self.progress_var.set(progress)
        self.status_label.config(text=f"Status: Collected {self.sample_count} samples")
        if self.sample_count >= 100:
            self.dataset_collection_active = False
            self.collect_btn.config(text="Start Collection")
            self.status_label.config(text="Status: Collection completed")
            messagebox.showinfo("Info", f"Dataset collection for {self.current_person_name} completed")
    def train_model(self):
        if self.dataset_collection_active or self.recognition_active:
            messagebox.showwarning("Warning", "Please stop collection or recognition first")
            return
        if not os.path.exists(self.dataset_dir) or len(os.listdir(self.dataset_dir)) == 0:
            messagebox.showwarning("Warning", "No dataset found. Please collect face data first.")
            return
        self.status_label.config(text="Status: Training model...")
        self.statusbar.config(text="Training model... This may take a while")
        self.root.update()
        threading.Thread(target=self._train_model_thread).start()
    def _train_model_thread(self):
        try:
            X = []
            y = []
            person_dirs = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
            if not person_dirs:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No valid dataset directories found"))
                self.root.after(0, lambda: self.status_label.config(text="Status: Training failed"))
                self.root.after(0, lambda: self.statusbar.config(text="Ready"))
                return
            for person_dir in person_dirs:
                person_path = os.path.join(self.dataset_dir, person_dir)
                person_id = person_dir.split("_")[0]
                for img_file in os.listdir(person_path):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(person_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            X.append(img)
                            y.append(person_dir)
            if not X:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No valid images found in dataset"))
                self.root.after(0, lambda: self.status_label.config(text="Status: Training failed"))
                self.root.after(0, lambda: self.statusbar.config(text="Ready"))
                return
            X = np.array(X)
            y = np.array(y)
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            num_classes = len(self.label_encoder.classes_)
            y_categorical = to_categorical(y_encoded, num_classes)
            with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            if self.model_type.get() == "ANN":
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                self.model_ann = Sequential([
                    Dense(512, activation='relu', input_shape=(X_train_flat.shape[1],)),
                    Dropout(0.2),
                    Dense(256, activation='relu'),
                    Dropout(0.2),
                    Dense(num_classes, activation='softmax')
                ])
                self.model_ann.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
                self.model_ann.fit(X_train_flat, y_train, batch_size=32, epochs=10, validation_data=(X_test_flat, y_test), verbose=1)
                self.model_ann.save(os.path.join(self.models_dir, 'ann_model.h5'))
                loss, accuracy = self.model_ann.evaluate(X_test_flat, y_test)
            else:
                X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
                X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
                self.model_cnn = Sequential([
                    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(64, kernel_size=(3, 3), activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(num_classes, activation='softmax')
                ])
                self.model_cnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
                self.model_cnn.fit(X_train_cnn, y_train, batch_size=32, epochs=10, validation_data=(X_test_cnn, y_test), verbose=1)
                self.model_cnn.save(os.path.join(self.models_dir, 'cnn_model.h5'))
                loss, accuracy = self.model_cnn.evaluate(X_test_cnn, y_test)
            self.root.after(0, lambda: self.status_label.config(text=f"Status: Training completed with {accuracy*100:.2f}% accuracy"))
            self.root.after(0, lambda: self.statusbar.config(text="Ready"))
            self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"Model trained with {accuracy*100:.2f}% accuracy"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Status: Training failed"))
            self.root.after(0, lambda: self.statusbar.config(text="Ready"))
    def load_existing_models(self):
        try:
            label_encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            ann_model_path = os.path.join(self.models_dir, 'ann_model.h5')
            if os.path.exists(ann_model_path):
                self.model_ann = load_model(ann_model_path)
            cnn_model_path = os.path.join(self.models_dir, 'cnn_model.h5')
            if os.path.exists(cnn_model_path):
                self.model_cnn = load_model(cnn_model_path)
            if self.model_ann is not None or self.model_cnn is not None:
                self.statusbar.config(text="Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    def toggle_recognition(self):
        if not self.camera_active:
            messagebox.showinfo("Info", "Please start the camera first")
            return
        if self.model_type.get() == "ANN" and self.model_ann is None:
            messagebox.showwarning("Warning", "ANN model not trained. Please train the model first.")
            return
        elif self.model_type.get() == "CNN" and self.model_cnn is None:
            messagebox.showwarning("Warning", "CNN model not trained. Please train the model first.")
            return
        if not self.recognition_active:
            self.recognition_active = True
            self.recognize_btn.config(text="Stop Recognition")
            self.recognition_label.config(text="Recognition: Active")
        else:
            self.recognition_active = False
            self.recognize_btn.config(text="Start Recognition")
            self.recognition_label.config(text="Recognition: Inactive")
    def recognize_face(self, gray, x, y, w, h):
        try:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            face_normalized = face_resized.astype('float32') / 255.0
            if self.model_type.get() == "ANN":
                face_flat = face_normalized.reshape(1, -1)
                prediction = self.model_ann.predict(face_flat, verbose=0)
            else:
                face_cnn = face_normalized.reshape(1, 100, 100, 1)
                prediction = self.model_cnn.predict(face_cnn, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            if confidence > 70:
                person_name = self.label_encoder.classes_[predicted_class]
                name = person_name.split('_', 1)[1] if '_' in person_name else person_name
                self.recognition_label.config(text=f"Recognized: {name}\nConfidence: {confidence:.2f}%")
            else:
                self.recognition_label.config(text=f"Unknown face\nConfidence: {confidence:.2f}%")
        except Exception as e:
            self.recognition_label.config(text=f"Error: {str(e)}")
    def on_closing(self):
        self.camera_active = False
        time.sleep(0.5)
        self.root.destroy()
        sys.exit(0)
def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()
