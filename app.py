import os
import csv
import zipfile
import threading
import uuid
import time
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for

# Импорты из проекта lenta_recognition (путь может отличаться – уточните)
try:
    from cv_module.detection.price_tag_detector import PriceTagDetector
except ImportError:
    # Заглушка на случай, если класс не найден – вы можете вручную указать правильный импорт
    print("Ошибка: не удалось импортировать PriceTagDetector. Проверьте структуру cv_module.")
    PriceTagDetector = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Хранилище задач
jobs = {}

class VideoFrameIterator:
    """Простой итератор по кадрам видео"""
    def __init__(self, video_path, max_frames=None):
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.max_frames = max_frames or self.total_frames
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max_frames or self.current >= self.total_frames:
            self.cap.release()
            raise StopIteration
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        frame_idx = self.current
        self.current += 1
        return frame_idx, frame

    def __len__(self):
        return min(self.max_frames, self.total_frames)

def process_video(job_id, video_path, output_dir, progress_callback):
    """
    Основная логика обработки видео детектором ценников.
    Сохраняет вырезанные ценники и CSV.
    """
    if PriceTagDetector is None:
        raise ImportError("PriceTagDetector не импортирован. Проверьте настройки.")

    rectified_dir = os.path.join(output_dir, 'rectified')
    os.makedirs(rectified_dir, exist_ok=True)

    # Инициализация детектора (возможно, требуется передать параметры)
    detector = PriceTagDetector()

    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'tag_id', 'confidence', 'text', 'barcode', 'rectified_image'])

        frame_iterator = VideoFrameIterator(video_path)
        total_frames = len(frame_iterator)
        tag_counter = 0
        saved_images = []

        for frame_idx, frame in frame_iterator:
            # Детекция – реальный метод может называться иначе
            # Ожидаем, что detections – список объектов с полями:
            #   bbox (x,y,w,h), confidence, text, barcode, rectified_image (numpy array)
            detections = detector.detect(frame)   # или detector.process(frame)

            for det in detections:
                tag_counter += 1
                tag_id = f"tag_{job_id}_{frame_idx}_{tag_counter}"
                img_filename = f"{tag_id}.png"
                img_path = os.path.join(rectified_dir, img_filename)

                # Сохраняем выпрямленное изображение, если есть
                if hasattr(det, 'rectified_image') and det.rectified_image is not None:
                    cv2.imwrite(img_path, det.rectified_image)
                elif hasattr(det, 'bbox'):
                    x, y, w, h = det.bbox
                    crop = frame[y:y+h, x:x+w]
                    cv2.imwrite(img_path, crop)
                else:
                    continue

                saved_images.append(img_filename)
                writer.writerow([
                    frame_idx,
                    tag_id,
                    getattr(det, 'confidence', 1.0),
                    getattr(det, 'text', ''),
                    getattr(det, 'barcode', ''),
                    img_filename
                ])

            progress = int((frame_idx + 1) / total_frames * 100)
            progress_callback(progress)

    # Создаём ZIP-архив
    zip_path = os.path.join(output_dir, f"{job_id}_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(csv_path, arcname='results.csv')
        for img in saved_images:
            img_full = os.path.join(rectified_dir, img)
            zf.write(img_full, arcname=f"rectified/{img}")

    return zip_path

def async_process(job_id, video_path, output_dir):
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0

        def progress_callback(p):
            jobs[job_id]['progress'] = p

        zip_path = process_video(job_id, video_path, output_dir, progress_callback)

        jobs[job_id]['status'] = 'done'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['zip_path'] = zip_path
        jobs[job_id]['csv_path'] = os.path.join(output_dir, 'results.csv')
    except Exception as e:
        import traceback
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['traceback'] = traceback.format_exc()
        print(traceback.format_exc())

# ------------------- Эндпоинты -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    job_id = str(uuid.uuid4())[:8] + '_' + str(int(time.time()))
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, file.filename)
    file.save(video_path)

    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(output_dir, exist_ok=True)

    jobs[job_id] = {
        'status': 'uploaded',
        'progress': 0,
        'video_path': video_path,
        'output_dir': output_dir,
        'original_filename': file.filename
    }

    # Запуск фоновой обработки
    thread = threading.Thread(target=async_process, args=(job_id, video_path, output_dir))
    thread.start()

    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'error': job.get('error', None)
    })

@app.route('/processing/<job_id>')
def processing_page(job_id):
    if job_id not in jobs:
        return "Job not found", 404
    return render_template('processing.html', job_id=job_id)

@app.route('/result/<job_id>')
def result_page(job_id):
    if job_id not in jobs:
        return "Job not found", 404
    if jobs[job_id]['status'] != 'done':
        # Если ещё не готово – перенаправляем на страницу обработки
        return redirect(url_for('processing_page', job_id=job_id))
    return render_template('result.html', job_id=job_id)

@app.route('/download_zip/<job_id>')
def download_zip(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return "File not ready", 404
    zip_path = job.get('zip_path')
    if not zip_path or not os.path.exists(zip_path):
        return "ZIP file not found", 404
    return send_file(zip_path, as_attachment=True, download_name=f"{job_id}_results.zip")

@app.route('/download_csv/<job_id>')
def download_csv(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return "File not ready", 404
    csv_path = job.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        return "CSV file not found", 404
    return send_file(csv_path, as_attachment=True, download_name=f"{job_id}_results.csv")

if __name__ == '__main__':
    print("=" * 60)
    print("Сервер детекции ценников lenta_recognition")
    print("Запущен на http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000)