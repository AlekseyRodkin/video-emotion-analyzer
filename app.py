from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Лимит 50 МБ

# Инициализация модели CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Список эмоций (английские названия)
emotion_labels = [
    "harmony and calm",
    "awe and wonder",
    "nostalgia and memories",
    "intellectual enrichment",
    "creative inspiration",
    "unity and connection",
    "mind games and exploration",
    "freedom and adventure",
    "triumph and achievement",
    "gambling excitement",
    "thrill and excitement"
]

# Словарь с русскими переводами
emotion_translations = {
    "harmony and calm": "гармония и спокойствие",
    "awe and wonder": "восторг и удивление",
    "nostalgia and memories": "ностальгия и воспоминания",
    "intellectual enrichment": "интеллектуальное обогащение",
    "creative inspiration": "творческое вдохновение",
    "unity and connection": "единство и связь",
    "mind games and exploration": "игры разума и исследование",
    "freedom and adventure": "свобода и приключения",
    "triumph and achievement": "триумф и достижение",
    "gambling excitement": "азарт и волнение",
    "thrill and excitement": "острые ощущения"
}

# Папка для временного хранения видео
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'mp4'}

def allowed_file(filename):
    """Проверка расширения файла."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_video(video_path):
    """Анализ эмоций в видео."""
    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = 0
    max_frames = 30
    frame_skip = 5

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            inputs = processor(text=emotion_labels, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
            scores.append(probs)
        frame_count += 1

    cap.release()
    if scores:
        mean_scores = np.mean(scores, axis=0)
        emotion_percentages = {emotion: prob * 100 for emotion, prob in zip(emotion_labels, mean_scores)}
    else:
        emotion_percentages = {emotion: 0.0 for emotion in emotion_labels}
    
    return emotion_percentages

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # Проверка наличия файла
        if 'file' not in request.files:
            return render_template('index.html', error='Файл не выбран')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Файл не выбран')
        if file and allowed_file(file.filename):
            # Сохранение файла с уникальным именем
            filename = str(uuid.uuid4()) + '.mp4'
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            try:
                # Анализ видео
                emotion_percentages = analyze_video(video_path)
                # Удаление временного файла
                os.remove(video_path)
                # Форматирование результатов с переводами
                results = [
                    {
                        "english": emotion,
                        "russian": emotion_translations[emotion],
                        "percentage": f"{percentage:.2f}%"
                    }
                    for emotion, percentage in emotion_percentages.items()
                ]
                return render_template('index.html', results=results, video_name=file.filename)
            except Exception as e:
                os.remove(video_path)
                return render_template('index.html', error=f'Ошибка обработки: {str(e)}')
        else:
            return render_template('index.html', error='Загружайте только MP4')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)