import cv2
from deepface import DeepFace
import json
from flask import Flask, jsonify, request
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    if 'image' not in request.files:
        return jsonify({"text": "Помилка: Файл не отримано"}), 400
    file = request.files['image']
    frame = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    earlyEmbedding = DeepFace.represent(img_path=file, model_name="Facenet", enforce_detection = False, detector_backend = "retinaface")
    embeding = earlyEmbedding[0]["embedding"]
    with open("database.json", "r", encoding="utf-8") as dtb:
        users = json.load(dtb)
    bestMatch = None
    minDistance = 1.0
    message = "Щось вийшло не так"
    for user in users:
        res = DeepFace.verify(embeding, user["embedding"], model_name="Facenet", distance_metric="cosine", threshold = 0.5)
        if res["verified"] and res["distance"] < minDistance:
            minDistance = res["distance"]
            bestMatch = user["name"]
    if bestMatch:
        if True: # тут буде перевірятись чи є вже запис у журналі
            message = f"Відмічається {bestMatch}"
            # Тут буде відбуватись запис в журнал
        else:
            message = f"{bestMatch} Вже відмічався сьогодні"
    else:
        message = "Такої людини нема в базі"
    print(f"Отримано кадр: {file.filename}")
    return jsonify({
        "text": message
    })
if __name__ == '__main__':
    app.run(debug=True, port=5000)