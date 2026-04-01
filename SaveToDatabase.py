from deepface import DeepFace
import json

pathToPhoto = "photos/MVO.jpg"
grade = "10-А"
name = "Мосійчук Владислав Олександрович"

def Save():
    result = DeepFace.represent(pathToPhoto, model_name="Facenet")
    embedding = result[0]["embedding"]
    embedding_float = [float(x) for x in embedding]
    userdata = {
        "name": name,
        "grade": grade,
        "embedding": embedding_float
    }
    with open("database.json", "r", encoding="utf-8") as dtb:
        try:
            data = json.load(dtb)
        except:
            data = []
    data.append(userdata)
    with open("database.json", "w", encoding="utf-8") as dtb:
        json.dump(data, dtb, ensure_ascii=False, indent=4)
Save()