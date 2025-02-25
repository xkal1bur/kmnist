# recognition/views.py
import base64
from io import BytesIO
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from PIL import Image, ImageOps
import math
import cv2
from sklearn.cluster import KMeans
import numpy as np

# import os
# import datetime
import torchvision.transforms as transforms
from django.shortcuts import render
from .models import model, label_map
import torch

def index(request):
    return render(request, 'index.html')


@csrf_exempt  # Para pruebas. En producción maneja CSRF de forma segura.
def predict(request):
    if request.method == "POST":
        data_url = request.POST.get("image")
        if not data_url:
            return HttpResponseBadRequest("No se recibió la imagen.")

        try:
            # Decodifica la imagen en base64
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)

            # Abre la imagen y la redimensiona a 64x64
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image = image.resize((64, 64))

            # Convertir a escala de grises (1 canal) que espera el modelo
            image = image.convert("L")

            # invertir los colores
            image = ImageOps.invert(image)

            # # Guarda la imagen para depuración
            # output_dir = "saved_images"
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            # filename = f"{timestamp}.png"
            # file_path = os.path.join(output_dir, filename)
            # image.save(file_path)

            # Define el pipeline de transformaciones
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.27045583724975586],
                                     std=[0.20256973803043365])
            ])

            # Aplica las transformaciones y añade la dimensión batch
            tensor_image = transform(image).unsqueeze(0)  # Forma: [1, 1, 64, 64]
        except Exception as e:
            return HttpResponseBadRequest("Error procesando la imagen: " + str(e))

        # Ejecuta la inferencia
        with torch.no_grad():
            output = model(tensor_image)
            pred = output.data.max(1)[1].item()

        predicted_kanji = label_map.get(pred, "Desconocido")
        return JsonResponse({
            "message": "Predicción exitosa.",
            "prediction": predicted_kanji
        })
    else:
        return HttpResponseBadRequest("Método no permitido, usa POST.")




def recortar_cuadrado_con_relleno(gray_img, cx, cy, side):
    """
    Crea una imagen cuadrada de lado 'side', centrada en (cx, cy) sobre 'gray_img'.
    Si se sale de la imagen original, se rellena con negro.
    Retorna la imagen cuadrada resultante (en escala de grises).
    """
    alto, ancho = gray_img.shape
    side = int(math.ceil(side))
    x1 = int(cx - side // 2)
    y1 = int(cy - side // 2)
    x2 = x1 + side
    y2 = y1 + side
    output = np.zeros((side, side), dtype=np.uint8)
    orig_x1 = max(x1, 0)
    orig_y1 = max(y1, 0)
    orig_x2 = min(x2, ancho)
    orig_y2 = min(y2, alto)
    if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
        return output
    dst_x = orig_x1 - x1
    dst_y = orig_y1 - y1
    subimg = gray_img[orig_y1:orig_y2, orig_x1:orig_x2]
    output[dst_y:dst_y + (orig_y2 - orig_y1),
    dst_x:dst_x + (orig_x2 - orig_x1)] = subimg
    return output

@csrf_exempt
def predict_batch(request):
    """
    Vista para procesar una imagen que contiene varios kanjis.
    Se segmenta la imagen en recortes cuadrados (usando clustering y la posición del "upper right")
    ordenados de derecha a izquierda y de arriba a abajo, y se aplica el modelo de reconocimiento
    a cada segmento.
    Se puede pasar opcionalmente 'n_clusters' en el POST (por defecto se usan 3).
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Método no permitido, use POST.")

    data_url = request.POST.get("image")
    if not data_url:
        return HttpResponseBadRequest("No se recibió la imagen.")

    try:
        # Decodificar la imagen base64 y convertirla a un arreglo OpenCV (BGR)
        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convertir a escala de grises para la segmentación
        gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Binarización (Otsu) y, si el fondo es claro, invertir
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binaria) > 127:
            binaria = 255 - binaria

        # Operaciones morfológicas para limpiar
        kernel = np.ones((3, 3), np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        centroids = []
        for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 5:
                continue
            boxes.append([x, y, x + w, y + h])
            centroids.append((x + w/2, y + h/2))

        # Número de clusters esperado (puede pasarse como parámetro; por defecto 3)
        n_clusters = request.POST.get("n_clusters")
        if n_clusters is not None:
            n_clusters = int(n_clusters)
        else:
            n_clusters = 3

        # Agrupar con K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(centroids)
        cluster_centers = kmeans.cluster_centers_

        # Agrupar cajas por cluster
        clusters = {}
        for cluster_id in range(n_clusters):
            clusters[cluster_id] = {"boxes": [], "centers": []}
        for i, lab in enumerate(labels):
            clusters[lab]["boxes"].append(boxes[i])
            clusters[lab]["centers"].append(centroids[i])

        # Para cada cluster, unir las cajas y calcular la esquina superior derecha (UR)
        results = []
        for cluster_id in clusters:
            cluster_boxes = clusters[cluster_id]["boxes"]
            min_x = min(b[0] for b in cluster_boxes)
            min_y = min(b[1] for b in cluster_boxes)
            max_x = max(b[2] for b in cluster_boxes)
            max_y = max(b[3] for b in cluster_boxes)
            # La esquina superior derecha (UR): (max_x, min_y)
            ur_x, ur_y = max_x, min_y

            # Usamos el centro del cluster obtenido por KMeans
            cx, cy = cluster_centers[cluster_id]
            dx1 = cx - min_x
            dx2 = max_x - cx
            dy1 = cy - min_y
            dy2 = max_y - cy
            needed_side = 2 * max(dx1, dx2, dy1, dy2)
            recorte_cuadrado = recortar_cuadrado_con_relleno(gris, cx, cy, needed_side)
            results.append({
                "cluster": cluster_id,
                "cx": cx,
                "cy": cy,
                "recorte": recorte_cuadrado,
                "ur_x": ur_x,
                "ur_y": ur_y,
                "ll_x": min_x,
                "ll_y": max_y
            })

        # Ordenar resultados:
        # Primero, por columna de derecha a izquierda (mayor ur_x primero)
        # y dentro de la misma columna, de arriba a abajo (menor ur_y primero).
        results.sort(key=lambda item: (-item['ur_x'], item['ur_y']))

        # Definir el pipeline de transformación para el modelo
        transform_pipeline = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.27045583724975586],
                                 std=[0.20256973803043365])
        ])

        # Procesar cada recorte con el modelo de reconocimiento
        predictions = []
        for res in results:
            recorte = res["recorte"]
            pil_img = Image.fromarray(recorte)  # Convertir a PIL para las transformaciones
            tensor_img = transform_pipeline(pil_img).unsqueeze(0)  # [1, 1, 64, 64]
            with torch.no_grad():
                output = model(tensor_img)
                pred = output.data.max(1)[1].item()
            predicted_kanji = label_map.get(pred, "Desconocido")
            predictions.append({
                "cluster": res["cluster"],
                "prediction": predicted_kanji,
                "center": {"cx": res["cx"], "cy": res["cy"]},
                "upper_right": {"x": res["ur_x"], "y": res["ur_y"]}
            })

        # # Opcional: guardar temporalmente los recortes para depuración (en "saved_images")
        # output_dir = "saved_images"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # for idx, res in enumerate(results, start=1):
        #     cv2.imwrite(os.path.join(output_dir, f"kanji_{idx}.png"), res["recorte"])
        #
        return JsonResponse({
            "message": "Predicción batch exitosa.",
            "num_segments": len(results),
            "predictions": predictions
        })

    except Exception as e:
        return HttpResponseBadRequest("Error procesando la imagen batch: " + str(e))
