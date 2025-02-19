# recognition/views.py
import base64
from io import BytesIO
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from PIL import Image, ImageOps

import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from django.shortcuts import render
def index(request):
    return render(request, 'index.html')

# Definición del modelo
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(14*14*64, 128)
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop1(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# Carga del modelo y pesos (se hace una sola vez)
model = Net()
model.load_state_dict(torch.load('recognition/modelo_kanji.pth', map_location=torch.device('cpu')))
model.eval()

# Diccionario de mapeo de índices a símbolos Kanji
label_map = {1: "一", 2: "事", 3: "云", 4: "人", 5: "出", 6:"又", 7:"見", 8:"入", 9:"大", 10:"物", 11:"子",12:"此",13:"其",14:"也",15:"日",16:"小",17:"方",18:"上",19:"是",20:"三"}

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

            # Guarda la imagen para depuración
            output_dir = "saved_images"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.png"
            file_path = os.path.join(output_dir, filename)
            image.save(file_path)

            # Define el pipeline de transformaciones
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.18330305814743042],
                                     std=[0.3583868443965912])
            ])

            # Aplica las transformaciones y añade la dimensión batch
            tensor_image = transform(image).unsqueeze(0)  # Forma: [1, 1, 64, 64]
        except Exception as e:
            return HttpResponseBadRequest("Error procesando la imagen: " + str(e))

        # Ejecuta la inferencia
        with torch.no_grad():
            output = model(tensor_image)
            pred = output.data.max(1, keepdim=True)[1].item() + 1

        predicted_kanji = label_map.get(pred, "Desconocido")
        return JsonResponse({
            "message": "Predicción exitosa.",
            "prediction": predicted_kanji
        })
    else:
        return HttpResponseBadRequest("Método no permitido, usa POST.")
