# 🛰️ Semantic Satellite Segmentation

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-FF6F00.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.21.0-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<h3>🚀 Веб-приложение для автоматической семантической сегментации спутниковых снимков</h3>

[**🔗 ПОПРОБОВАТЬ ПРИЛОЖЕНИЕ**](https://dpo-segmentation-model.streamlit.app/) • [Установка](#-быстрый-старт) • [API](#-api-документация) 

</div>

---

## 🎥 Демонстрация

<div align="center">
  <img src="example_segmentation.gif?text=Демонстрация+сегментации" alt="Demo" width="80%">
</div>

## 🌐 Онлайн-версия

<div align="center">

### 👉 [**ОТКРЫТЬ ПРИЛОЖЕНИЕ**](https://dpo-segmentation-model.streamlit.app/) 👈

| 🔗 Компонент | URL | ⚡ Статус |
|--------------|-----|----------|
| **Frontend** | [dpo-segmentation-model.streamlit.app](https://dpo-segmentation-model.streamlit.app/) | ![Streamlit](https://img.shields.io/badge/Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| **Backend API** | [dpo-segmentation-model.onrender.com](https://dpo-segmentation-model.onrender.com) | ![Render](https://img.shields.io/badge/Live-46E3B7?style=for-the-badge&logo=render&logoColor=white) |

</div>

> ⚠️ **Важно:** Первый запуск может занять до 2-3 минут из-за холодного старта сервера. Используйте изображения с разрешением не более **512x512 px** из-за ограничений бесплатного хостинга. Для работы с большими изображениями рекомендуется локальное развертывание.

## 🚀 Возможности

<table>
<tr>
<td width="50%">

### 🚀 Основные функции

- **6 классов сегментации** с высокой точностью
- **Веб-интерфейс** для быстрой обработки
- **REST API** для интеграции
- **Пакетная обработка** изображений
- **QGIS плагин** (в разработке)

</td>
<td width="50%">

### 🎨 Классы объектов

| Класс | Цвет | Hex |
|-------|------|-----|
| 🏠 Здания | ![#3C1098](https://placehold.co/15x15/3C1098/3C1098.png) | `#3C1098` |
| 🌍 Земля | ![#8429F6](https://placehold.co/15x15/8429F6/8429F6.png) | `#8429F6` |
| 🛣️ Дороги | ![#6EC1E4](https://placehold.co/15x15/6EC1E4/6EC1E4.png) | `#6EC1E4` |
| 🌳 Растительность | ![#FEDD3A](https://placehold.co/15x15/FEDD3A/FEDD3A.png) | `#FEDD3A` |
| 💧 Вода | ![#E2A929](https://placehold.co/15x15/E2A929/E2A929.png) | `#E2A929` |
| ⬜ Неразмеченное | ![#9B9B9B](https://placehold.co/15x15/9B9B9B/9B9B9B.png) | `#9B9B9B` |

</td>
</tr>
</table>

## 🏗️ Архитектура

```mermaid
graph LR
    A[Спутниковый снимок] --> B[Веб-интерфейс<br/>Streamlit]
    A --> C[REST API<br/>FastAPI]
    B --> D[TensorFlow Lite<br/>Модель]
    C --> D
    D --> E[Сегментированное<br/>изображение]
    C --> F[QGIS плагин<br/>🚧 В разработке]
```

## 💻 Технологический стек

<div align="center">

| Backend | Frontend | ML/DL | Обработка изображений |
|---------|----------|-------|----------------------|
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | ![Pillow](https://img.shields.io/badge/Pillow-8CAAE6?style=for-the-badge&logo=python&logoColor=white) |

</div>

## 🚀 Быстрый старт

### 📋 Требования

- **Python** 3.8-3.10
- **RAM** 4GB+
- **Disk** 500MB свободного места

### 📦 Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/satellite-segmentation.git
cd satellite-segmentation

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### ⚡ Запуск приложения

<details>
<summary><b>🖥️ Backend сервер</b></summary>

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Сервер будет доступен по адресу: http://localhost:8000

</details>

<details>
<summary><b>🎨 Frontend интерфейс</b></summary>

```bash
streamlit run app.py
```

Интерфейс будет доступен по адресу: http://localhost:8501

</details>

## 📖 API Документация

### 🔧 Endpoints

#### `POST /predict/`

Выполняет семантическую сегментацию загруженного изображения.

<details>
<summary><b>Параметры запроса</b></summary>

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `file` | `File` | обязательный | Изображение для сегментации |
| `patch_size` | `int` | 256 | Размер патча (128-512) |
| `subdivisions` | `int` | 2 | Количество подразделений (1-4) |

</details>

<details>
<summary><b>Пример использования</b></summary>

**Python (requests):**
```python
import requests

url = "http://localhost:8000/predict/"
files = {"file": open("satellite.jpg", "rb")}
params = {"patch_size": 256, "subdivisions": 2}

response = requests.post(url, files=files, params=params)

with open("result.png", "wb") as f:
    f.write(response.content)
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict/?patch_size=256&subdivisions=2" \
     -H "accept: image/png" \
     -F "file=@satellite.jpg" \
     -o result.png
```

</details>

## 🔬 О проекте

### 🧠 Модель

<table>
<tr>
<td width="60%">

**Архитектура:**
- 🏗️ **Backbone:** EfficientNet-B1
- 🔄 **Архитектура:** U-Net
- 📊 **Формат:** TensorFlow Lite (квантованная)
- 📏 **Размер модели:** ~20MB
- ⚡ **Скорость:** 2-5 сек на изображение 1024x1024

</td>
<td width="40%">

**Алгоритмы обработки:**
1. 🚀 **Быстрый** - оптимизированная обработка
2. 🎨 **Качественный** - плавные переходы между патчами

</td>
</tr>
</table>

### 📂 Структура проекта

```
📦 satellite-segmentation/
├── 📄 main.py                 # FastAPI backend
├── 🎨 app.py                  # Streamlit frontend  
├── ⚙️ config.py               # Конфигурация
├── 📋 requirements.txt        # Зависимости
├── 🤖 model_f16.tflite       # Модель (скачать отдельно)
└── 📁 utils/                  # Вспомогательные модули
    ├── 🔧 model_loader.py     # Загрузчик модели
    ├── 🎯 prediction.py       # Алгоритмы предсказания
    ├── 🖼️ image_processing.py # Обработка изображений
    └── 🪟 window.py           # Оконные функции
```

## 🔮 Планы развития

- [ ] 🗺️ **QGIS плагин** - интеграция с ГИС
- [ ] 🌐 **Веб-сервис** - облачная версия
- [ ] 📱 **Мобильное приложение**
- [ ] 🚀 **GPU ускорение**
- [ ] 📊 **Дополнительные классы** объектов
- [ ] 🌍 **Поддержка GeoTIFF**


