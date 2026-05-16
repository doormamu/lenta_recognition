# Структура проекта

```text
lenta-price-recognition/
│
├── app/
│   ├── main.py                     # точка входа приложения
│   ├── ui/                         # интерфейс
│   ├── api/                        # ручки API, если используется FastAPI
│   ├── services/                   # сценарии обработки
│   └── config.py                   # настройки приложения
│
├── cv_module/
│   ├── video/
│   │   ├── reader.py               # чтение видео
│   │   ├── frame_sampler.py        # выбор кадров из видеопотока
│   │   └── quality.py              # оценка резкости, бликов и качества кадра
│   │
│   ├── detection/
│   │   ├── price_tag_detector.py   # поиск ценников на кадрах
│   │   ├── qr_detector.py          # поиск QR-кодов
│   │   └── candidate_merger.py     # объединение найденных кандидатов
│   │
│   ├── recognition/
│   │   ├── ocr_engine.py           # распознавание текста
│   │   ├── barcode_reader.py       # чтение QR-кодов и штрихкодов
│   │   └── field_parser.py         # извлечение структурированных полей
│   │
│   ├── tracking/
│   │   ├── tracker.py              # отслеживание ценников между кадрами
│   │   └── deduplicator.py         # удаление повторно найденных ценников
│   │
│   ├── postprocessing/
│   │   ├── validators.py           # проверки цен, дат, штрихкодов и других полей
│   │   └── field_fusion.py         # объединение данных из OCR и QR-кода
│   │
│   └── export/
│       └── csv_exporter.py         # формирование итогового CSV-файла
│
├── models/
│   ├── detector/                   # модели для детекции ценников
│   └── ocr/                        # модели для OCR
│
├── data/
│   ├── input/                      # входные видео
│   ├── output/                     # результаты обработки
│   └── samples/                    # небольшие примеры для демонстрации
│
├── configs/
│   └── default.yaml                # базовая конфигурация проекта
│
├── tests/                          # тесты
│
├── Dockerfile                      # сборка Docker-образа
├── docker-compose.yml              # локальный запуск через Docker Compose
├── requirements.txt                # Python-зависимости
└── README.md                       # описание проекта и инструкция по запуску


python -m pip install requirements.txt