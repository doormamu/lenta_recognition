lenta-price-recognition/
│
├── app/
│   ├── main.py                     # точка входа приложения
│   ├── ui/                         # интерфейс
│   ├── api/                        # ручки API, если делаем FastAPI
│   ├── services/                   # сценарии обработки
│   └── config.py                   # настройки
│
├── cv_module/
│   ├── video/
│   │   ├── reader.py               # чтение видео
│   │   ├── frame_sampler.py        # выбор кадров
│   │   └── quality.py              # оценка резкости/бликов/качества
│   │
│   ├── detection/
│   │   ├── price_tag_detector.py   # поиск ценников
│   │   ├── qr_detector.py          # поиск QR
│   │   └── candidate_merger.py     # объединение кандидатов
│   │
│   ├── recognition/
│   │   ├── ocr_engine.py           # OCR
│   │   ├── barcode_reader.py       # QR/штрихкод
│   │   └── field_parser.py         # извлечение полей
│   │
│   ├── tracking/
│   │   ├── tracker.py              # отслеживание ценников между кадрами
│   │   └── deduplicator.py         # удаление дублей
│   │
│   ├── postprocessing/
│   │   ├── validators.py           # проверки цены, дат, штрихкодов
│   │   └── field_fusion.py         # объединение OCR + QR
│   │
│   └── export/
│       └── csv_exporter.py         # итоговый CSV
│
├── models/
│   ├── detector/
│   └── ocr/
│
├── data/
│   ├── input/
│   ├── output/
│   └── samples/
│
├── configs/
│   └── default.yaml
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md