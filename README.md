# Раннее предупреждение нестабильности в данных воздушного движения

Магистерская выпускная квалификационная работа, НИУ ВШЭ, факультет компьютерных наук, программа «Магистр по наукам о данных», 2026.

Тема: «Система раннего выявления нестабильности в данных воздушного движения: сравнение методов обнаружения аномалий и прототип мониторинга на данных OpenSky Network».

**Студент:** Строкин Никита Алексеевич

**Научный руководитель:** Саночкин Юрий Ильич

**Формат работы:** индивидуальная

## Аннотация

В работе предложен ML-пайплайн обнаружения аномалий в ADS-B траекторных данных, основанный на ансамбле трёх unsupervised-моделей (Isolation Forest, HDBSCAN+GLOSH, LSTM-autoencoder), DQ-aware категоризации событий и Streamlit-прототипе мониторинга. Pipeline применён к набору EUROCONTROL PRC 2024 Data Challenge на основе данных OpenSky Network: 149 миллионов точек телеметрии в 29 788 рейсах за 14 дней ноября 2022 года. Поэтапная обработка переводит сырые ADS-B данные в 26 083 интерпретируемых события у 8 420 рейсов, ранжированных по калиброванному risk-score.

## Ссылки на данные и результаты

Из-за объёма данные хранятся на Google Drive:

| Ссылка | Содержание | Размер |
|---|---|---|
| [OSN-parquets](https://drive.google.com/drive/folders/1K3ttjL0uqezkc4eqjjz12RnKpm0jp2yQ?usp=drive_link) | Исходные ADS-B parquet-файлы (14 дней) | ~11 ГБ |
| [thesis_processed](https://drive.google.com/drive/folders/1ECfCcrWFDuy8a2brGxF0n2WO7BhlL-Vi?usp=drive_link) | Все артефакты pipeline: clean/enriched/annotated parquet, обученные модели, рисунки, таблицы | ~40 ГБ |

PDF выпускной квалификационной работы: [`thesis.pdf`](thesis.pdf) в этом репозитории.

## Структура репозитория

```
air-traffic-instability-early-warning/
├── README.md                   ← этот файл
├── LICENSE                     ← MIT
├── requirements.txt            ← зависимости для dashboard
├── thesis.pdf                  ← полный текст ВКР
├── pipeline/                   ← основной ML-пайплайн (порядок запуска по нумерации)
│   ├── 01_eda_air_traffic_v1.ipynb
│   ├── 02_preprocessing_v3.ipynb
│   ├── 02b_feature_engineering_v4.ipynb
│   ├── 02c_dq_filter_v3.ipynb
│   ├── 03_1_data_prep.ipynb
│   ├── 03_1_contract_patch.ipynb
│   ├── 03_2_isolation_forest.ipynb
│   ├── 03_3_hdbscan.ipynb
│   ├── 03_4_lstm_autoencoder.ipynb
│   ├── 03_5_ensemble_events_evaluation.ipynb
│   └── 04_prepare_dashboard_data.ipynb
├── dashboard/
│   └── 04_dashboard_v3.py      ← Streamlit-прототип мониторинга
└── tables/                     ← CSV-результаты для Главы 3 диплома
    ├── chapter3_phase_distribution.csv
    ├── chapter3_coverage.csv
    ├── chapter3_stability_summary.csv
    ├── chapter3_agreement.csv
    ├── chapter3_events_categories.csv
    ├── chapter3_events_per_phase.csv
    ├── chapter3_events_per_split.csv
    ├── chapter3_risk_levels_by_split.csv
    ├── chapter3_n_events_per_flight.csv
    ├── chapter3_event_duration_stats.csv
    ├── chapter3_phase_coverage.csv
    ├── chapter3_boundary_diagnostic.csv
    └── chapter3_key_numbers.md
```

PNG-рисунки, обученные модели (`.joblib`, `.pt`), parquet-артефакты и диагностические выгрузки хранятся на Google Drive в `thesis_processed/`.

## Pipeline: краткое описание этапов

Скрипты в `pipeline/` пронумерованы в порядке запуска. Каждый этап читает выход предыдущего и сохраняет свой результат в `thesis_processed/` на Google Drive.

| Скрипт | Что делает | Выход |
|---|---|---|
| `01_eda_air_traffic_v1.ipynb` | EDA сырых ADS-B данных, статистика DQ, проверка фильтра EU bbox | EDA-отчёт |
| `02_preprocessing_v3.ipynb` | Очистка: threshold-фильтры, спайки, stale altitude. Первичные DQ-флаги | `european_flights_clean_v3.parquet` |
| `02b_feature_engineering_v4.ipynb` | Вычисление производных, ветра, энергетических признаков; классификация фаз; фазовая нормализация | `european_flights_enriched_v4.parquet` |
| `02c_dq_filter_v3.ipynb` | Финальные три категории DQ-флагов (hard / soft / feature_quality) | `european_flights_annotated_v3.parquet` |
| `03_1_data_prep.ipynb` | Разбиение train/calval/test, обучение StandardScaler, clipping P0.1/P99.9 | `split_metadata.json`, `clip_bounds.json`, `scaler.joblib` |
| `03_1_contract_patch.ipynb` | Фиксация воспроизводимого контракта между моделями | `contract.json` |
| `03_2_isolation_forest.ipynb` | Обучение трёх IF-моделей (seeds 1321/2321/3321), point-level scoring | `if_*.joblib`, `if_scores.parquet`, `if_stability.json` |
| `03_3_hdbscan.ipynb` | Обучение 12 HDBSCAN-моделей (4 фазовых группы × 3 seeds), window-level scoring | `hdb_*.joblib`, `hdb_scores.parquet`, `hdb_stability.json` |
| `03_4_lstm_autoencoder.ipynb` | Обучение двух LSTM-AE (seeds 1321/2321), sequence-level scoring | `lstm_*.pt`, `lstm_scores.parquet`, `lstm_stability.json` |
| `03_5_ensemble_events_evaluation.ipynb` | Ансамбль, calibration risk-score, event extraction, DQ-aware категоризация, оценочный отчёт | `model_scores_points_v3.parquet`, `events_v3.parquet`, `flight_risk_summary_v3.csv`, `evaluation_report_v3.json` |
| `04_prepare_dashboard_data.ipynb` | Подготовка pre-joined артефактов для быстрой загрузки в dashboard | `dashboard_flight_summary.parquet`, `dashboard_points_v3.parquet` |

## Системные требования

### Pipeline (`pipeline/*.ipynb`)

Все скрипты разработаны для **Google Colab Pro** с подключённым Google Drive. Доп. зависимости поверх стандартного Colab Pro-окружения:

```python
!pip install hdbscan traffic
```

Остальные пакеты (`pandas 2.x`, `numpy`, `scipy`, `scikit-learn`, `tensorflow`, `pyarrow`) уже установлены в Colab Pro по умолчанию.

Объём оперативной памяти: достаточно стандартного Colab Pro (~52 ГБ RAM, ~225 ГБ disk).

Времена прогона на Colab Pro (T4 GPU, для LSTM):
- `02_preprocessing_v3.ipynb`: ~25 минут
- `02b_feature_engineering_v4.ipynb`: ~35 минут
- `03_2_isolation_forest.ipynb`: ~15 минут (3 модели)
- `03_3_hdbscan.ipynb`: ~50 минут (12 моделей)
- `03_4_lstm_autoencoder.ipynb`: ~3 часа (2 модели на GPU)
- `03_5_ensemble_events_evaluation.ipynb`: ~20 минут
- `04_prepare_dashboard_data.ipynb`: ~15 минут

### Dashboard (`dashboard/04_dashboard_v3.py`)

Зависимости в `requirements.txt`. Локальный запуск:

```bash
pip install -r requirements.txt
streamlit run dashboard/04_dashboard_v3.py
```

Для работы dashboard'а нужны pre-joined артефакты `dashboard_flight_summary.parquet` и `dashboard_points_v3.parquet` (есть в `thesis_processed/models_v3_artifacts/`). Пути к ним прописаны в верхней части скрипта.

## Воспроизводимость

Pipeline воспроизводим: random seeds зафиксированы в `contract.json` (1321/2321/3321 для IF и HDBSCAN, 1321/2321 для LSTM-AE). Параметры clipping, calibration fraction, dt-thresholds сохранены в том же файле.

Возможны три сценария:

**1. Полное воспроизведение с нуля.** Скачать [OSN-parquets](https://drive.google.com/drive/folders/1K3ttjL0uqezkc4eqjjz12RnKpm0jp2yQ?usp=drive_link), разместить в `/content/drive/MyDrive/`, прогнать скрипты `pipeline/01_*.ipynb → 04_*.ipynb` по порядку. Полное время ~5 часов.

**2. Анализ только результатов.** Скачать содержимое [thesis_processed](https://drive.google.com/drive/folders/1ECfCcrWFDuy8a2brGxF0n2WO7BhlL-Vi?usp=drive_link) и работать с уже посчитанными артефактами и таблицами.

**3. Только dashboard.** Скачать `dashboard_flight_summary.parquet`, `dashboard_points_v3.parquet` и `events_v3.parquet` из `thesis_processed/models_v3_artifacts/`, поправить пути в `dashboard/04_dashboard_v3.py`, запустить локально.

## Данные

Исходный набор — EUROCONTROL Performance Review Commission (PRC) 2024 Data Challenge on Aircraft Take-off Weight Estimation. Изначально подготовлен для задачи оценки взлётной массы воздушного судна; в настоящей работе используется для задачи поиска аномалий и ранних признаков нестабильности.

Содержание датасета:
- ADS-B траекторные данные из OpenSky Network (lat/lon, altitude, groundspeed, track, vertical_rate)
- Метеорологические признаки (ветер, температура, удельная влажность)
- 14 дней ноября 2022 года (с 1 по 14 ноября)
- ~149 миллионов точек телеметрии в 29 788 рейсах европейского воздушного пространства (после проведения предобработки)

## Лицензия

MIT License — см. [LICENSE](LICENSE).

## Контакт

GitHub: [@Diplomundo](https://github.com/Diplomundo)
