# Сегментация легких на рентгенограммах с помощью U-net++

**Русский** | [English](#en)

Проект посвящен бинарной сегментации легких на рентгенограммах класса `Normal` из датасета COVID-19 Radiography Database. Модель получает рентгенограмму грудной клетки и предсказывает маску области легких.

В проекте используется архитектура U-net++ - развитие U-net с nested dense skip pathways и deep supervision. Такой подход помогает точнее восстанавливать границы объектов за счет постепенного согласования признаков кодировщика и декодировщика.

Проект оформлен как воспроизводимый notebook pipeline: EDA, подготовка данных, адаптивная конфигурация Colab GPU, построение модели, обучение, подбор threshold, оценка на test-выборке и визуальный анализ предсказаний.

## Ноутбуки

- [Русская версия ноутбука](notebooks/unetpp_segmentation_ru.ipynb)
- [English notebook version](notebooks/unetpp_segmentation_en.ipynb)

## Датасет

Источник данных: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/).

В проекте используется только класс `Normal`:

- изображения: `COVID-19_Radiography_Dataset/Normal/images`;
- маски: `COVID-19_Radiography_Dataset/Normal/masks`.

Особенности данных:

- рентгенограммы имеют исходный размер `299x299`;
- маски имеют исходный размер `256x256`;
- рабочая сетка модели - `256x256`;
- задача является бинарной: `background` и `lung`.

## Pipeline проекта

1. Загрузка датасета через `opendatasets`.
2. Поиск и сопоставление пар `рентгенограмма - маска`.
3. EDA: размеры файлов, доля foreground-пикселей, яркость изображений, визуальная проверка масок.
4. Стратифицированное разбиение на train, validation и test по площади маски.
5. Определение GPU и выбор профиля Colab runtime.
6. Подготовка `tf.data` pipeline с cache, batch-level augmentation и prefetch.
7. Построение U-net++ с deep supervision.
8. Обучение модели с mixed precision, XLA/JIT и наблюдаемостью времени эпох.
9. Подбор threshold по validation Dice.
10. Оценка на test-выборке и визуальная проверка предсказаний.

## Адаптация под Colab GPU

Все GPU-профили используют единую рабочую сетку `256x256`, чтобы результаты были сопоставимыми между разными runtime.

| GPU | System RAM | VRAM | Batch size | Base filters | Mixed precision | XLA/JIT | Cache |
|---|---:|---:|---:|---:|---|---|---|
| T4 | 12.5 GB | 16 GB | 16 | 32 | yes | yes | train |
| L4 | 32 GB | 24 GB | 24 | 40 | yes | yes | train, validation, test |
| A100 | 83 GB | 40 GB | 32 | 48 | yes | yes | train, validation, test |

## Архитектура модели

Модель строится как U-net++ с четырьмя segmentation heads:

- `mask_x01`;
- `mask_x02`;
- `mask_x03`;
- `mask_x04`.

Во время обучения loss считается для всех выходов deep supervision. Основной финальный выход - `mask_x04`.

## Метрики

- **Dice** - основная метрика перекрытия предсказанной и истинной маски.
- **IoU** - отношение пересечения масок к их объединению, более строгая метрика перекрытия.
- **Precision** - доля пикселей, предсказанных как легкие, которые действительно относятся к легким.
- **Recall** - доля настоящих пикселей легких, найденных моделью.
- **Binary accuracy** - доля правильно классифицированных пикселей фона и легких.

## Результаты

Пример результата запуска на A100:

| Metric | Value |
|---|---:|
| Validation best threshold | 0.6 |
| Test Dice | 0.993326 |
| Test IoU | 0.986788 |
| Test precision | 0.993950 |
| Test recall | 0.992747 |
| Test binary accuracy | 0.996870 |

Визуальная проверка показывает, что основные ошибки находятся на тонких границах легких. Крупных ложных областей вне легких в test-примерах не наблюдается.

## Используемые технологии

- Python
- TensorFlow / Keras
- tf.data
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- opendatasets
- Google Colab GPU

## Запуск

1. Открыть русскую или английскую версию ноутбука в Google Colab.
2. Включить GPU runtime.
3. Запустить ноутбук сверху вниз.
4. При загрузке датасета указать Kaggle credentials, если они не настроены заранее.

## Ограничения

Проект не является медицинской диагностической системой. Модель обучается на масках легких класса `Normal` и предназначена для исследовательской демонстрации pipeline сегментации медицинских изображений.

<a id="en"></a>

# Lung segmentation on radiographs with U-net++

[Русский](#сегментация-легких-на-рентгенограммах-с-помощью-u-net) | **English**

This project is dedicated to binary lung segmentation on `Normal`-class radiographs from the COVID-19 Radiography Database. The model receives a chest radiograph and predicts a lung region mask.

The project uses the U-net++ architecture - an extension of U-net with nested dense skip pathways and deep supervision. This approach helps recover object boundaries more accurately by gradually aligning encoder and decoder features.

The project is organized as a reproducible notebook pipeline: EDA, data preparation, adaptive Colab GPU configuration, model construction, training, threshold selection, test evaluation and visual analysis of predictions.

## Notebooks

- [Русская версия ноутбука](notebooks/unetpp_segmentation_ru.ipynb)
- [English notebook version](notebooks/unetpp_segmentation_en.ipynb)

## Dataset

Data source: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/).

The project uses only the `Normal` class:

- images: `COVID-19_Radiography_Dataset/Normal/images`;
- masks: `COVID-19_Radiography_Dataset/Normal/masks`.

Data properties:

- radiographs have the original size of `299x299`;
- masks have the original size of `256x256`;
- the model working grid is `256x256`;
- the task is binary: `background` and `lung`.

## Project pipeline

1. Dataset download through `opendatasets`.
2. Search and matching of `radiograph - mask` pairs.
3. EDA: file sizes, foreground pixel fraction, image brightness, visual mask inspection.
4. Stratified train, validation and test split by mask area.
5. GPU detection and Colab runtime profile selection.
6. Preparation of the `tf.data` pipeline with cache, batch-level augmentation and prefetch.
7. Construction of U-net++ with deep supervision.
8. Model training with mixed precision, XLA/JIT and epoch time observability.
9. Threshold selection by validation Dice.
10. Test-set evaluation and visual inspection of predictions.

## Colab GPU adaptation

All GPU profiles use the same `256x256` working grid so that results remain comparable across different runtime types.

| GPU | System RAM | VRAM | Batch size | Base filters | Mixed precision | XLA/JIT | Cache |
|---|---:|---:|---:|---:|---|---|---|
| T4 | 12.5 GB | 16 GB | 16 | 32 | yes | yes | train |
| L4 | 32 GB | 24 GB | 24 | 40 | yes | yes | train, validation, test |
| A100 | 83 GB | 40 GB | 32 | 48 | yes | yes | train, validation, test |

## Model architecture

The model is built as U-net++ with four segmentation heads:

- `mask_x01`;
- `mask_x02`;
- `mask_x03`;
- `mask_x04`.

During training, the loss is computed for all deep supervision outputs. The main final output is `mask_x04`.

## Metrics

- **Dice** - the main overlap metric between the predicted and ground-truth mask.
- **IoU** - the ratio of mask intersection to mask union, a stricter overlap metric.
- **Precision** - the fraction of pixels predicted as lungs that really belong to lungs.
- **Recall** - the fraction of true lung pixels found by the model.
- **Binary accuracy** - the fraction of correctly classified background and lung pixels.

## Results

Example result from an A100 run:

| Metric | Value |
|---|---:|
| Validation best threshold | 0.6 |
| Test Dice | 0.993326 |
| Test IoU | 0.986788 |
| Test precision | 0.993950 |
| Test recall | 0.992747 |
| Test binary accuracy | 0.996870 |

Visual inspection shows that the main errors are located on thin lung boundaries. No large false regions outside the lungs are observed in the test examples.

## Technologies

- Python
- TensorFlow / Keras
- tf.data
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- opendatasets
- Google Colab GPU

## Running

1. Open the Russian or English notebook version in Google Colab.
2. Enable GPU runtime.
3. Run the notebook from top to bottom.
4. Provide Kaggle credentials during dataset download if they are not configured in advance.

## Limitations

This project is not a medical diagnostic system. The model is trained on `Normal`-class lung masks and is intended as a research demonstration of a medical image segmentation pipeline.
