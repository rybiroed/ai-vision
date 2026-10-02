# hoopid: постоянные Player ID баскетболистов на видео (прототип 0.1)

Пользователь регистрирует игроков как `PLAYER_01`, `PLAYER_02`, … Система ведёт каждого игрока,
сохраняет его ID при пересечениях и восстанавливает после исчезновения. При сомнении показывает
`UNKNOWN`, а не угадывает.

* Отчёт с метриками, ошибками и ограничениями: **[docs/REPORT.md](docs/REPORT.md)**
* Модели, версии, лицензии: [docs/MODELS_AND_LICENSES.md](docs/MODELS_AND_LICENSES.md)
* Эталонная разметка и её журнал: [gt/README.md](gt/README.md)
* Бизнес-план: [docs/BUSINESS_PLAN.md](docs/BUSINESS_PLAN.md)
* Исходная постановка задачи: [docs/TASK_PROMPT.md](docs/TASK_PROMPT.md)

## Результаты на приложенном видео (54,6 с, тестовая половина 27 с)

| | Базовый вариант | Система | Система + 17 правок |
|---|---|---|---|
| Неверные назначения | 3915 | **0** | 0 |
| Верно подтверждённое видимое время | 79,2% | **90,9%** | 95,1% |
| IDF1 | 0,619 | 0,953 | 0,975 |

Цифры получены на CPU-машине (4 vCPU Xeon), **не на RTX 5080**. Видео короткое, поэтому независимая
проверка недостаточна. Подробности в отчёте.

Готовые выходные файлы лежат в `results/`:

| Файл | Что это |
|---|---|
| `results/system/annotated.mp4` | видео с обозначениями: P01…P09, `?` = UNKNOWN, `x` = незарегистрированный, красный пунктир = LOST (прогноз) |
| `results/system/tracks.csv` | время, координаты, track_id, segment, player_id, статус, score, отрыв, кандидаты; решения онлайн- и второго прохода |
| `results/system/assignments.jsonl` | журнал: старт и конец траекторий, разрезы, назначения с причинами, отзывы, правки второго прохода |
| `results/system/players.json` | реестр игроков на момент прогона |
| `results/baseline/*` | то же для базового варианта |
| `results/system_corrected/*` | после смоделированной ручной коррекции (`corrections.json`) |
| `results/experiments.json` | сравнение, вклад признаков, абляции (настройка и тест) |
| `results/coach/coach_dashboard.mp4` | **панель тренера**: видео + справа по каждому игроку время в кадре, время с мячом, касания, дистанция (ходьба/бег), скорость, мини-карта |
| `results/coach/coach_stats.csv` | итоговая статистика по игрокам (см. §10 отчёта: что это оценки и почему) |
| `results/ocr.json`, `results/precompute_stats.json` | проверка читаемости номеров; скорость и память |

## Установка

Требуются Python 3.10+, ffmpeg/ffprobe в PATH, около 250 МБ под модели.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# GPU (RTX 5080): вместо onnxruntime поставьте onnxruntime-gpu (CUDA 12.x) и проверьте:
#   python -c "import onnxruntime as o; print(o.get_available_providers())"
bash scripts/download_models.sh        # скачивает веса и проверяет SHA-256
```

Ничего из существующего окружения не заменяется: всё ставится в `.venv` внутри проекта.

## Запуск

```bash
V=data/raw/VIDEO-2026-10-02-14-05-11.mp4

# 1. Один потоковый проход: детекции, движение камеры, признаки (кэш на диск, шардами)
python -m hoopid.precompute --video $V --out runs/full/precompute

# 2. Регистрация игроков: веб-интерфейс (клик по игроку → PLAYER_XX, команда, номер, ракурс)
python -m hoopid.ui --video $V --pre runs/full/precompute --registry match/players.json \
    --corrections match/corrections.json --run runs/full/system
# http://127.0.0.1:5000  (вкладки Frames / Players / Review & correct)
#   или из командной строки:
python -m hoopid.register --pre runs/full/precompute --registry match/players.json --video $V \
    add --player PLAYER_01 --frame 300 --x 120 --y 400 --view front

# 3. Трекинг + личности + экспорт (секунды; можно перезапускать после каждой правки)
python -m hoopid.pipeline --pre runs/full/precompute --registry match/players.json \
    --corrections match/corrections.json --out runs/full/system --mode system --video $V
python -m hoopid.pipeline ... --mode baseline        # базовый вариант для сравнения

# 4. Оценка по эталону, подбор порогов (только на настройке), эксперименты
python -m hoopid.evaluate --tracks runs/full/system/tracks.csv --frames 820,1638
python -m hoopid.calibrate        # модель оценки + пороги → configs/calibration.json
python -m hoopid.experiments --out runs/report
```

Полное воспроизведение одной командой: `bash scripts/run_all.sh`. На 4 vCPU это займёт около 12 минут.

## Как устроено

`det_id` (рамка) → `track_id` (короткая траектория) → `segment` (часть траектории между
контактами, пропусками и скачками внешности) → `player_id` (постоянная личность).
Личность решается по сегменту совместным назначением (венгерский алгоритм) с проверкой
отрыва от второго кандидата и запретом одного ID на двух одновременно видимых людях.
Второй проход по записи уточняет ранние фрагменты по более поздним кадрам и сохраняет
исходное решение. Подробно — §3 отчёта.

Конфигурация: `configs/default.json` (детектор, трекер, правила) и `configs/calibration.json`
(веса признаков и пороги, подобранные на кадрах 0–819).

## Структура

```
hoopid/        исходный код
configs/       конфигурация и калибровка
match/         реестр игроков этого матча (players.json) и кропы примеров
gt/            ручная эталонная разметка
data/raw/      оригинальное видео (не перекодировано) и SHA-256
results/       выходные файлы и отчёты экспериментов
docs/          отчёт, лицензии, бизнес-план, иллюстрации
scripts/       загрузка моделей, полный прогон
```
