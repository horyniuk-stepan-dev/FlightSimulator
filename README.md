# FlightSimulator

Генератор відтворюваних відео для побудови багатошарової бази та перевірки
`DroneLocalization` при невідомій висоті, зміні масштабу й нахилі камери.

Симулятор записує чисте відео без HUD, калібрування для слотів бази, точний
ground truth кожного кадру, телеметрію та маніфест із контрольними сумами.

## Швидкий перевірочний запуск

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m simulator.main `
  --mode record `
  --scenario scenarios/ascent_descent.json `
  --geotiff .tile_cache/terrain_692d8e067077_z18.tif `
  --elevation .tile_cache/elevation_d53cb8b9998e_z15.tif `
  --renderer cpu `
  --video-file output/ascent/video.mp4 `
  --calib-file output/ascent/calibration.json `
  --gt-file output/ascent/ground_truth.json `
  --telemetry-file output/ascent/telemetry.csv `
  --manifest-file output/ascent/manifest.json `
  --no-display --fast --yes
```

Якщо `--frame-gt-file` не задано, поруч із відео автоматично створюється
`video.frames.jsonl`. У ньому є рівно один запис на кожний відеокадр:
індекс і час, поза камери, AGL, пряме проєктування центра кадру на поверхню,
GPS, висота поверхні, валідність і частка валідного зображення.

`ground_truth.json` має інше призначення: один запис на слот бази з кроком
`--frame-step`. `calibration.json` містить відібрані якорі у форматі, який
очікує локалізатор. Не використовуйте запитний GT як вхід алгоритму.

Коли задано **і** `--video-file`, **і** `--calib-file`, усе відбувається за один
запуск симулятора. Спочатку він читає чинні параметри відбору з сусіднього
`DroneLocalization`, записує відео, а після закриття MP4 автоматично проходить
його тим самим декодером, маскуванням, локальними ознаками, матчером і
`FrameProcessor`, які використовує побудова бази. Результат записується в
`video.keyframes.json`; калібрувальні якорі отримують **власну** ground-truth
матрицю тільки на відібраних слотах з ознаками. `manifest.json` перевіряє цей
контракт. Окремо запускати локалізатор, створювати базу, вказувати слоти або
повторювати запис не потрібно. Додатковий прохід займає час, але не створює
`database.h5` і не змінює алгоритм локалізатора.

За замовчуванням симулятор знаходить `DroneLocalization` у сусідній папці та
запускає його `.venv` для цього проходу. Якщо проєкти розташовані інакше,
один раз задайте `DRONE_LOCALIZATION_ROOT` і, за потреби,
`DRONE_LOCALIZATION_PYTHON`. Якщо код, моделі чи середовище недоступні, запис
завершиться з явною помилкою замість видачі калібрування з неперевіреними
слотами. Для відтворення тих самих слотів у майбутній базі використовуйте ту
саму версію локалізатора, ваги моделей і його конфігурацію, що діяли під час
запису; параметри відбору збережено у `video.keyframes.json`.

## Пакет базових шарів і запитів

```powershell
& $python -m simulator.batch `
  --out output/scale_benchmark `
  --geotiff .tile_cache/terrain_692d8e067077_z18.tif `
  --elevation .tile_cache/elevation_d53cb8b9998e_z15.tif `
  --reference-altitude 500 `
  --reference-altitude 1000 `
  scenarios/ascent_descent.json `
  scenarios/tilt_only.json `
  scenarios/combined_scale_tilt.json `
  scenarios/layer_boundary_oscillation.json `
  --renderer cpu --no-hillshade
```

Результат розділяється на `references/` і `queries/`. Калібрування та слотний
GT створюються лише для базових шарів; запити містять покадровий GT тільки для
оцінювача. Кожен запуск має власну копію сценарію та `manifest.json`; наявні
або незавершені каталоги не перезаписуються. `--resume` пропускає лише запуски
зі статусом `complete`.

Доступні сценарії:

- `ascent_descent.json` — підйом і повернення між шарами;
- `tilt_only.json` — нахил без зміни висоти;
- `combined_scale_tilt.json` — одночасна зміна масштабу й ракурсу;
- `layer_boundary_oscillation.json` — коливання біля межі шарів.

## Геометричний контракт

- `cpu` — повнорозмірний еталонний renderer; `gpu` вимагає CuPy і завершується
  помилкою, якщо GPU-шлях недоступний; `auto` явно фіксує фактичний backend у
  маніфесті.
- Локальні X/Y — наземні метри з поправкою масштабу Web Mercator у центрі
  карти. Експорт Web Mercator виконується окремим перетворенням.
- Z — висота над базовою площиною симулятора. AGL обчислюється з DEM під
  камерою. Вертикальний datum вхідного DEM симулятор не вгадує.
- Ортофото й DEM зв'язуються через їхні CRS та affine transform. Вихід за DEM,
  `nodata`, горизонт і промені позаду камери не підміняються краєм растра.
- `--elevation-format auto` розрізняє RGB Terrarium і одноканальний DEM у
  метрах; формат можна зафіксувати явно як `terrarium` або `meters`.
- Кожен стан кадру семплюється в `frame_index / fps`; фізичні підкроки точно
  заповнюють цей інтервал і не залежать від швидкості рендера.

Для еталонних наборів використовуйте `--renderer cpu`. GPU-шлях реалізує той
самий контракт, але його числову еквівалентність потрібно перевірити на машині
з CuPy перед використанням у benchmark.

## Перевірки

```powershell
$env:PYTHONUTF8 = "1"
& $python test_renderer_contract.py
& $python test_calibration_logger.py
& $python test_ground_truth_export.py
& $python test_keyframe_predictor.py
& $python test_parallax_calibration.py
& $python -m pytest -q test_database_keyframe_contract.py test_dataset_manifest_keyframes.py
& '..\DroneLocalization\.venv\Scripts\python.exe' -m pytest -q test_database_keyframe_scan.py
```

Детальний план, обґрунтування геометрії та стан реалізації:
[`docs/SIMULATOR_VALIDATION_IMPLEMENTATION_PLAN.md`](docs/SIMULATOR_VALIDATION_IMPLEMENTATION_PLAN.md).
