# Learning Frontier UED

Исследование teacher для JaxUED Maze. Student во всех запусках зафиксирован: исходный PPO + LSTM из `examples/maze_plr.py`. Архитектура, параметры PPO и бюджет обучения не менялись.

Полный бюджет одного запуска: `30000 updates = 245760000 шагов среды`.

Восемь public уровней SixteenRooms, Labyrinth и StandardMaze используются только для оценки. Они не участвуют в обучении, генерации, replay, мутациях или выборе метода.

## Метод

Для каждого уровня teacher хранит сглаженную частоту успеха `p`. Приоритет replay вычисляется так:

```text
score = MNA * (4p(1-p) + 0.25 * max(p_new - p_old, 0))
```

MNA взят из DEGen и используется только на стороне teacher. PPO по-прежнему использует обычный GAE. Множитель `4p(1-p)` понижает приоритет крайних по сложности уровней, а небольшая положительная добавка сохраняет уровни, которые student только начал осваивать. Replay, ранжирование и staleness остаются такими же, как в PLR. Финальный вариант не использует мутации ACCEL.

## Итог

Основной результат получен отдельной оценкой сохраненных checkpoint на восьми public уровнях, по 10 попыток на уровень.

| Метод | Seeds | Solve rate, среднее ± std |
|---|---:|---:|
| DR | 3 | 0.4417 ± 0.1377 |
| PLR MaxMC | 3 | 0.3625 ± 0.1984 |
| ACCEL MaxMC | 6 | 0.5125 ± 0.2158 |
| MNA + граница обучения | 6 | **0.5500 ± 0.1670** |

Новый teacher дал самое высокое среднее и меньший разброс, чем ACCEL. Средняя парная разница по seeds 0–5 равна `+0.0375`, но ее 95% интервал `[-0.2953, 0.3703]` включает ноль. То есть результат лучше, но шести seeds пока мало для уверенного вывода о превосходстве над ACCEL.

Подробные результаты по уровням и все ограничения находятся в [REPORT.md](REPORT.md), журнал всех проверенных гипотез — в [EXPERIMENTS.md](EXPERIMENTS.md).

## Установка

Рекомендуется Python 3.11. Python 3.14 несовместим с зафиксированным `jaxlib 0.4.30`.

CPU:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-research-cpu.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
```

GPU с CUDA 12:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-research-gpu.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
.venv/bin/python -c "import sys, jax, numpy; print(sys.version); print(jax.__version__, numpy.__version__); print(jax.devices())"
```

В последней строке GPU-установки должен быть `CudaDevice`, а не `CpuDevice`.

## Проверки

```bash
.venv/bin/python scripts/check_experiment_integrity.py
.venv/bin/python -m unittest tests.test_teacher_scores -v
```

Первая команда сравнивает `ActorCritic`, GAE, PPO update и параметры student с исходным коммитом JaxUED `0f8f128`.

## Запуск baseline

Короткая проверка:

```bash
bash scripts/run_baseline.sh dr 0 250
bash scripts/run_baseline.sh plr 0 250
bash scripts/run_baseline.sh accel 0 250
```

Полный бюджет:

```bash
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_baseline.sh dr 0 30000
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_baseline.sh plr 0 30000
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_baseline.sh accel 0 30000
```

## Запуск метода

```bash
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_method.sh mna_frontier_lp025_plr 0 30000
```

Seed задается вторым аргументом. Для итогового сравнения использовались seeds 0–5 для нового метода и ACCEL.

## Оценка checkpoint

```bash
bash scripts/evaluate.sh checkpoints/<название_запуска>/<seed> -1
```

Сводные таблицы пересчитываются командой:

```bash
.venv/bin/python scripts/summarize_results.py
```

## Структура

```text
examples/maze_plr.py          обучение PLR, ACCEL и новых teacher
src/jaxued/teacher_scores.py  расчет MNA
scripts/run_baseline.sh       запуск DR, PLR и ACCEL
scripts/run_method.sh         запуск вариантов teacher
scripts/evaluate.sh           отдельная оценка checkpoint
scripts/analyze_teacher.py    анализ выбранных уровней
results/                      компактные настройки и метрики запусков
REPORT.md                     итоговый отчет
EXPERIMENTS.md                журнал экспериментов
```

Checkpoint и большие диагностические массивы намеренно не хранятся в Git. Исходный проект: [DramaCow/jaxued](https://github.com/DramaCow/jaxued), коммит `0f8f128`.
