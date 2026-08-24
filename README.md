# Learning Frontier UED

Эксперименты с teacher для JaxUED Maze. Student во всех запусках один и тот же: PPO + LSTM из исходного `examples/maze_plr.py`. Архитектура, параметры PPO и полный бюджет обучения не менялись.

Полный бюджет одного запуска:

```text
30000 updates = 245760000 шагов среды
```

Восемь готовых уровней SixteenRooms, Labyrinth и StandardMaze используются только для оценки. Они не попадают в генерацию, replay, мутации или обучение teacher.

## Итоговый teacher

Для каждого уровня хранится сглаженная частота успеха `p`. При повторном прохождении она обновляется с коэффициентом 0.3. Приоритет уровня равен

$$
S = MNA(4p(1-p)+0.25max(p_{new}-p_{old},0)).
$$

MNA взят из DEGen и используется только на стороне teacher. Обычный GAE по-прежнему используется для PPO. Множитель `4p(1-p)` понижает приоритет слишком легких и слишком трудных уровней. Последний небольшой член не дает сразу исключить уровень, который student только начал решать.

Replay и учет давности остаются такими же, как в PLR. Финальный вариант не использует мутации ACCEL: в коротких сравнениях они проиграли обычному PLR.

## Что получилось

На 250 updates проведено сравнение трех методов по seeds 0, 1, 2:

| Метод | Доля решенных проверочных уровней |
|---|---:|
| PLR MaxMC | 0.2521 ± 0.0346 |
| PLR MNA | 0.2615 ± 0.0191 |
| MNA + граница обучения | **0.2865 ± 0.0377** |

На 1000 updates пока есть только seed 0:

| Метод | Public | Проверочная выборка |
|---|---:|---:|
| PLR MaxMC | 0.0250 | 0.8469 |
| PLR MNA | 0.0375 | 0.8406 |
| MNA + граница обучения | 0.0250 | **0.8531** |

Это предварительный результат. Он показывает, что метод имеет смысл проверять на полном бюджете, но пока не доказывает превосходство над PLR и ACCEL. Полного запуска на 30000 updates здесь еще нет.

## Установка

CPU:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-research-cpu.txt
.venv/bin/python -m pip install -e . --no-deps
```

GPU с CUDA 12:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "jax[cuda12]==0.4.30"
.venv/bin/python -m pip install -r requirements-research-gpu.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -c "import jax; print(jax.devices())"
```

## Проверки

```bash
.venv/bin/python scripts/check_experiment_integrity.py
.venv/bin/python -m unittest tests.test_teacher_scores -v
```

Первая команда сравнивает `ActorCritic`, GAE, PPO update и зафиксированные параметры с исходным коммитом JaxUED `0f8f128`.

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

## Запуск нового метода

```bash
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_method.sh mna_frontier_lp025_plr 0 30000
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_method.sh mna_frontier_lp025_plr 1 30000
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_method.sh mna_frontier_lp025_plr 2 30000
```

## Оценка checkpoint

```bash
bash scripts/evaluate.sh checkpoints/<название_запуска>/<seed> -1
```

Итоговые таблицы собираются командой

```bash
.venv/bin/python scripts/summarize_results.py
```

## Файлы

```text
examples/maze_plr.py          обучение PLR, ACCEL и новых teacher
src/jaxued/teacher_scores.py  расчет MNA
scripts/run_baseline.sh       запуск DR, PLR и ACCEL
scripts/run_method.sh         запуск новых вариантов
scripts/evaluate.sh           отдельная оценка checkpoint
scripts/analyze_teacher.py    анализ выбранных teacher уровней
REPORT.md                     описание метода и выводы
EXPERIMENTS.md                журнал всех проверенных гипотез
results/                      настройки, метрики и диагностика запусков
```

Исходный проект: [DramaCow/jaxued](https://github.com/DramaCow/jaxued), коммит `0f8f128`.
