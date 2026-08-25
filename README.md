# Learning Frontier UED

Здесь исследуется обучение агента на автоматически подобранных лабиринтах. Сам агент во всех опытах один и тот же: PPO + LSTM из исходного JaxUED. Мы меняем только алгоритм, который решает, какие уровни показывать агенту во время обучения.

Главная трудность в том, что слишком простые лабиринты уже ничему не учат, а слишком сложные почти не дают успешных траекторий. Поэтому нужен дешевый способ находить уровни около текущей границы возможностей агента.

После обучения агент проверяется на восьми нарисованных вручную лабиринтах, которых он раньше не видел. Эти уровни не использовались ни при обучении, ни при выборе формулы, ни как заготовки для генерации.

Полный бюджет одного запуска:

```text
30000 обновлений = 245760000 шагов среды
```

## Наш метод

Для каждого уровня хранится сглаженная частота успеха `p`. Оценка уровня вычисляется так:

```text
score = MNA * (4p(1-p) + 0.25 * max(p_new - p_old, 0))
```

MNA взят из DEGen и применяется только при выборе уровней. Сам PPO по-прежнему использует исходный GAE.

Множитель `4p(1-p)` уменьшает приоритет почти всегда решаемых и почти всегда нерешаемых уровней. Последняя добавка ненадолго сохраняет высокий приоритет у лабиринта, который агент только начал осваивать. Повторный выбор уровней, учет давности и устройство буфера остаются такими же, как в PLR. Мутации ACCEL в итоговом варианте отключены.

## Результат

Основные числа получены отдельной оценкой сохраненных контрольных точек. На каждом из восьми уровней агент сделал по 10 попыток.

| Метод | Число запусков | Доля решенных уровней, среднее ± отклонение |
|---|---:|---:|
| DR | 3 | 0.4417 ± 0.1377 |
| PLR MaxMC | 3 | 0.3625 ± 0.1984 |
| ACCEL MaxMC | 6 | 0.5125 ± 0.2158 |
| MNA + граница обучения | 6 | **0.5500 ± 0.1670** |

Наш метод дал лучшее среднее и меньший разброс, чем ACCEL. Разница небольшая: в парном сравнении одинаковых начальных значений генератора средний выигрыш равен `0.0375`, а 95-процентный интервал для него — `[-0.2953, 0.3703]`. Поэтому по этим шести запускам еще нельзя уверенно сказать, что метод стабильно лучше ACCEL.

Полный разбор по уровням находится в [REPORT.md](REPORT.md), история проверенных идей — в [EXPERIMENTS.md](EXPERIMENTS.md), сырые числа — в [results](results).

## Установка

Рекомендуется Python 3.11. Python 3.14 несовместим с зафиксированной версией JAX.

Для процессора:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-research-cpu.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
```

Для видеокарты с CUDA 12:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-research-gpu.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip check
.venv/bin/python -c "import sys, jax, numpy; print(sys.version); print(jax.__version__, numpy.__version__); print(jax.devices())"
```

В последней строке должен появиться `CudaDevice`, а не `CpuDevice`.

## Проверка кода

```bash
.venv/bin/python scripts/check_experiment_integrity.py
.venv/bin/python -m unittest tests.test_teacher_scores -v
```

Первая команда сравнивает `ActorCritic`, GAE, обновление PPO и его параметры с исходным коммитом JaxUED `0f8f128`.

## Запуск исходных методов

Короткая проверка:

```bash
bash scripts/run_baseline.sh dr 0 250
bash scripts/run_baseline.sh plr 0 250
bash scripts/run_baseline.sh accel 0 250
```

Полное обучение:

```bash
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_baseline.sh dr 0 30000
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_baseline.sh plr 0 30000
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_baseline.sh accel 0 30000
```

Второй аргумент — начальное значение генератора случайных чисел.

## Запуск нашего метода

```bash
CHECKPOINT_SAVE_INTERVAL=20 bash scripts/run_method.sh mna_frontier_lp025_plr 0 30000
```

## Отдельная оценка

```bash
bash scripts/evaluate.sh checkpoints/<название_запуска>/<номер_запуска> -1
```

Сводные таблицы пересчитываются так:

```bash
.venv/bin/python scripts/summarize_results.py
```

## Структура репозитория

```text
examples/maze_plr.py          обучение и логика выбора уровней
src/jaxued/teacher_scores.py  расчет MNA
scripts/run_baseline.sh       запуск DR, PLR и ACCEL
scripts/run_method.sh         запуск новых вариантов
scripts/evaluate.sh           отдельная оценка контрольной точки
scripts/analyze_teacher.py    анализ выбранных уровней
results/                      настройки и метрики запусков
REPORT.md                     итоговый отчет
EXPERIMENTS.md                журнал экспериментов
```

Исходный проект: [DramaCow/jaxued](https://github.com/DramaCow/jaxued), коммит `0f8f128`.
