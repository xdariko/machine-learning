# Лабораторная 2 — отчёт и объяснение

## Цель и суть работы

Работа учит отличать настоящую структуру в данных от картинки, которую рисует чистый шум. Когда признаков почти столько же, сколько наблюдений, даже абсолютно случайные данные выглядят так, будто в них есть «главные компоненты». Теория случайных матриц говорит, как именно выглядит спектр шума, и всё, что выходит за его рамки, можно считать сигналом.

Три задания — три шага одной идеи:

| Задание | Вопрос | Ответ, который получаем |
| --- | --- | --- |
| 1. Закон Марченко–Пастура | Как выглядит спектр чистого шума? | Собственные числа размазаны по отрезку [λ-, λ+], а не собраны в точке 1 |
| 2. Фазовый переход BBP | Когда в шуме можно увидеть спрятанный сигнал? | Только если сила сигнала θ больше порога √(p/n); иначе PCA слеп |
| 3. Диагностика ResNet-18 | Можно ли по весам понять, хорошо ли обучен слой? | Да: шум даёт балк МП, обучение — тяжёлый хвост, деградация — схлопывание спектра |

Зачем это нужно на практике:

- **PCA и снижение размерности.** Не все главные компоненты значимы: те, что внутри шумового отрезка, — артефакт выборки.
- **Оценка силы сигнала.** Наблюдаемая дисперсия по главной компоненте всегда завышена, и её можно исправить формулой.
- **Аудит нейросетей.** Качество слоя видно по его весам без тестовых данных.

## Теория

### Случайная величина и её распределение

**Случайная величина** — число, значение которого заранее неизвестно и определяется случаем: результат броска кубика, показание шумного датчика, число из `random_gen.normal()`. Одно конкретное значение ничего не говорит, важно **распределение** — какие значения и как часто величина принимает.

- **Плотность** ρ(x): площадь под кривой на отрезке [a, b] равна вероятности попасть в этот отрезок. Вся площадь равна 1. Гистограмма с `density=True` — эмпирическая оценка плотности.
- **Функция распределения** F(x) = P(X ≤ x) и **функция выживания (CCDF)** P(X ≥ x) = 1 − F(x). Их можно построить по выборке без бинов (задание 3).
- **Среднее** E X — центр распределения; **дисперсия** Var X = E(X − E X)² — разброс вокруг центра; **стандартное отклонение** std = √Var.
- **Нормальное распределение** N(0, σ²) — колокол со средним 0 и дисперсией σ². В работе им моделируется **белый шум**: независимые величины без какой-либо структуры.
- **Выборка и оценка.** Истинные среднее и дисперсию мы не знаем и оцениваем по n наблюдениям. Оценка — тоже случайная величина: новая выборка даст другое число. Поэтому в заданиях 1.3 и 2.3 мы повторяем опыт несколько раз и смотрим на среднее и разброс.

### Случайный вектор, ковариация и PCA

Одно наблюдение с p признаками — **случайный вектор** из p случайных величин. Их связь описывает **ковариационная матрица** Σ размера p×p: на диагонали дисперсии признаков, вне диагонали — ковариации пар. У белого шума Σ = σ²I: все дисперсии одинаковы, связей нет.

Сложим n наблюдений столбцами в матрицу X (p×n). Оценка ковариации по выборке:

$$
\hat\Sigma = \frac{1}{n} X X^T
$$

**Собственные числа и векторы.** Для единичного направления u число uᵀΣ̂u — выборочная дисперсия данных вдоль u. Собственные векторы — особые направления, собственные числа — дисперсии вдоль них. Старший собственный вектор — направление наибольшего разброса; это и есть **первая главная компонента (PCA)**:

$$
\hat\lambda_1 = \max_{\|u\|=1} u^T \hat\Sigma u
$$

**Спектр** — набор всех собственных чисел. Если данные случайны, то и Σ̂ — **случайная матрица**, а её собственные числа — случайные величины со своим распределением. Гистограмма всех собственных чисел называется **ESD** (empirical spectral distribution).

**Сингулярное разложение (SVD).** Любая матрица X = U·diag(s)·Vᵀ. Тогда XXᵀ = U·diag(s²)·Uᵀ, т.е. собственные числа ковариации = квадраты сингулярных чисел, делённые на n. Это позволяет не строить ковариацию, а функция `svds(k=1)` находит только старшее число — быстрее полного разложения.

### Закон Марченко–Пастура (МП)

Классическая статистика: при n → ∞ и фиксированном p оценка Σ̂ сходится к Σ. В машинном обучении p и n сопоставимы, и важно их отношение γ = p/n. Тогда спектр чистого шума (истинные собственные числа все равны σ²) не собирается в точке σ², а размазывается в «балк» с плотностью:

$$
\rho_{MP}(\lambda)=\frac{\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}}{2\pi\sigma^2\gamma\lambda},\qquad \lambda_\pm=\sigma^2(1\pm\sqrt\gamma)^2
$$

При p > n (γ > 1) матрица Σ̂ имеет ранг не выше n, и ровно p − n собственных чисел равны нулю. Это учитывает дельта-функция в полной мере:

$$
d\mu_{MP} = \Big(1-\frac{1}{\gamma}\Big)^+\delta_0 + \rho_{MP}(\lambda)\,d\lambda, \qquad \int_{\lambda_-}^{\lambda_+}\rho_{MP}\,d\lambda = \min\Big(1, \frac{1}{\gamma}\Big)
$$

### Распределение Трейси–Видома

Наибольшее собственное число шума — тоже случайная величина. В среднем оно идёт к λ+, а его отклонения имеют размер порядка n^(−2/3) и подчиняются распределению Трейси–Видома TW₁: среднее ≈ −1.21, std ≈ 1.27, 99%-квантиль ≈ 2.02. Распределение несимметрично: сдвинуто влево, но с длинным правым хвостом.

$$
\hat\lambda_{\max} \approx \lambda_+ + \sigma_n \cdot TW_1, \qquad \sigma_n = 2^{4/3} n^{-2/3} \ (\gamma = 1)
$$

### Фазовый переход BBP

Добавим в шум один сигнал силы θ вдоль направления v₁: Σ = I + θ·v₁v₁ᵀ («модель одиночного шипа»). Истинная дисперсия вдоль v₁ равна 1 + θ. Теорема Baik–Ben Arous–Péché: есть порог θc = √γ.

$$
\hat\lambda_1 \to \begin{cases}(1+\sqrt\gamma)^2 = \lambda_+, & \theta \le \theta_c\\(1+\theta)(1+\gamma/\theta), & \theta > \theta_c\end{cases} \qquad \langle\hat v_1, v_1\rangle^2 \to \begin{cases}0, & \theta \le \theta_c\\\dfrac{1-\gamma/\theta^2}{1+\gamma/\theta}, & \theta > \theta_c\end{cases}
$$

- Ниже порога сигнал тонет в шуме: старшее число сидит на краю балка, а найденное направление случайно (PCA «слеп»).
- Выше порога сигнал виден, но λ̂₁ завышает 1 + θ, а вектор скошен. Обратная формула позволяет восстановить θ:

$$
\hat\theta = \frac{\hat\lambda_1 + 1 - \gamma + \sqrt{(\hat\lambda_1 + 1 - \gamma)^2 - 4\hat\lambda_1}}{2} - 1
$$

### Тяжёлые хвосты и HTSR (Мартин–Махони)

У необученной (случайной) матрицы весов спектр — балк МП. Обучение SGD создаёт корреляции, и спектр отращивает **степенной (тяжёлый) хвост** ρ(λ) ∝ λ^(−α) при λ ≥ λ_min. «Тяжёлый» значит, что большие значения встречаются гораздо чаще, чем у нормального распределения. В log-log масштабе степенной закон — прямая, а CCDF имеет наклон −(α − 1).

| α | Режим |
| --- | --- |
| хвоста нет или α > 4 | слабообученный слой, преобладает шум |
| 2 < α ≤ 4 | «зона Златовласки»: оптимальная неявная регуляризация |
| α ≤ 2 или вырожденный спектр | коллапс ранга / переобучение |

### Как оценить α

- **МНК по гистограмме** — провести прямую через точки (ln λ, ln ρ). Ненадёжно: результат зависит от числа бинов, ошибки в редких бинах огромные, пустые бины приходится выбрасывать.
- **Метод максимального правдоподобия (MLE)** — выбрать α, при котором наблюдаемые числа наиболее вероятны. Для степенного закона ответ в явном виде — оценка Хилла:

$$
\alpha = 1 + N_{tail}\Big[\sum_{i=1}^{N_{tail}} \ln\frac{\lambda_i}{\lambda_{\min}}\Big]^{-1}
$$

- **Статистика Колмогорова–Смирнова** D_KS = max |F_эмп(λ) − F_теор(λ)| — наибольшее расхождение двух функций распределения. Начало хвоста λ_min выбираем так, чтобы D_KS была минимальной.
- **Эффективный (стабильный) ранг** srank = Σλᵢ / λ_max — сколько направлений реально несут энергию. Одно доминирующее направление даёт srank ≈ 1.

## Разбор кода по ячейкам

В ноутбуке `2.ipynb` 18 ячеек с кодом, и в самом коде нет ни комментариев, ни пояснений — всё объяснение здесь. Ячейки пронумерованы по порядку (считаются только ячейки с кодом). Для каждой приведён код целиком и разобрано, что делает каждая строка.

### Ячейка 1. Импорты и настройки

```python
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
random_gen = np.random.default_rng(42)
```

- `time` — замер времени работы (сравниваем быстрые и медленные способы).
- `numpy` (`np`) — массивы, матрицы, линейная алгебра (`eigvalsh`, `eigh`, `svd`).
- `matplotlib.pyplot` (`plt`) — графики.
- `svds` — находит только **k старших** сингулярных чисел (и векторов) матрицы итерационным методом Ланцоша. Нужен, когда полное разложение не требуется: так намного быстрее.
- `plt.rcParams.update(...)` — общие настройки всех графиков: чёткость 110 dpi, сетка, полупрозрачные линии сетки.
- `random_gen = np.random.default_rng(42)` — генератор случайных чисел. Число 42 — **seed** (зерно): с ним последовательность «случайных» чисел каждый раз одна и та же, и результаты воспроизводятся при перезапуске.

### Ячейка 2. Функции для закона Марченко–Пастура (задание 1, п. 1)

```python
def noise_eigenvalues(p, n, sigma2=1.0):
    X = random_gen.normal(loc=0.0, scale=np.sqrt(sigma2), size=(p, n))
    sample_cov = X @ X.T / n
    return np.linalg.eigvalsh(sample_cov)


def mp_bounds(gamma, sigma2=1.0):
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    return lam_minus, lam_plus


def mp_density(lam, gamma, sigma2=1.0):
    lam_minus, lam_plus = mp_bounds(gamma, sigma2)
    lam = np.asarray(lam, dtype=float)
    density = np.zeros_like(lam)
    inside = (lam > lam_minus) & (lam < lam_plus)
    x = lam[inside]
    density[inside] = np.sqrt((lam_plus - x) * (x - lam_minus)) / (2 * np.pi * sigma2 * gamma * x)
    return density
```

**`noise_eigenvalues(p, n, sigma2)`** — то, что требует п. 1 задания 1.

- `random_gen.normal(loc=0.0, scale=np.sqrt(sigma2), size=(p, n))` — матрица X размера p×n, каждый элемент — независимая нормальная случайная величина N(0, σ²). `scale` — это стандартное отклонение, поэтому передаём √σ². Это и есть **белый шум**: строки — признаки, столбцы — наблюдения.
- `sample_cov = X @ X.T / n` — выборочная ковариация Σ̂ = (1/n)·X·Xᵀ, матрица p×p. `@` — матричное умножение, `.T` — транспонирование.
- `np.linalg.eigvalsh(sample_cov)` — собственные числа. Буква `h` (Hermitian) означает «для симметричных матриц»: Σ̂ симметрична, поэтому этот метод быстрее и точнее общего `eigvals`, и он гарантированно возвращает вещественные числа, **отсортированные по возрастанию**.

**`mp_bounds(gamma, sigma2)`** — границы шумового балка по формулам λ- = σ²(1 − √γ)², λ+ = σ²(1 + √γ)². Возвращает пару чисел.

**`mp_density(lam, gamma, sigma2)`** — теоретическая плотность ρ_MP(λ) в точках `lam`.

- `np.asarray(lam, dtype=float)` — превращаем вход (число, список или массив) в массив numpy.
- `density = np.zeros_like(lam)` — массив нулей той же формы: вне отрезка [λ-, λ+] плотность и должна быть нулём.
- `inside = (lam > lam_minus) & (lam < lam_plus)` — логическая маска: True там, где точка внутри отрезка. `&` — поэлементное «и».
- `x = lam[inside]` — только внутренние точки.
- Следующая строка — формула плотности, записанная только во внутренние точки (`density[inside] = ...`). Внешние точки не трогаем: там под корнем было бы отрицательное число.

### Ячейка 3. Эксперимент в двух режимах (задание 1, пп. 2–3)

```python
n = 1000
regimes = {"A (полный ранг)": 500, "B (дефект ранга)": 2000}
table_rows = []

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (regime_name, p) in zip(axes, regimes.items()):
    gamma = p / n
    eigenvalues = noise_eigenvalues(p, n)

    zero_tol = 1e-10 * eigenvalues.max()
    nonzero = eigenvalues[eigenvalues > zero_tol]
    n_zeros = len(eigenvalues) - len(nonzero)

    lam_minus, lam_plus = mp_bounds(gamma)
    scale = gamma if gamma > 1 else 1.0

    lam_grid = np.linspace(0.01, lam_plus + 0.3, 1000)
    ax.hist(nonzero, bins=60, density=True, alpha=0.55, label="гистограмма собственных чисел")
    ax.plot(lam_grid, scale * mp_density(lam_grid, gamma), "r", lw=2, label=f"теория МП × {scale:g}")
    ax.axvline(lam_minus, color="k", ls="--", lw=1, label=r"границы $\lambda_-$, $\lambda_+$")
    ax.axvline(lam_plus, color="k", ls="--", lw=1)
    ax.set_title(f"Режим {regime_name}: p={p}, n={n}, γ={gamma:g}")
    ax.set_xlabel(r"$\lambda$")
    ax.legend()

    table_rows.append((regime_name, gamma, lam_minus, nonzero.min(), lam_plus, nonzero.max(), n_zeros))
    if n_zeros > 0:
        zeros = eigenvalues[eigenvalues <= zero_tol]
        print(f"{regime_name}: нулевых собственных чисел {n_zeros} (ожидалось p - n = {p - n}); "
              f"самый большой |ноль| = {np.abs(zeros).max():.1e}; отрицательных среди них: {(zeros < 0).sum()}")
plt.tight_layout()
plt.show()
```

- `n = 1000` — число наблюдений, фиксировано заданием.
- `regimes` — словарь «название режима → число признаков p»: режим A (p = 500, γ = 0.5) и режим B (p = 2000, γ = 2).
- `table_rows = []` — сюда складываем строки для итоговой таблицы (ячейка 4).
- `plt.subplots(1, 2, ...)` — рисунок из двух графиков рядом; `axes` — список из двух осей.
- `for ax, (regime_name, p) in zip(axes, regimes.items())` — идём по режимам параллельно с осями: первый режим рисуется на левом графике, второй на правом.
- `gamma = p / n` — отношение γ.
- `eigenvalues = noise_eigenvalues(p, n)` — спектр шума.

**Отсечение машинных нулей (методическое предостережение 1).**

- `zero_tol = 1e-10 * eigenvalues.max()` — порог. При p > n у матрицы Σ̂ ранг не больше n, значит ровно p − n собственных чисел **равны нулю**. Но компьютер считает с округлением (машинная точность ≈ 2.2·10⁻¹⁶), поэтому вместо нулей получаются числа порядка ±10⁻¹⁵, примерно половина из них отрицательные. Условие `eigs > 0` пропустило бы положительный мусор. Порог 10⁻¹⁰·λmax ≈ 6·10⁻¹⁰ в 10⁵ раз выше мусора и в 10⁸ раз ниже самого маленького настоящего числа (λ- ≈ 0.17), поэтому разделяет их надёжно.
- `nonzero = eigenvalues[eigenvalues > zero_tol]` — оставляем только настоящие собственные числа (булева индексация).
- `n_zeros = ...` — сколько отбросили; для проверки должно быть p − n.

**Множитель масштаба (методическое предостережение 2).**

- `scale = gamma if gamma > 1 else 1.0` — при γ > 1 умножаем теорию на γ. Обоснование: полная мера dμ_MP = (1 − 1/γ)⁺δ₀ + ρ_MP dλ имеет массу 1. При γ > 1 на атом в нуле приходится 1 − 1/γ, значит интеграл непрерывной части ∫ρ_MP dλ = 1/γ (для γ = 2 это 0.5). А гистограмма с `density=True` по одним ненулевым числам всегда имеет площадь 1. Сравнивать нужно условную плотность «при условии λ ≠ 0»: ρ_MP / (1/γ) = γ·ρ_MP. При γ ≤ 1 нулей нет, множитель 1.

**График.**

- `lam_grid = np.linspace(0.01, lam_plus + 0.3, 1000)` — 1000 точек, в которых рисуем теоретическую кривую, с небольшим запасом справа от λ+.
- `ax.hist(nonzero, bins=60, density=True, ...)` — эмпирическая плотность: 60 столбиков, высоты нормированы так, чтобы площадь была 1. `alpha=0.55` — полупрозрачность.
- `ax.plot(lam_grid, scale * mp_density(...), "r", lw=2, ...)` — красная теоретическая кривая с множителем.
- `ax.axvline(..., ls="--")` — две вертикальные пунктирные линии λ- и λ+ (требование задания).
- `set_title`, `set_xlabel`, `legend` — подписи. Строки `r"$\lambda$"` — LaTeX-формулы в подписях; `r` перед строкой отключает обработку обратных слешей.

**Сбор данных и печать.**

- `table_rows.append(...)` — запоминаем: имя режима, γ, теоретические λ-, экспериментальный минимум `nonzero.min()`, теоретическое λ+, экспериментальный максимум `nonzero.max()`, число нулей.
- Блок `if n_zeros > 0:` срабатывает только в режиме B: печатает число нулей, самый большой по модулю «ноль» и сколько из них отрицательных — доказательство, что `eigs > 0` не годится.
- `plt.tight_layout()` убирает наложение подписей, `plt.show()` выводит рисунок.

### Ячейка 4. Таблица теоретических и экспериментальных границ (задание 1, п. 4)

```python
print(f"{'Режим':<18}{'γ':>4}{'λ- теор':>10}{'λmin эксп':>11}{'λ+ теор':>10}{'λmax эксп':>11}{'нулей':>7}")
for regime_name, gamma, lam_minus, lam_min_exp, lam_plus, lam_max_exp, n_zeros in table_rows:
    print(f"{regime_name:<18}{gamma:>4g}{lam_minus:>10.4f}{lam_min_exp:>11.4f}"
          f"{lam_plus:>10.4f}{lam_max_exp:>11.4f}{n_zeros:>7}")
```

- Первая строка печатает заголовок таблицы. В f-строке `{'Режим':<18}` — выровнять текст по левому краю в поле ширины 18, `:>10` — по правому краю в поле ширины 10.
- Цикл распаковывает каждый кортеж из `table_rows` в именованные переменные и печатает строку. `:.4f` — 4 знака после запятой, `:g` — короткий формат числа.

### Ячейка 5. Флуктуации наибольшего собственного числа (задание 1, вопрос про λmax)

```python
def largest_noise_eigenvalue(p, n):
    X = random_gen.standard_normal((p, n))
    largest_singular_value = svds(X, k=1, return_singular_vectors=False)[0]
    return largest_singular_value ** 2 / n


sizes = [100, 200, 400, 800, 1600]
n_repeats = 40

print("Прогноз Трейси–Видома: n^(2/3)·(среднее - 4) ≈ -3.04,  n^(2/3)·std ≈ 3.20\n")
print(f"{'n = p':>6}{'среднее λmax':>14}{'std':>8}{'n^(2/3)·(среднее-4)':>21}{'n^(2/3)·std':>13}{'доля λmax > 4':>15}")
for size in sizes:
    lam_max_samples = np.array([largest_noise_eigenvalue(size, size) for _ in range(n_repeats)])
    scale = size ** (2 / 3)
    print(f"{size:>6}{lam_max_samples.mean():>14.4f}{lam_max_samples.std():>8.4f}"
          f"{scale * (lam_max_samples.mean() - 4):>21.2f}{scale * lam_max_samples.std():>13.2f}"
          f"{(lam_max_samples > 4).mean():>15.2f}")
```

Эта ячейка — дополнительный эксперимент, который подкрепляет ответ на вопрос задания «почему λ̂max правее λ+ и каким распределением описываются флуктуации».

- `largest_noise_eigenvalue(p, n)` генерирует шум N(0, 1) и находит **только** наибольшее собственное число Σ̂. Используется связь с SVD: если X = U·diag(s)·Vᵀ, то (1/n)XXᵀ имеет собственные числа s²/n. `svds(X, k=1, return_singular_vectors=False)` возвращает массив из одного старшего сингулярного числа, `[0]` берёт его.
- `sizes` — размеры матриц (p = n, т.е. γ = 1, λ+ = 4); `n_repeats = 40` — сколько раз повторяем опыт для каждого размера.
- Первые два `print` — прогноз теории и заголовок таблицы. Прогноз: λ̂max ≈ 4 + 2^(4/3)·n^(−2/3)·TW₁, а у TW₁ среднее −1.21 и std 1.27; 2^(4/3)·(−1.21) ≈ −3.04, 2^(4/3)·1.27 ≈ 3.20.
- `lam_max_samples = np.array([... for _ in range(n_repeats)])` — 40 независимых значений случайной величины λ̂max (list comprehension; `_` — переменная цикла, которая не нужна).
- `scale = size ** (2/3)` — множитель n^(2/3).
- В печати: среднее и std по 40 опытам; те же величины, умноженные на n^(2/3) (если теория верна, они не зависят от n); `(lam_max_samples > 4).mean()` — доля опытов, где максимум оказался правее λ+ (среднее от массива True/False = доля True).

### Ячейка 6. Параметры и генерация данных с сигналом (задание 2, пп. 1–2)

```python
n_features = 1500
n_samples = 3000
gamma2 = n_features / n_samples
theta_c = np.sqrt(gamma2)
lam_minus2, lam_plus2 = mp_bounds(gamma2)
print(f"γ = {gamma2}, θc = {theta_c:.4f}, λ- = {lam_minus2:.4f}, λ+ = {lam_plus2:.4f}")


def make_spiked_data(theta):
    Y = random_gen.standard_normal((n_features, n_samples))
    Y[0] *= np.sqrt(1 + theta)
    return Y


def top_eigenpair(Y):
    U, S, _ = svds(Y / np.sqrt(Y.shape[1]), k=1)
    return S[0] ** 2, U[:, 0]
```

- `n_features = 1500` (p) и `n_samples = 3000` (n) — параметры из задания. Отсюда `gamma2` = 0.5, порог `theta_c` = √γ ≈ 0.7071 и границы балка `lam_minus2`, `lam_plus2`. Суффикс `2` — чтобы не путать с переменными задания 1.
- `print` — проверка, что числа совпадают с условием (0.0858 и 2.9142).

**`make_spiked_data(theta)`** — модель одиночного шипа Y = (I + θ·v₁v₁ᵀ)^(1/2)·X, v₁ = e₁.

- `Y = random_gen.standard_normal(...)` — шум X размера 1500×3000 из N(0, 1).
- `Y[0] *= np.sqrt(1 + theta)` — умножаем первую строку на √(1+θ). Матрица (I + θe₁e₁ᵀ) диагональная: diag(1+θ, 1, …, 1), её корень diag(√(1+θ), 1, …, 1). Умножение слева на диагональную матрицу масштабирует строки, поэтому достаточно изменить первую строку и не строить матрицу 1500×1500 (оптимизация из задания). В итоге дисперсия первого признака 1 + θ, остальных 1.

**`top_eigenpair(Y)`** — старшая пара (λ̂₁, v̂₁) матрицы Σ̂ = (1/n)·Y·Yᵀ (совет по производительности из задания).

- `Y / np.sqrt(Y.shape[1])` — делим на √n, `Y.shape[1]` — число столбцов (наблюдений).
- `svds(..., k=1)` возвращает `U` (левые сингулярные векторы), `S` (сингулярные числа) и `Vt` (правые; не нужны, поэтому `_`).
- Если Y/√n = U·S·Vᵀ, то (1/n)YYᵀ = U·S²·Uᵀ. Значит λ̂₁ = `S[0] ** 2`, а собственный вектор v̂₁ = первый столбец `U[:, 0]`.

### Ячейка 7. Ловушка с порядком собственных чисел и сравнение скорости (задание 2, предостережение)

```python
Y = make_spiked_data(theta=2.0)

start = time.perf_counter()
all_eigenvalues, all_eigenvectors = np.linalg.eigh(Y @ Y.T / n_samples)
time_eigh = time.perf_counter() - start

start = time.perf_counter()
lam1, v1_hat = top_eigenpair(Y)
time_svds = time.perf_counter() - start

print(f"eigh, элемент [0]  (ошибка): λ = {all_eigenvalues[0]:.4f}, <v,e1>² = {all_eigenvectors[0, 0] ** 2:.2e}  <- это λ_min!")
print(f"eigh, элемент [-1] (верно):  λ = {all_eigenvalues[-1]:.4f}, <v,e1>² = {all_eigenvectors[0, -1] ** 2:.4f}")
print(f"svds, k=1:                   λ = {lam1:.4f}, <v,e1>² = {v1_hat[0] ** 2:.4f}")
print(f"Время: eigh {time_eigh:.2f} c, svds {time_svds:.2f} c")
```

- `Y = make_spiked_data(theta=2.0)` — одни данные с заметным сигналом.
- `time.perf_counter()` — точный таймер; разность двух вызовов — время выполнения кода между ними.
- `np.linalg.eigh(Y @ Y.T / n_samples)` — полное разложение ковариации 1500×1500: все 1500 собственных чисел (по **возрастанию**) и матрица собственных векторов, где **столбец** i соответствует числу i.
- `top_eigenpair(Y)` — то же старшее число быстрым способом.
- Печать:
    - `all_eigenvalues[0]`, `all_eigenvectors[:, 0]` — первый элемент «в лоб». Это **самое маленькое** число (≈ λ-), и его вектор не связан с сигналом — ошибка, о которой предупреждает задание.
    - `all_eigenvalues[-1]`, `all_eigenvectors[:, -1]` — последний элемент, т.е. наибольший. Это правильный срез.
    - ⟨v, e₁⟩² вычисляется как `v[0] ** 2`: скалярное произведение с e₁ = (1, 0, …, 0) равно первой координате вектора. `all_eigenvectors[0, -1]` — первая координата последнего столбца.
    - Время обоих способов.

### Ячейка 8. Теория BBP и перебор силы сигнала (задание 2, п. 2)

```python
def bbp_lambda_theory(theta, gamma):
    if theta <= np.sqrt(gamma):
        return (1 + np.sqrt(gamma)) ** 2
    return (1 + theta) * (1 + gamma / theta)


def bbp_overlap_theory(theta, gamma):
    if theta <= np.sqrt(gamma):
        return 0.0
    return (1 - gamma / theta ** 2) / (1 + gamma / theta)


thetas = np.round(np.arange(0.1, 2.51, 0.1), 1)
n_runs = 10
lam1_results = np.zeros((len(thetas), n_runs))
overlap_results = np.zeros((len(thetas), n_runs))

start = time.perf_counter()
for i, theta in enumerate(thetas):
    for run in range(n_runs):
        lam1, v1_hat = top_eigenpair(make_spiked_data(theta))
        lam1_results[i, run] = lam1
        overlap_results[i, run] = v1_hat[0] ** 2
print(f"Готово за {time.perf_counter() - start:.0f} c")
```

- **`bbp_lambda_theory`** — предел λ̂₁, записанный **кусочно**, как требует предостережение к теоретическим графикам. При θ ≤ θc возвращает край балка λ+ = (1 + √γ)², иначе — формулу (1 + θ)(1 + γ/θ). Если считать формулу «в лоб» везде, то при θ → 0 член γ/θ уходит в бесконечность и ломает график. В точке θc обе ветви совпадают.
- **`bbp_overlap_theory`** — предел перекрытия ⟨v̂₁, v₁⟩² (Paul, Benaych-Georges): 0 ниже порога, (1 − γ/θ²)/(1 + γ/θ) выше.
- `thetas = np.round(np.arange(0.1, 2.51, 0.1), 1)` — сетка θ = 0.1, 0.2, …, 2.5 (25 значений). Правая граница 2.51, чтобы 2.5 точно вошло; `np.round(..., 1)` убирает хвосты вроде 0.30000000000000004.
- `n_runs = 10` — число независимых генераций для каждого θ (задание требует M ≥ 5).
- `lam1_results`, `overlap_results` — таблицы 25×10: строка — значение θ, столбец — номер генерации.
- Двойной цикл: `enumerate(thetas)` даёт номер `i` и значение `theta`; для каждой генерации строим данные, находим старшую пару, записываем λ̂₁ и квадрат первой координаты вектора (перекрытие с v₁ = e₁). Знак собственного вектора произволен, поэтому берём квадрат.
- Последняя строка печатает время всего перебора (250 разложений).

### Ячейка 9. Графики BBP (задание 2, п. 3)

```python
theta_grid = np.linspace(0.05, 2.5, 500)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.errorbar(thetas, lam1_results.mean(axis=1), yerr=lam1_results.std(axis=1), fmt="o", ms=4, capsize=3,
            label=r"эксперимент $\hat\lambda_1$ (среднее ± std)")
ax.plot(theta_grid, [bbp_lambda_theory(t, gamma2) for t in theta_grid], "r", lw=2, label="теория BBP")
ax.plot(theta_grid, 1 + theta_grid, "g:", lw=1.5, label=r"истинная дисперсия $1+\theta$")
ax.axhline(lam_plus2, color="gray", ls="-.", lw=1, label=r"$\lambda_+$ (край шума)")
ax.axvline(theta_c, color="k", ls="--", lw=1, label=rf"$\theta_c={theta_c:.3f}$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\hat\lambda_1$")
ax.set_title("Старшее собственное число")
ax.legend(fontsize=8)

ax = axes[1]
ax.errorbar(thetas, overlap_results.mean(axis=1), yerr=overlap_results.std(axis=1), fmt="o", ms=4, capsize=3,
            label=r"эксперимент $\langle\hat v_1,v_1\rangle^2$")
ax.plot(theta_grid, [bbp_overlap_theory(t, gamma2) for t in theta_grid], "r", lw=2,
        label="теория (Paul, Benaych-Georges)")
ax.axvline(theta_c, color="k", ls="--", lw=1, label=r"$\theta_c$")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel("квадрат косинуса")
ax.set_title("Насколько найденное направление совпадает с истинным")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

- `theta_grid` — 500 точек для гладких теоретических кривых.
- **Левый график:**
    - `ax.errorbar(...)` — точки эксперимента с «усами». `mean(axis=1)` — среднее по строке, т.е. по 10 генерациям при одном θ; `std(axis=1)` — разброс, длина усов. `fmt="o"` — кружки, `capsize=3` — поперечные чёрточки на концах усов.
    - Красная кривая — `bbp_lambda_theory`, посчитанная в каждой точке сетки (list comprehension, потому что функция работает с одним числом).
    - Зелёная пунктирная — истинная дисперсия вдоль сигнала 1 + θ, чтобы было видно завышение.
    - `axhline` — горизонтальная линия λ+, `axvline` — вертикальный пунктир θc (требование задания).
    - `rf"..."` — строка одновременно raw (для LaTeX) и f-строка (для подстановки числа).
- **Правый график** — то же для перекрытия с теоретической кривой Paul–Benaych-Georges и пунктиром θc.

### Ячейка 10. Калибровка: функция-валидатор (задание 2, п. 4)

```python
TW_QUANTILE_99 = 2.02
NO_SIGNAL = "Сигнал отсутствует или находится ниже предела детектирования"


def detection_threshold(p, n):
    sigma_tw = (np.sqrt(n) + np.sqrt(p)) * (1 / np.sqrt(n) + 1 / np.sqrt(p)) ** (1 / 3) / n
    return (1 + np.sqrt(p / n)) ** 2 + TW_QUANTILE_99 * sigma_tw


def calibrate_theta(lam1, p=n_features, n=n_samples):
    if lam1 <= detection_threshold(p, n):
        return NO_SIGNAL
    gamma = p / n
    b = lam1 + 1 - gamma
    return (b + np.sqrt(b ** 2 - 4 * lam1)) / 2 - 1


print(f"Порог детектирования: λ̂1 > {detection_threshold(n_features, n_samples):.4f}  (λ+ = {lam_plus2:.4f})\n")
for lam1 in [2.9, 2.95, 3.0, 3.75, 5.0]:
    result = calibrate_theta(lam1)
    print(f"λ̂1 = {lam1:<5} ->", result if result == NO_SIGNAL else f"θ̂ = {result:.4f}")
```

- `TW_QUANTILE_99 = 2.02` — 99%-квантиль распределения Трейси–Видома TW₁: чистый шум превышает её лишь в 1% случаев. Имена заглавными буквами — константы.
- `NO_SIGNAL` — текст вердикта ровно в формулировке задания.
- **`detection_threshold(p, n)`** — порог обнаружения: λ+ плюс запас. `sigma_tw` — масштаб флуктуаций наибольшего собственного числа (формула Johnstone, 2001), для p = 1500, n = 3000 это ≈ 0.011. Порог = 2.9142 + 2.02·0.011 ≈ 2.9365.
- **`calibrate_theta(lam1, p, n)`** — функция-валидатор (guard clause):
    - Проверка `if lam1 <= detection_threshold(...)` стоит первой: если λ̂₁ не вышло за порог, сразу возвращаем вердикт и дальше не считаем. Это и есть guard clause («охранное условие»).
    - Почему проверка обязательна: формула обращения получается из уравнения λ = (1+θ)(1+γ/θ). Обозначим t = 1 + θ, тогда t² − (λ + 1 − γ)t + λ = 0. Дискриминант (λ + 1 − γ)² − 4λ = (λ − λ-)(λ − λ+), он отрицателен при λ- < λ < λ+, и корень не существует. Запас на TW нужен, потому что чистый шум выходит за λ+ примерно в половине случаев (ячейка 5), и без запаса шум выдавал бы «сигнал».
    - `b = lam1 + 1 - gamma`, затем больший корень квадратного уравнения (b + √(b² − 4λ))/2 = t и θ̂ = t − 1. Это ровно формула из задания.
- Цикл проверяет функцию на 5 значениях λ̂₁. Выражение `result if result == NO_SIGNAL else f"θ̂ = {result:.4f}"` печатает либо вердикт, либо число.

### Ячейка 11. Сравнение истинной, наивной и калиброванной оценок (задание 2, п. 4)

```python
theta_naive = lam1_results - 1
theta_calibrated = np.full(lam1_results.shape, np.nan)
for i in range(len(thetas)):
    for run in range(n_runs):
        result = calibrate_theta(lam1_results[i, run])
        if result != NO_SIGNAL:
            theta_calibrated[i, run] = result

calibrated_mean = np.full(len(thetas), np.nan)
calibrated_std = np.full(len(thetas), np.nan)
n_detected = np.zeros(len(thetas), dtype=int)
for i, row in enumerate(theta_calibrated):
    found = row[~np.isnan(row)]
    n_detected[i] = len(found)
    if len(found) > 0:
        calibrated_mean[i], calibrated_std[i] = found.mean(), found.std()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thetas, thetas, "k-", lw=1.5, label=r"истинное $\theta$")
ax.errorbar(thetas, theta_naive.mean(axis=1), yerr=theta_naive.std(axis=1), fmt="s", ms=4, capsize=3,
            color="tab:orange", label=r"наивная $\hat\theta_{naive}=\hat\lambda_1-1$")
ax.errorbar(thetas, calibrated_mean, yerr=calibrated_std, fmt="o", ms=4, capsize=3,
            color="tab:blue", label=r"калиброванная $\hat\theta$ (где сигнал обнаружен)")
ax.axvline(theta_c, color="k", ls="--", lw=1, label=r"$\theta_c$")
ax.axvspan(0, theta_c, color="gray", alpha=0.12)
ax.set_xlabel(r"истинное $\theta$")
ax.set_ylabel("оценка")
ax.set_title("Калибровка силы сигнала")
ax.legend(fontsize=8)
plt.show()

print(f"{'θ':>5}{'наивная':>10}{'калибр.':>10}{'обнаружен':>12}")
for theta, naive, calibrated, detected in zip(thetas, theta_naive.mean(axis=1), calibrated_mean, n_detected):
    print(f"{theta:>5.1f}{naive:>10.3f}{calibrated:>10.3f}{detected:>9}/{n_runs}")
```

- `theta_naive = lam1_results - 1` — наивная оценка θ̂_naive = λ̂₁ − 1 сразу для всей таблицы 25×10 (numpy вычитает поэлементно).
- `theta_calibrated = np.full(..., np.nan)` — таблица того же размера, заполненная `nan` («не число»). `nan` будет означать «сигнал не обнаружен».
- Двойной цикл прогоняет каждое из 250 значений λ̂₁ через `calibrate_theta`; если ответ не вердикт, записываем число.
- Второй блок считает по каждому θ среднее и std калиброванной оценки **только по тем генерациям, где сигнал найден**:
    - `row[~np.isnan(row)]` — элементы строки, которые не `nan` (`~` — логическое «не»).
    - `n_detected[i]` — сколько из 10 генераций прошло порог.
    - Если найдено хоть что-то, записываем среднее и std; иначе остаётся `nan`, и точка на графике не рисуется.
- График: чёрная диагональ — истинное θ; оранжевые квадраты — наивная оценка; синие кружки — калиброванная; пунктир θc; `axvspan` закрашивает серым зону ниже порога.
- Таблица в конце: для каждого θ наивная оценка, калиброванная и сколько раз из 10 сигнал обнаружен.

### Ячейка 12. Загрузка ResNet-18 и быстрый спектр слоя (задание 3, п. 1)

```python
import torch
import torchvision.models as models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()
model_weights = dict(model.named_parameters())


def layer_matrix(layer_name):
    W = model_weights[layer_name + ".weight"].detach().double().numpy()
    return W.reshape(W.shape[0], -1)


def layer_eigenvalues(W):
    N = max(W.shape)
    singular_values = np.linalg.svd(W, compute_uv=False)
    eigenvalues = singular_values ** 2 / N
    eigenvalues = eigenvalues[eigenvalues > 1e-10 * eigenvalues.max()]
    return np.sort(eigenvalues)


W = layer_matrix("layer4.1.conv2")

start = time.perf_counter()
eigenvalues_fast = layer_eigenvalues(W)
time_fast = time.perf_counter() - start

start = time.perf_counter()
eigenvalues_full = np.linalg.eigvalsh(W.T @ W / max(W.shape))
time_full = time.perf_counter() - start

print(f"layer4.1.conv2: W {W.shape}, ненулевых собственных чисел: {len(eigenvalues_fast)}")
print(f"через SVD: {time_fast:.2f} c;  полная матрица 4608x4608: {time_full:.2f} c, "
      f"из них почти нулевых: {(np.abs(eigenvalues_full) < 1e-10).sum()}")
print("максимальное расхождение 512 старших чисел:",
      np.abs(eigenvalues_full[-len(eigenvalues_fast):] - eigenvalues_fast).max())
```

- `import torch`, `torchvision.models` — PyTorch и готовые архитектуры.
- `models.resnet18(weights=models.ResNet18_Weights.DEFAULT)` — предобученная ResNet-18 (веса ~45 МБ скачиваются один раз и кэшируются), как в задании.
- `model.eval()` — режим применения (не обучения); на веса не влияет.
- `model_weights = dict(model.named_parameters())` — словарь «имя параметра → тензор», например `"layer4.1.conv2.weight"`.
- **`layer_matrix(layer_name)`** — матрица весов слоя:
    - `.detach()` отвязывает тензор от графа градиентов, `.double()` переводит в float64 (точнее считаются собственные числа), `.numpy()` превращает в массив numpy.
    - `W.reshape(W.shape[0], -1)` — для свёртки форма (Cout, Cin, K, K) превращается в Cout × (Cin·K·K); `-1` значит «посчитай размер сам». У `fc` форма уже 1000×512 и не меняется. Это ровно требование задания.
- **`layer_eigenvalues(W)`** — ненулевые собственные числа S = (1/N)WᵀW, N = max(M, K), без построения S:
    - `np.linalg.svd(W, compute_uv=False)` — только сингулярные числа (векторы не нужны, так быстрее).
    - Если W = U·diag(s)·Vᵀ, то WᵀW = V·diag(s²)·Vᵀ и WWᵀ = U·diag(s²)·Uᵀ. Значит ненулевые собственные числа S равны s²/N, и их ровно min(M, K).
    - Отсекаем машинные нули тем же относительным порогом 10⁻¹⁰, что в задании 1 (важно для слоя с коллапсом), и сортируем.
- Дальше на `layer4.1.conv2` (512×4608) сравниваем два способа: быстрый через SVD и «в лоб» — `eigvalsh` матрицы WᵀW размера 4608×4608 (предупреждение задания).
- Печать: число ненулевых чисел (512), время обоих способов, сколько «почти нулей» дала большая матрица (4096 = 4608 − 512) и максимальное расхождение старших 512 чисел между способами (проверка, что ответ тот же).

### Ячейка 13. Наивная оценка α методом наименьших квадратов (задание 3, п. 2)

```python
def ols_alpha(eigenvalues, n_bins, log_bins=False):
    if log_bins:
        edges = np.geomspace(eigenvalues.min(), eigenvalues.max(), n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(eigenvalues.min(), eigenvalues.max(), n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(eigenvalues, bins=edges)
    density = counts / (counts.sum() * np.diff(edges))
    use = (np.arange(n_bins) >= np.argmax(density)) & (counts > 0)
    slope, _ = np.polyfit(np.log(centers[use]), np.log(density[use]), 1)
    return -slope, centers[use], density[use]


print(f"{'бинов':>6}{'α (линейные бины)':>20}{'α (лог. бины)':>16}")
for n_bins in [10, 20, 30, 50, 80, 120, 200]:
    alpha_linear = ols_alpha(eigenvalues_fast, n_bins)[0]
    alpha_log = ols_alpha(eigenvalues_fast, n_bins, log_bins=True)[0]
    print(f"{n_bins:>6}{alpha_linear:>20.3f}{alpha_log:>16.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, log_bins in zip(axes, [False, True]):
    for n_bins in [20, 80, 200]:
        alpha, centers, density = ols_alpha(eigenvalues_fast, n_bins, log_bins)
        ax.loglog(centers, density, "o-", ms=3, lw=0.8, label=f"бинов={n_bins}: α={alpha:.2f}")
    ax.set_title(("Логарифмические" if log_bins else "Линейные") + " бины, МНК в log-log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\rho(\lambda)$")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
```

Идея: если ρ(λ) ∝ λ^(−α), то ln ρ = −α·ln λ + const — прямая в log-log, и её наклон равен −α.

- **`ols_alpha(eigenvalues, n_bins, log_bins)`:**
    - Границы бинов `edges`: при линейных бинах `np.linspace` (одинаковая ширина), при логарифмических `np.geomspace` (одинаковое *отношение* соседних границ, ширина растёт пропорционально λ). Центры бинов — среднее арифметическое границ для линейных и среднее геометрическое для логарифмических (середина в лог-шкале).
    - `np.histogram(..., bins=edges)` — сколько чисел попало в каждый бин.
    - `density = counts / (counts.sum() * np.diff(edges))` — переводим счётчики в плотность: доля точек, делённая на ширину бина (`np.diff` — ширины). Без деления на ширину логарифмические бины дали бы неверный наклон.
    - `use` — какие бины берём в регрессию: правее пика гистограммы (`np.argmax(density)` — номер самого высокого бина) и только непустые (`counts > 0`), иначе ln 0 = −∞.
    - `np.polyfit(x, y, 1)` — МНК-прямая степени 1, возвращает наклон и сдвиг. Возвращаем α = −наклон и точки для графика.
- Таблица: α при 7 разных числах бинов для обоих типов бинов.
- Графики: точки (ln λ, ln ρ) в log-log для 20, 80 и 200 бинов.

**Почему МНК здесь не работает (вопросы задания):**

- *Линейные бины.* В хвосте степенного закона точек мало, и правые бины содержат 0 или 1 число. Пустые бины приходится выбрасывать — это систематический сдвиг. Бины с одной точкой дают искусственный «пол» плотности 1/(N·ширина). При смене числа бинов меняется и пол, и набор выброшенных бинов, поэтому α скачет.
- *Логарифмические бины.* Ширина бина растёт пропорционально λ, поэтому вероятностная мера бина ∫ от λ до cλ от λ^(−α) dλ ∝ λ^(1−α) убывает гораздо медленнее, чем при линейных бинах. В каждый бин попадает сопоставимое число точек, пустых меньше, оценка стабильнее. Но это всё ещё МНК.
- *Начало хвоста.* МНК не умеет выбирать λ_min и смешивает в одну прямую балк и хвост.
- *Принципиально.* МНК предполагает независимые ошибки одинаковой дисперсии (гауссовы). У логарифма пуассоновских счётчиков ошибки разного размера (относительная ошибка ~1/√count, в редких бинах огромная), а точки гистограммы зависимы через нормировку. Оценка смещена, корректных доверительных интервалов нет (Clauset, Shalizi, Newman, 2009). Правильный метод — максимальное правдоподобие.

### Ячейка 14. Оценка α методом максимального правдоподобия (задание 3, п. 2)

```python
def hill_alpha(tail, lam_min):
    return 1 + len(tail) / np.sum(np.log(tail / lam_min))


def fit_power_law(eigenvalues, min_tail_size=10):
    eigenvalues = np.sort(eigenvalues)
    if len(eigenvalues) <= min_tail_size:
        return None
    best_fit = None
    for start in range(len(eigenvalues) - min_tail_size + 1):
        lam_min = eigenvalues[start]
        tail = eigenvalues[start:]
        alpha = hill_alpha(tail, lam_min)
        theory_cdf = 1 - (tail / lam_min) ** (-(alpha - 1))
        k = len(tail)
        cdf_after_step = np.arange(1, k + 1) / k
        cdf_before_step = np.arange(k) / k
        ks_distance = max(np.max(cdf_after_step - theory_cdf), np.max(theory_cdf - cdf_before_step))
        if best_fit is None or ks_distance < best_fit["ks_distance"]:
            best_fit = {"alpha": alpha, "lam_min": lam_min, "ks_distance": ks_distance, "tail_size": len(tail)}
    return best_fit


fit = fit_power_law(eigenvalues_fast)
print(f"layer4.1.conv2: α = {fit['alpha']:.3f}, λ_min = {fit['lam_min']:.3e}, D_KS = {fit['ks_distance']:.3f}, "
      f"точек в хвосте: {fit['tail_size']} из {len(eigenvalues_fast)}")

print("\nДля сравнения — α при λ_min, выбранном «на глаз» (квантиль спектра):")
for quantile in [0.5, 0.7, 0.8, 0.9]:
    lam_min = np.quantile(eigenvalues_fast, quantile)
    tail = eigenvalues_fast[eigenvalues_fast >= lam_min]
    print(f"  λ_min = квантиль {quantile}: α = {hill_alpha(tail, lam_min):.3f}")
```

- **`hill_alpha(tail, lam_min)`** — оценка Хилла из задания: α = 1 + N_tail / Σ ln(λᵢ/λ_min). Это решение задачи максимального правдоподобия для степенного закона: такое α делает наблюдаемые числа самыми вероятными.
- **`fit_power_law(eigenvalues, min_tail_size=10)`** — подбор λ_min по Колмогорову–Смирнову:
    - Сортируем числа. Если их не больше 10, возвращаем `None`: по паре точек распределение не оценить (так будет у слоя с коллапсом).
    - Цикл перебирает каждое собственное число как кандидата в начало хвоста `lam_min`, оставляя в хвосте минимум 10 точек.
    - `tail = eigenvalues[start:]` — все числа от кандидата и правее; по ним считаем α.
    - `theory_cdf` — теоретическая функция распределения степенного закона P(λ) = 1 − (λ/λ_min)^(−(α−1)) в точках хвоста.
    - Эмпирическая функция распределения — «ступенька»: сразу перед i-й точкой она равна i/k (`cdf_before_step`), сразу после — (i+1)/k (`cdf_after_step`). Максимальное расхождение может быть с любой стороны ступеньки, поэтому `ks_distance` — максимум из двух разностей. Это статистика D_KS = max |F_эмп − F_теор|.
    - Если D_KS меньше лучшего найденного, запоминаем этот вариант в словаре `best_fit` (α, λ_min, D_KS, размер хвоста).
- `fit = fit_power_law(eigenvalues_fast)` — результат для `layer4.1.conv2`, печатаем его.
- Последний цикл для сравнения считает α при λ_min, выбранном «на глаз» как квантиль спектра (`np.quantile` — значение, левее которого лежит заданная доля чисел). Он показывает, насколько α зависит от порога, — поэтому порог выбирается формально, по минимуму D_KS.

### Ячейка 15. CCDF в log-log (задание 3, п. 2, визуализация)

```python
def plot_ccdf(ax, eigenvalues, fit, title):
    eigenvalues = np.sort(eigenvalues)
    ccdf = 1 - np.arange(len(eigenvalues)) / len(eigenvalues)
    ax.loglog(eigenvalues, ccdf, ".", ms=4, label="эмпирическая CCDF")
    if fit is not None:
        x = np.geomspace(fit["lam_min"], eigenvalues.max(), 100)
        tail_share = fit["tail_size"] / len(eigenvalues)
        ax.loglog(x, tail_share * (x / fit["lam_min"]) ** (-(fit["alpha"] - 1)), "r-", lw=2,
                  label=rf"MLE: $\alpha={fit['alpha']:.2f}$")
        ax.axvline(fit["lam_min"], color="k", ls="--", lw=1, label=r"$\lambda_{\min}$ (граница шума)")
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$P(X\geq\lambda)$")
    ax.set_title(title)
    ax.legend(fontsize=8)


fig, ax = plt.subplots(figsize=(7, 4.5))
plot_ccdf(ax, eigenvalues_fast, fit, "layer4.1.conv2: CCDF в log-log")
plt.show()
```

- **`plot_ccdf(ax, eigenvalues, fit, title)`** рисует на оси `ax` функцию выживания (CCDF) P(X ≥ λ):
    - После сортировки для i-го числа (с нуля) доля чисел, которые не меньше него, равна 1 − i/N — это `ccdf`. Бины не нужны, каждая точка — одно собственное число.
    - `ax.loglog(...)` — график с логарифмическими осями X и Y.
    - Если подгонка есть (`fit is not None`), рисуем красную прямую степенного закона. Для плотности λ^(−α) CCDF ∝ λ^(−(α−1)), т.е. прямая с наклоном −(α−1). Её умножаем на `tail_share` — долю чисел в хвосте: хвост составляет лишь часть всех чисел, и без множителя прямая начиналась бы с 1 и не легла бы на точки.
    - `axvline` — вертикальный пунктир λ_min, граница шума (требование задания).
    - `set_minor_formatter(plt.NullFormatter())` убирает подписи мелких делений оси X, иначе в узком диапазоне они слипаются.
- Ниже функция вызывается для `layer4.1.conv2`. Та же функция потом рисует нижний ряд графиков в ячейке 17.

### Ячейка 16. Четыре тестовых слоя (задание 3, п. 3)

```python
def keep_top_singular_values(W, k):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    S[k:] = 0.0
    return U @ np.diag(S) @ Vt


original_layer1 = layer_matrix("layer1.0.conv1")
test_layers = {
    "layer1.0.conv1": ("случайный", random_gen.normal(0.0, original_layer1.std(), size=original_layer1.shape)),
    "layer2.0.conv1": ("штатный", layer_matrix("layer2.0.conv1")),
    "layer3.0.conv1": ("коллапс", keep_top_singular_values(layer_matrix("layer3.0.conv1"), k=2)),
    "fc": ("выходной", layer_matrix("fc")),
}

with torch.no_grad():
    for layer_name, (_, W) in test_layers.items():
        weight = model_weights[layer_name + ".weight"]
        weight.copy_(torch.from_numpy(W).reshape(weight.shape))
```

- **`keep_top_singular_values(W, k)`** — коллапс ранга, как в задании: SVD W = U·Σ·Vᵀ (`full_matrices=False` — экономная форма без лишних столбцов), `S[k:] = 0.0` зануляет все сингулярные числа, кроме k старших, и матрица собирается обратно. `U @ np.diag(S) @ Vt` — произведение трёх матриц; `np.diag(S)` строит диагональную матрицу из вектора.
- `original_layer1` — исходные веса первого слоя: нужны, чтобы взять их стандартное отклонение.
- Словарь `test_layers`: имя слоя в модели → (подпись, матрица весов).
    1. `layer1.0.conv1` — **случайный**: новые веса из N(0, σ²), где σ = `original_layer1.std()`. Масштаб как у обученного слоя, а структура уничтожена — имитация необученного слоя.
    2. `layer2.0.conv1` — **штатный**: исходные веса.
    3. `layer3.0.conv1` — **коллапс**: оставлены 2 старших сингулярных числа.
    4. `fc` — **выходной** слой как есть.
- Блок `with torch.no_grad():` записывает изменённые веса обратно в модель («перезапишите веса» в задании). `no_grad` нужен, потому что PyTorch запрещает менять «на месте» параметры, за которыми следит autograd. `torch.from_numpy(W)` превращает массив обратно в тензор, `.reshape(weight.shape)` возвращает исходную 4D-форму свёртки, `weight.copy_(...)` копирует значения в параметр модели (подчёркивание в конце имени — операция на месте).

### Ячейка 17. Спектры, srank и α четырёх слоёв (задание 3, п. 3)

```python
def stable_rank(eigenvalues):
    return eigenvalues.sum() / eigenvalues.max()


summary = []
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for column, (layer_name, (label, W)) in enumerate(test_layers.items()):
    eigenvalues = layer_eigenvalues(W)
    gamma = min(W.shape) / max(W.shape)
    lam_minus, lam_plus = mp_bounds(gamma, sigma2=W.var())
    fit = fit_power_law(eigenvalues)
    summary.append({
        "name": f"{layer_name} ({label})", "shape": W.shape, "n_eigenvalues": len(eigenvalues),
        "srank": stable_rank(eigenvalues), "alpha": fit["alpha"] if fit else None,
        "lam_max": eigenvalues.max(), "lam_plus": lam_plus, "n_above_mp": int((eigenvalues > lam_plus).sum()),
    })

    ax = axes[0, column]
    if len(eigenvalues) > 10:
        ax.hist(eigenvalues, bins=min(50, len(eigenvalues) // 4), density=True, alpha=0.55, label="спектр слоя")
        lam_grid = np.linspace(1e-12, 1.2 * lam_plus, 500)
        ax.plot(lam_grid, mp_density(lam_grid, gamma, W.var()), "r", lw=1.5, label="МП для шума того же масштаба")
    else:
        ax.stem(eigenvalues, np.ones_like(eigenvalues), basefmt=" ", label=f"все ненулевые λ ({len(eigenvalues)} шт.)")
        ax.text(0.5, 0.5, f"остальные {min(W.shape) - len(eigenvalues)} λ = 0", transform=ax.transAxes, ha="center")
    ax.axvline(lam_plus, color="k", ls="--", lw=1, label=r"$\lambda_+^{MP}$")
    ax.set_title(f"{layer_name} ({label})\n{W.shape}, γ={gamma:.3f}", fontsize=10)
    ax.set_xlabel(r"$\lambda$")
    ax.legend(fontsize=7)

    plot_ccdf(axes[1, column], eigenvalues, fit, "CCDF, log-log")
plt.tight_layout()
plt.show()
```

- **`stable_rank(eigenvalues)`** — эффективный ранг из задания: srank = ‖W‖²_F / ‖W‖²₂ = Σλᵢ / λmax. Квадрат нормы Фробениуса равен сумме квадратов сингулярных чисел, квадрат спектральной нормы — квадрату наибольшего, а общий множитель 1/N сокращается.
- `summary = []` — сюда собираем строки итоговой таблицы.
- Рисунок 2×4: верхний ряд — спектральная плотность, нижний — CCDF; столбец — слой.
- Цикл по слоям (`enumerate` даёт номер столбца):
    - `eigenvalues = layer_eigenvalues(W)` — ненулевой спектр.
    - `gamma = min(W.shape) / max(W.shape)` — отношение сторон. Для S = (1/N)WWᵀ при M < K это ровно γ = M/K из закона МП.
    - `mp_bounds(gamma, sigma2=W.var())` — где закончился бы спектр, если бы слой был чистым шумом с той же дисперсией элементов σ² = Var(W). Это ключ к доказательству для случайного слоя.
    - `fit = fit_power_law(eigenvalues)` — α по MLE+КС (или `None`).
    - В `summary` добавляем словарь: имя, форма, число ненулевых λ, srank, α (`None`, если не определён), λmax, λ+ МП и `n_above_mp` — сколько собственных чисел вышло за край шума.
    - Верхний график: если чисел больше 10 — гистограмма (число бинов `min(50, N // 4)`, `//` — целочисленное деление) и красная кривая МП для шума того же масштаба. Иначе (коллапс, 2 числа) гистограмма бессмысленна, поэтому `ax.stem` рисует «палочки» в позициях чисел, а `ax.text` подписывает, сколько остальных λ равны нулю (`transform=ax.transAxes` — координаты в долях графика).
    - Пунктир λ+ МП на каждом графике.
    - Нижний график — `plot_ccdf` из ячейки 15.

### Ячейка 18. Итоговая таблица слепого теста

```python
print(f"{'слой':<26}{'форма':>13}{'#λ':>5}{'srank':>8}{'α (MLE)':>9}{'λmax':>11}{'λ+ МП':>11}{'#λ > λ+':>9}")
for row in summary:
    alpha = "нет" if row["alpha"] is None else f"{row['alpha']:.2f}"
    print(f"{row['name']:<26}{str(row['shape']):>13}{row['n_eigenvalues']:>5}{row['srank']:>8.2f}{alpha:>9}"
          f"{row['lam_max']:>11.2e}{row['lam_plus']:>11.2e}{row['n_above_mp']:>9}")
```

- Заголовок таблицы, затем строка на каждый слой из `summary`.
- Если α не определён (коллапс), печатаем «нет», иначе число с двумя знаками.
- `:.2e` — экспоненциальная запись (например, `4.83e-03`): собственные числа весов маленькие, так их удобнее сравнивать.
- По этой таблице делается диагноз: у случайного слоя ни одно λ не выходит за λ+ МП, у обученных — десятки; у коллапса 2 числа и srank ≈ 1.5.

**Доказательство, что у случайного слоя нет степенного хвоста.** Алгоритм MLE+КС **всегда** выдаёт какое-то α: он ищет лучший степенной закон, а не проверяет, существует ли он. По теории МП у матрицы с независимыми элементами дисперсии σ² весь спектр лежит в [λ-, λ+], λ+ = σ²(1 + √γ)² (с поправкой Трейси–Видома порядка N^(−2/3)). В таблице у случайного слоя λmax = 4.83·10⁻³ ≤ λ+ = 5.07·10⁻³, за край не вышло ни одно число. Степенной хвост — это масса **за** пределами балка, простирающаяся на порядки величины. Здесь же весь «хвост» — правый край МП-плотности шириной меньше порядка величины, а большое α = 11.5 — артефакт подгонки к нему.

## Соответствие заданию

Каждый пункт `lab2.pdf` и где он выполнен в `2.ipynb`.

| Пункт задания | Где | Статус |
| --- | --- | --- |
| 1.1 Функция: X ~ N(0, σ²), σ² = 1, Σ̂ = (1/n)XXᵀ, спектр через `eigvalsh` | ячейка 2, `noise_eigenvalues` | выполнено |
| 1.2 Режимы A (p = 500) и B (p = 2000) при n = 1000 | ячейка 3 | выполнено |
| 1.3 Гистограмма плотности + кривая ρ_MP + пунктиры λ-, λ+ | ячейка 3 | выполнено |
| Предостережение B.1: порог машинного нуля вместо `eigs > 0` | ячейка 3, `zero_tol` | выполнено, обоснование в разборе ячейки 3 |
| Предостережение B.2: множитель масштаба и его математическое обоснование | ячейка 3, `scale` | выполнено, вывод множителя γ в разборе ячейки 3 и в разборе выводов |
| 1.4 Таблица [λ-, λ+] и [λ̂min, λ̂max] для обоих режимов | ячейка 4 | выполнено |
| 1.4 Совпали ли границы, почему λ̂max правее λ+, какое распределение | ячейка 5 + разбор выводов | выполнено (Трейси–Видом, проверено экспериментом) |
| 1.4 Почему при p = n максимум стремится к 4 | разбор выводов | выполнено |
| 2.1 Spiked-модель через масштабирование первой строки на √(1+θ) | ячейка 6, `make_spiked_data` | выполнено |
| 2.2 θ от 0.1 до 2.5 с шагом 0.1, M ≥ 5 генераций | ячейка 8 (M = 10) | выполнено |
| 2.2 λ̂₁, v̂₁ и ⟨v̂₁, v₁⟩² | ячейки 6, 8 | выполнено |
| Предостережение: порядок `eigh`, правильный срез `[-1]` | ячейка 7 | выполнено, ошибка показана |
| Совет: быстрый поиск только старшей компоненты | ячейки 6, 7 (`svds`) | выполнено, время сравнено |
| 2.3 График λ̂₁(θ) + теория BBP | ячейка 9 | выполнено |
| 2.3 График перекрытия + теория Paul–Benaych-Georges | ячейка 9 | выполнено |
| 2.3 Пунктир θc | ячейки 9, 11 | выполнено |
| Предостережение: кусочная теоретическая кривая | ячейка 8 | выполнено |
| 2.4 Функция-валидатор: θ̂ или вердикт «Сигнал отсутствует…» | ячейка 10, `calibrate_theta` | выполнено, текст вердикта дословно |
| 2.4 График: истинное θ, наивная и калиброванная оценки | ячейка 11 | выполнено |
| 3.1 fc напрямую, Conv2d → Cout × (Cin·K·K) | ячейка 12, `layer_matrix` | выполнено |
| 3.1 S = (1/N)WᵀW через сингулярные числа, без матрицы 4608×4608 | ячейка 12, `layer_eigenvalues` | выполнено, сравнение с «лобовым» способом |
| 3.2 МНК по гистограмме, зависимость от числа бинов | ячейка 13 | выполнено (линейные и лог. бины) |
| 3.2 Вопросы: линейные бины, лог. бины, неприменимость МНК | разбор ячейки 13 | выполнено |
| 3.2 Оценщик Хилла + λ_min по минимуму D_KS | ячейка 14 | выполнено |
| 3.2 CCDF в log-log + аппроксимация MLE + линия λ_min | ячейка 15 | выполнено |
| 3.3 Случайный / штатный / коллапс (2 сингулярных числа) / fc | ячейка 16 | выполнено, веса записаны в модель |
| 3.3 Для каждого слоя: спектральная плотность, srank, α | ячейки 17, 18 | выполнено |
| 3.3 Доказательство через λ+ МП, что у случайного слоя хвоста нет | разбор ячейки 18 | выполнено |
| 3.3 Итог: фазы Мартина–Махони, зона Златовласки, коллапс, шум | итоговые выводы | выполнено |

Два места, где решение сознательно отличается от буквального текста задания:

- **Порог в валидаторе** (п. 2.4). Формула обращения работает при λ̂₁ ≥ λ+, а порог взят чуть выше: λ+ + 2.02·σ_TW ≈ 2.9365 вместо 2.9142. Без запаса чистый шум примерно в половине случаев выдавал бы «сигнал» (это видно в ячейке 5). Если преподаватель хочет ровно λ+, достаточно в `detection_threshold` убрать слагаемое `TW_QUANTILE_99 * sigma_tw`.
- **«λ̂max систематически правее λ+»** (п. 1.4). Эксперимент показывает, что в среднем максимум чуть **левее** λ+ и правее оказывается только в 5–23% случаев. Это не ошибка кода, а точный ответ теории Трейси–Видома (среднее TW₁ отрицательное); в отчёте это объяснено, а не подогнано под формулировку.

## Разбор выводов

Все числа ниже — из прогона ноутбука с seed 42; при повторном запуске они те же.

### Задание 1: спектр шума

**Нули в режиме B.** Найдено 1000 нулевых собственных чисел — ровно p − n. Самый большой «ноль» по модулю равен 3.4·10⁻¹⁵, а 494 из них отрицательны. Значит, с условием `eigs > 0` в гистограмму попали бы ~506 мусорных чисел, а относительный порог 10⁻¹⁰·λmax (~6·10⁻¹⁰) отсекает их все.

**Графики.** В обоих режимах гистограмма лежит на красную кривую, а все столбики — между пунктирами λ- и λ+. Истинная дисперсия по всем направлениям равна 1, но выборочные собственные числа размазаны от 0.09 до 2.9 (γ = 0.5) и от 0.17 до 5.8 (γ = 2). Пик плотности смещён влево, длинный правый склон плавно уходит в ноль у λ+.

**Почему множитель 2 в режиме B.** Полная мера МП состоит из атома в нуле веса 1 − 1/γ и непрерывной части веса 1/γ = 0.5. Гистограмма по ненулевым числам с `density=True` имеет площадь 1. Чтобы сравнивать одинаковое, берём условную плотность «при условии λ ≠ 0»:

$$
\rho_{cond}(\lambda) = \frac{\rho_{MP}(\lambda)}{\int_{\lambda_-}^{\lambda_+}\rho_{MP}\,d\lambda} = \frac{\rho_{MP}(\lambda)}{1/\gamma} = \gamma\,\rho_{MP}(\lambda)
$$

**Таблица границ.**

| Режим | γ | λ- теория | λmin эксп. | λ+ теория | λmax эксп. | Нулей |
| --- | --- | --- | --- | --- | --- | --- |
| A (полный ранг) | 0.5 | 0.0858 | 0.0928 | 2.9142 | 2.8841 | 0 |
| B (дефект ранга) | 2 | 0.1716 | 0.1734 | 5.8284 | 5.7837 | 1000 |

Границы совпали не точно, а с точностью ~0.01–0.05: закон МП — предельный, а матрицы конечные. В этом прогоне λmax оказался чуть **левее** λ+, а λmin — чуть правее λ-: конечная выборка обычно не дотягивает до краёв.

**Флуктуации максимума (γ = 1, 40 повторов).**

| n = p | среднее λmax | std | n^(2/3)·(среднее − 4) | n^(2/3)·std | доля λmax > 4 |
| --- | --- | --- | --- | --- | --- |
| 100 | 3.8259 | 0.1524 | −3.75 | 3.28 | 0.12 |
| 200 | 3.8955 | 0.0877 | −3.58 | 3.00 | 0.07 |
| 400 | 3.9249 | 0.0517 | −4.08 | 2.81 | 0.05 |
| 800 | 3.9639 | 0.0379 | −3.11 | 3.26 | 0.15 |
| 1600 | 3.9813 | 0.0224 | −2.55 | 3.07 | 0.23 |
| прогноз TW | → 4 | — | −3.04 | 3.20 | — |

Как читать: среднее с ростом n подходит к 4, а разброс сжимается (0.15 → 0.02). После умножения на n^(2/3) столбцы перестают зависеть от n и колеблются около прогноза (−3, 3.2) — масштаб n^(−2/3) и распределение Трейси–Видома подтверждены. Разброс от строки к строке (−2.55…−4.08) — обычная погрешность среднего по 40 опытам (≈ 3.2/√40 ≈ 0.5). Правее λ+ максимум оказался в 5–23% опытов: так работает тяжёлый правый хвост TW₁ при отрицательном среднем.

**Ответ на вопрос задания «почему λmax правее λ+».** Систематически правее он не оказывается. Флуктуации описывает распределение Трейси–Видома: в среднем максимум чуть левее края, но регулярно выскакивает за него. Практический вывод: выход за λ+ на величину порядка n^(−2/3) ещё не сигнал.

**Почему при p = n максимум идёт к 4, а не к 1.** λ̂₁ — это наибольшая выборочная дисперсия среди **всех** направлений. Вдоль одного заранее выбранного направления оценка ≈ 1, но максимум из огромного числа случайных оценок смещён вверх. Геометрически: n точек в n-мерном пространстве образуют не шар, а вытянутый эллипсоид с дисперсиями по осям от 0 до 4; средняя дисперсия при этом остаётся 1. Разброс не появляется из ниоткуда, а перераспределяется между осями.

### Задание 2: BBP и калибровка

**Параметры:** γ = 0.5, θc = 0.7071, λ- = 0.0858, λ+ = 2.9142 — совпадают с условием.

**Ловушка с порядком (θ = 2).**

| Способ | λ | 〈v, e₁〉² |
| --- | --- | --- |
| `eigh`, элемент [0] | 0.0864 | 6.4·10⁻⁵ |
| `eigh`, элемент [-1] | 3.7107 | 0.7053 |
| `svds`, k = 1 | 3.7107 | 0.7053 |

Элемент [0] — самое маленькое число (≈ λ-), и его вектор не имеет ничего общего с сигналом. [-1] и `svds` дают одно и то же, только `svds` быстрее: 0.15 с против 0.48 с. Проверка с теорией: (1+2)(1+0.5/2) = 3.75 ≈ 3.71, а перекрытие (1−0.125)/(1+0.25) = 0.70 ≈ 0.705.

**Графики BBP** (перебор занял 91 с):

- Слева до θc точки лежат на горизонтали λ+ ≈ 2.91: сигнала не видно. После θc они идут вверх точно по красной кривой и всегда выше зелёной 1 + θ (при θ = 2.5 измерено ~4.2 при истинных 3.5).
- Справа перекрытие ≈ 0 до θc, далее растёт до ~0.77 при θ = 2.5. Даже при сильном сигнале найденное направление не совпадает с истинным (косинус ≈ 0.88, угол ~28°).
- Около θc (0.7–0.9) у точек самые длинные «усы»: на конечной выборке излом размыт, и в одних генерациях сигнал уже вырвался, в других ещё нет.

**Порог и проверка калибровки.** Порог детектирования 2.9365 = λ+ + 0.022 (запас на TW).

| λ̂₁ | Результат | Почему |
| --- | --- | --- |
| 2.90 | сигнал отсутствует | ниже λ+, формула дала бы корень из отрицательного числа |
| 2.95 | θ̂ = 0.885 | чуть выше порога |
| 3.00 | θ̂ = 1.000 | проверка: (1+1)(1+0.5) = 3 |
| 3.75 | θ̂ = 2.000 | проверка: (1+2)(1+0.25) = 3.75 |
| 5.00 | θ̂ = 3.351 | наивная оценка дала бы 4.0 |

**График калибровки и таблица.**

- Наивная оценка λ̂₁ − 1 ниже порога стоит на ~1.90 (= λ+ − 1) даже при θ = 0.1, т.е. «видит» сильный сигнал в чистом шуме. Выше порога она завышает на 0.7–1.0 (при θ = 1: 2.015 вместо 1).
- Калиброванная оценка при θ ≥ 1.0 лежит на диагонали с ошибкой ≤ 0.04 (например, 1.029 при 1.0, 2.032 при 2.0, 2.518 при 2.5); сигнал найден во всех 10 из 10 генераций.
- При θ = 0.1–0.4 и 0.6 — 0 из 10 обнаружений: валидатор честно говорит «сигнала нет».
- Переходная зона 0.7–0.9: 2, 3 и 6 из 10. Здесь найденные оценки ≈ 0.89–0.92 — завышены, потому что порог проходят только генерации со случайно большим λ̂₁ (эффект отбора).
- Единичное срабатывание при θ = 0.5 (1 из 10) — ожидаемая редкая ложная тревога на конечной выборке: порог рассчитан на ~1% таких случаев для чистого шума, а слабый шип ещё и чуть приподнимает край.

### Задание 3: ResNet-18

**Быстрый спектр.** Для `layer4.1.conv2` (512×4608) через SVD — 0.43 с и 512 ненулевых чисел. Через полную матрицу 4608×4608 — 5.94 с, из них 4096 машинных нулей. Старшие 512 чисел совпадают с точностью 5·10⁻¹⁸: результат тот же, в ~14 раз быстрее.

**МНК по гистограмме.**

| Бинов | α, линейные бины | α, лог. бины |
| --- | --- | --- |
| 10 | 2.205 | 1.287 |
| 20 | 1.692 | 1.374 |
| 30 | 1.408 | 1.246 |
| 50 | 1.286 | 1.169 |
| 80 | 1.095 | 1.079 |
| 120 | 1.019 | 0.991 |
| 200 | 0.881 | 0.945 |

Линейные бины дают α от 2.2 до 0.88 — оценка меняется в 2.5 раза от одного лишь числа бинов. Логарифмические стабильнее (0.95–1.37), но все значения далеко от MLE: МНК смешивает балк и хвост. На графиках видно, что при линейных бинах хвост состоит из зубцов (бины с 0–1 точкой).

**MLE + КС.** α = 5.83, λ_min = 4.74·10⁻⁴, D_KS = 0.064, в хвосте 44 числа из 512. Если выбрать λ_min «на глаз», α меняется от 1.96 (медиана) до 5.47 (90%-квантиль) — поэтому порог надо выбирать формально. На CCDF хвост правее λ_min идёт по красной прямой. α > 4 — фаза Bulk-Decay, слой слабо коррелирован.

**Слепой тест.**

| Слой | Форма | #λ | srank | α (MLE) | λmax | λ+ МП | #λ > λ+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| layer1.0.conv1 (случайный) | 64×576 | 64 | 37.82 | 11.51 | 4.83·10⁻³ | 5.07·10⁻³ | 0 |
| layer2.0.conv1 (штатный) | 128×576 | 128 | 31.78 | 2.21 | 6.97·10⁻³ | 3.74·10⁻³ | 18 |
| layer3.0.conv1 (коллапс) | 256×1152 | 2 | 1.55 | — | 7.25·10⁻³ | 9.10·10⁻⁵ | 2 |
| fc (выходной) | 1000×512 | 512 | 34.60 | 3.34 | 7.14·10⁻² | 1.42·10⁻² | 35 |

- **Случайный слой.** Гистограмма лежит на кривой МП, λmax = 4.83·10⁻³ < λ+ = 5.07·10⁻³, за край не вышло ни одно число. Алгоритм всё равно выдал α = 11.5, но это подгонка к убывающему краю МП: весь «хвост» умещается менее чем в один порядок величины. Это и есть строгое доказательство через λ+ = σ²(1 + √γ)², что степенного хвоста нет.
- **Штатный слой.** Спектр не похож на МП: много маленьких чисел и 18 чисел за λ+, λmax почти вдвое больше края шума. α = 2.21 — в «зоне Златовласки» (2; 4].
- **Коллапс.** Осталось 2 ненулевых числа из 256 (на графике — две «палочки»), srank = 1.55: вся энергия в двух направлениях. α не определён — по двум точкам распределение не оценить. λ+ МП крошечный, потому что дисперсия элементов после сжатия мала.
- **fc.** 35 чисел за λ+, λmax в 5 раз больше края шума, на CCDF чёткая прямая, α = 3.34 — тоже зона Златовласки.
- **srank** у случайного, штатного и fc близок (32–38), а у коллапса падает до 1.55. Значит, srank хорошо ловит коллапс, но не отличает шум от обученного слоя — для этого нужны α и сравнение с λ+.

## Итоговые выводы

1. **Шум имеет свою «форму».** В высокой размерности собственные числа чистого шума заполняют отрезок [λ-, λ+] по закону Марченко–Пастура. Главные компоненты внутри этого отрезка — иллюзия, а не структура.
2. **Край шума сам случаен.** Максимум колеблется вокруг λ+ на величину порядка n^(−2/3) по закону Трейси–Видома, поэтому для обнаружения сигнала нужен порог с запасом.
3. **У PCA есть предел чувствительности.** Сигнал слабее θc = √(p/n) невидим (у нас θc = 0.71); выше порога его сила завышена, но исправляется обратной формулой BBP с точностью ~0.04 при θ ≥ 1.
4. **По весам можно диагностировать сеть без данных.** Для оценки хвоста нужен MLE с выбором λ_min по КС, а МНК по гистограмме ненадёжен. Любое число α нужно проверять сравнением с λ+ МП.

| Слой | Фаза Мартина–Махони | Диагноз |
| --- | --- | --- |
| layer1.0.conv1 (случайный) | 1. Random-like | чистый белый шум |
| layer2.0.conv1 (штатный) | 5. Heavy-Tailed, α = 2.21 | «зона Златовласки», оптимальный режим |
| layer3.0.conv1 (коллапс) | Rank-Collapse | слой выродился |
| fc (выходной) | 5. Heavy-Tailed, α = 3.34 | хорошо обученный классификатор |
| layer4.1.conv2 | 4. Bulk-Decay, α = 5.83 | слабообученный / избыточный |

## Видео по теме

Подборка найдена поиском; сами ролики я не смотрел, поэтому содержание проверяйте по названиям.

**Основы (на русском):**

- [Теория вероятностей #25: ковариация и корреляция, ковариационная матрица](https://www.youtube.com/watch?v=_jlO2a0GBiE) — случайные величины, ковариация, ковариационная матрица.
- [#25. Метод главных компонент (PCA)](https://www.youtube.com/watch?v=AoBykkvOMDw) — что такое главные компоненты и при чём тут собственные векторы.

**Теория случайных матриц (на русском):**

- [Коллоквиум «Случайные матрицы: теория и приложения»](https://www.youtube.com/watch?v=-JuL2Mmrre8) — обзорный доклад А. Наумова (ВШЭ), 2021. Ещё одна запись доклада с тем же названием: [видео](https://www.youtube.com/watch?v=KGhaKh1Of-4).
- Курс «Теория случайных матриц», Т. Щербина (лаборатория им. Чебышева): [лекция 1](https://www.youtube.com/watch?v=sTV2CisOhOw), [лекция 2](https://www.youtube.com/watch?v=MDrSYypfpbw) — строгий математический курс.

**Задания 1–2: Марченко–Пастур, Трейси–Видом, BBP (на английском):**

- [The Marchenko–Pastur Distribution: The Spectrum of Noise — RMT (2/7)](https://www.youtube.com/watch?v=ZikTFMudwVg) — вторая часть серии из 7 видео по теории случайных матриц; остальные части ищите на том же канале.

**Задание 3: тяжёлые хвосты и WeightWatcher (на английском, автор теории — Чарльз Мартин):**

- [Why Deep Learning Works: Implicit Self-Regularization in Deep Neural Networks](https://www.youtube.com/watch?v=2qF8TezRwS0) — доклад 2019 года, ровно та теория HTSR, что в лабе.
- [Доклад Мартина в LBNL, 2018](https://www.youtube.com/watch?v=_Ni5UDrVwYU) — более ранняя версия того же доклада.
- [WeightWatcher: Data-Free Diagnostics for Deep Learning](https://www.youtube.com/watch?v=NXqO4nDNIwo) — как по спектру весов диагностировать сеть без данных.
