import sys
import os
import numpy as np
import pandas as pd
import threading

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
from typing import Union, Any, Optional, Dict
from datetime import datetime
from pathlib import Path
from math import ceil
from sklearn.neighbors import NearestNeighbors



class DataOversampler:
    """
    Потокобезопасный класс для увеличения объёма и / или балансировки данных.

    Изменения
    ---------
    • multiplier: float ≥ 1 (можно дробный, например 1.2).  
      Целевой размер каждого класса = ceil(multiplier × исходный_размер).  
      multiplier < 1 → ValueError.
    • Реализован method ``fit_resample`` (imblearn‑API).
    • Конструктор принимает multiplier / algorithm / add_noise, чтобы
      ``ModelTrainer`` мог передать их напрямую.

    Поддерживаемые алгоритмы
    ------------------------
      - "random" | "random_with_noise"
      - "smote"
      - "adasyn"

    Логи INFO / ERROR пишутся в <log_dir>/oversampler.log
    """

    # ------------------------------------------------------------------ #
    #  Инициализация                                                     #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        multiplier: float | int = 1,
        algorithm: str = "random",
        add_noise: bool = False,
        n_jobs: int = 1,
        log_dir: Optional[str] = None,
    ):
        self.multiplier = float(multiplier)
        self.algorithm = algorithm.lower().replace(" ", "_")
        self.add_noise = add_noise

        self.n_jobs = n_jobs
        self._lock = threading.RLock()

        # лог‑директория
        self.log_dir = log_dir or self._get_default_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "oversampler.log")

    # ------------------------------------------------------------------ #
    #  Внутренние утилиты                                                #
    # ------------------------------------------------------------------ #
    def _get_default_log_dir(self) -> str:
        root = Path(__file__).resolve().parents[2]
        return str(root / "logs")

    def _log(self, message: str, error: bool = False) -> None:
        level = "ERROR" if error else "INFO"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame = sys._getframe(2)
        fname = os.path.basename(frame.f_code.co_filename)
        func = frame.f_code.co_name
        line = f"[{ts}] [{fname}:{func}] {level}: {message}\n"
        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                print(f"Не удалось записать лог: {e}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    #  Проверка входных данных                                           #
    # ------------------------------------------------------------------ #
    def _validate(self, data: Any, multiplier: float) -> pd.DataFrame:
        if not isinstance(multiplier, (int, float)) or multiplier < 1:
            raise ValueError("multiplier должен быть числом ≥ 1 (целым или дробным).")

        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, (np.ndarray, list)):
            arr = np.asarray(data)
            if arr.ndim != 2:
                raise ValueError(
                    "Массив/список должен быть двумерным: samples × features(+label)."
                )
            df = pd.DataFrame(arr)
        else:
            raise TypeError("data должен быть pandas DataFrame, numpy array или list.")

        if df.shape[1] < 2:
            raise ValueError("Нужно минимум два столбца: признаки + метка.")

        return df

    # ------------------------------------------------------------------ #
    #  Подсчёт целевых размеров                                          #
    # ------------------------------------------------------------------ #
    def _strategy(self, y: pd.Series, multiplier: float) -> Dict[Any, int]:
        counts = Counter(y)
        return {cls: ceil(cnt * multiplier) for cls, cnt in counts.items()}

    @staticmethod
    def _add_noise(X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        return X + np.random.normal(scale=noise_level, size=X.shape)

    # ------------------------------------------------------------------ #
    #  Основная «рабочая» функция                                        #
    # ------------------------------------------------------------------ #
    def _resample(
        self,
        df: pd.DataFrame,
        multiplier: float,
        algo: str,
        add_noise: bool,
        target_column: str | None = None,
    ) -> pd.DataFrame:
        # --- разделяем признаки / метку
        if target_column:
            y = df[target_column]
            X = df.drop(target_column, axis=1)
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

        counts = Counter(y)

        # Corner‑case: dataset с одним классом
        if len(counts) == 1:
            n_target = ceil(len(df) * multiplier)
            n_extra = n_target - len(df)
            if n_extra <= 0:
                return df.reset_index(drop=True)

            idx_extra = np.random.choice(df.index, size=n_extra, replace=True)
            res = pd.concat([df, df.loc[idx_extra]], ignore_index=True)
            return res.reset_index(drop=True)

        # Общее
        strategy = self._strategy(y, multiplier)
        min_count = min(counts.values())
        k_neighbors = max(1, min(5, min_count - 1))
        nn_estimator = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=self.n_jobs)

        # ---------------------------------------------------------------- #
        #  RANDOM oversampling                                              #
        # ---------------------------------------------------------------- #
        if algo == "random":
            # целевой общий размер датасета
            n_target = ceil(len(df) * multiplier)
            n_extra  = n_target - len(df)

            if n_extra > 0:
                # детерминированно повторяем исходные индексы,
                # чтобы вся выборка (включая соотношение классов) умножилась на multiplier
                base_idx  = df.index.to_numpy()
                reps      = (n_extra + len(base_idx) - 1) // len(base_idx)
                idx_extra = np.tile(base_idx, reps)[:n_extra]

                X_res = pd.concat([X, X.loc[idx_extra]], ignore_index=True)
                y_res = pd.concat([y, y.loc[idx_extra]], ignore_index=True)
            else:
                # multiplier == 1 или n_extra == 0 — возвращаем без изменений
                X_res, y_res = X.copy(), y.copy()



        # ---------------------------------------------------------------- #
        #  SMOTE                                                           #
        # ---------------------------------------------------------------- #
        elif algo == "smote":
            sm = SMOTE(sampling_strategy=strategy, k_neighbors=nn_estimator)
            X_res, y_res = sm.fit_resample(X, y)

        # ---------------------------------------------------------------- #
        #  ADASYN + ROS                                                    #
        # ---------------------------------------------------------------- #
        elif algo == "adasyn":
            minority = min(counts, key=counts.get)
            majority = max(counts, key=counts.get)
            target = {cls: ceil(cnt * multiplier) for cls, cnt in counts.items()}

            ada = ADASYN(
                sampling_strategy={minority: target[minority]},
                n_neighbors=nn_estimator,
            )
            X_tmp, y_tmp = ada.fit_resample(X, y)

            ros = RandomOverSampler(sampling_strategy={majority: target[majority]})
            X_res, y_res = ros.fit_resample(X_tmp, y_tmp)

        else:
            raise ValueError(f"Неизвестный алгоритм: {algo}")

        # --- добавляем шум при необходимости
        X_arr = X_res.to_numpy()
        if algo == "random" and add_noise:
            X_arr = self._add_noise(X_arr)

        # --- собираем DataFrame обратно
        res_df = pd.DataFrame(X_arr, columns=X.columns)
        if target_column:
            res_df[target_column] = y_res.values
        else:
            res_df[df.columns[-1]] = y_res.values

        return res_df.reset_index(drop=True)

    def fit_resample(
        self,
        X: Union[pd.DataFrame, np.ndarray, list],
        y: Union[pd.Series, np.ndarray, list],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Специальный режим для обучения: для 'random' балансируем min-класс до max-класса,
        остальные алгоритмы идут через общий _resample.
        """
        # 1) Приводим к pandas
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y_s = pd.Series(y) if not isinstance(y, pd.Series) else y.reset_index(drop=True)

        # 2) Спец‑случай для алгоритма "random" (и "random_with_noise")
        if self.algorithm in ("random", "random_with_noise"):
            # балансируем только min‑класс
            ros = RandomOverSampler(random_state=None)
            X_bal, y_bal = ros.fit_resample(X_df, y_s)  

            # добавляем шум, если нужно
            if self.add_noise:
                arr = self._add_noise(X_bal.to_numpy())
                X_bal = pd.DataFrame(arr, columns=X_df.columns)

            # возвращаем DataFrame и Series
            return X_bal, pd.Series(y_bal, name=y_s.name).reset_index(drop=True)

        # 3) Все остальные алгоритмы — идём привычным путём
        df = X_df.reset_index(drop=True)
        df["_label_"] = y_s.values

        res = self._resample(
            df,
            multiplier=self.multiplier,
            algo=self.algorithm,
            add_noise=self.add_noise,
            target_column="_label_",
        )
        y_res = res.pop("_label_")
        return res, y_res


    def oversample(
        self,
        data: Union[pd.DataFrame, np.ndarray, list],
        multiplier: float,
        algorithm: str = "random",
        add_noise: bool = False,
        target: str | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        data        : входные данные (DataFrame / ndarray / list)
        multiplier  : float ≥ 1
        algorithm   : 'random' | 'random_with_noise' | 'smote' | 'adasyn'
        add_noise   : если True и выбран random → добавляет Gaussian‑noise
        target      : имя столбца‑метки (если не последний)
        """
        with self._lock:
            try:
                df = self._validate(data, multiplier)

                key = algorithm.lower().replace(" ", "_")
                if key == "random_with_noise":
                    key = "random"
                    add_noise = True

                res = self._resample(
                    df,
                    multiplier=float(multiplier),
                    algo=key,
                    add_noise=add_noise,
                    target_column=target,
                )
                self._log(
                    f"Оверсемплинг: {len(df)} → {len(res)} строк  "
                    f"(mult={multiplier}, algo='{algorithm}')"
                )
                return res
            except Exception as e:
                self._log(f"Ошибка: {e}", error=True)
                raise

def oversample(
    data: Union[pd.DataFrame, np.ndarray, list],
    multiplier: float,
    algorithm: str = "random",
    add_noise: bool = False,
    target: Optional[str] = None,
    ) -> pd.DataFrame:
    """
    Обёртка над DataOversampler.oversample — позволяет вызывать
    oversample напрямую по имени "oversampling" или "oversample" в конфигах.
    """
    sampler = DataOversampler(
        multiplier=multiplier,
        algorithm=algorithm,
        add_noise=add_noise
    )
    return sampler.oversample(data, multiplier, algorithm, add_noise, target)
