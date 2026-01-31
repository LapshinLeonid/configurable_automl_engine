import os
import pytest

# Получаем абсолютный путь к текущему файлу
current_file = os.path.abspath(__file__)
# Получаем путь к директории, где лежит run_tests.py
test_dir = os.path.dirname(current_file)

# Меняем текущую рабочую директорию на директорию с тестами
os.chdir(test_dir)


# Запускаем тесты в текущей директории
pytest_args = ["-v","-s", 
                "test_train_model.py",
               "test_algorithms_extended.py",
               "test_algorithms.py",
               "test_hyperopt_module.py",
               "test_objective.py",
               "test_validation.py",
               "test_component.py",
               "test_pipeline_with_fractional_os.py",
               "test_validation_strategy.py",
               "test_config_validation.py",
               "test_e2e_pipeline.py",
               "test_metrics.py",
               "test_oversampling.py",
               "test_parallel.py"
               ]

pytest.main(pytest_args)
