set -e

echo "===================================================================="
echo "   ЗАПУСК ОКРУЖЕНИЯ ALFA-BANK TRANSACTION PREDICTION"
echo "===================================================================="

echo ""
echo "--- Шаг 1: Проверка/Настройка NVIDIA Container Toolkit ---"
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "GPU не обнаружен в Docker. Установка драйверов..."
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo "Проверка GPU..."
    docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
else
    echo "NVIDIA Container Toolkit уже настроен."
fi

echo ""
echo "--- Шаг 2: Сборка и запуск контейнера ---"
docker compose down
docker compose up -d --build

if [ $? -ne 0 ]; then
    echo "Ошибка сборки Docker образа."
    exit 1
fi
echo "Запуск тестов."
docker compose run --rm alfa-solution pytest tests/

echo "Контейнер запущен в фоне."

echo ""
echo "========================================================================"
echo "   ИНСТРУКЦИЯ ПО ЗАПУСКУ (внутри контейнера)"
echo "========================================================================"
echo "Вы сейчас попадете в консоль контейнера. Команды:"
echo ""
echo "1. Подготовка данных (Сплит, SVD, Словари):"
echo "   python scripts/1_prepare_data.py"
echo ""
echo "2. Обучение модели (Опционально):"
echo "   python scripts/2_train.py"
echo ""
echo "3. Поиск порогов на валидации:"
echo "   python scripts/3_optimize.py"
echo ""
echo "4. Финальный инференс и сабмит:"
echo "   python scripts/4_predict.py"
echo ""
echo "5. Генерация отчетов о важности признаков:"
echo "   python scripts/5_explain.py"
echo ""
echo "========================================================================"
echo "Вход в контейнер..."

docker compose exec alfa-solution bash

echo ""
echo "--- Сессия завершена. Останавливаем контейнер... ---"
docker compose down