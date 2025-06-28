import pandas as pd
from llama_cpp import Llama
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, filename="app.log", filemode="w")
logger = logging.getLogger(__name__)

def analyze_data(df):
    """
    Анализирует структуру и контекст данных в DataFrame.
    Возвращает словарь с типами данных, количеством NULL, уникальных значений и контекстом столбцов.
    """
    logger.debug("Начало анализа данных")
    analysis = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "unique_counts": df.nunique().to_dict(),
        "context": {}
    }
    
    # Путь к локальной модели GGUF
    model_path = "./google_model_llm/gemma-3-4b-it-Q4_K_M.gguf"
    logger.debug(f"Инициализация модели из {model_path}")
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена в {model_path}. Убедитесь, что модель gemma-3-4b-it-GGUF загружена в папку google_model_llm."
            )
        # Инициализация модели с llama.cpp
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Контекстное окно
            n_threads=4,  # Количество потоков (настройте под ваш CPU/GPU)
            n_gpu_layers=-1 if os.uname().sysname == "Darwin" else 0  # Используем MPS на macOS
        )
        logger.debug("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        return {"error": f"Ошибка загрузки модели: {str(e)}"}
    
    logger.debug("Анализ контекста столбцов")
    for column in df.columns:
        sample = df[column].dropna().iloc[:5].tolist()
        if sample:
            prompt = f"Определи смысловой контекст столбца на основе примеров: {sample}. Кратко опиши, что содержит столбец (например, имена, адреса, даты)."
            try:
                response = llm(
                    prompt,
                    max_tokens=50,
                    temperature=0.7,
                    stop=["\n"],
                )
                context = response["choices"][0]["text"].strip()
                analysis["context"][column] = context
                logger.debug(f"Контекст для столбца {column}: {context}")
            except Exception as e:
                analysis["context"][column] = f"Ошибка анализа контекста: {str(e)}"
                logger.error(f"Ошибка анализа контекста столбца {column}: {str(e)}")
        else:
            analysis["context"][column] = "Нет данных для анализа"
            logger.debug(f"Столбец {column} пустой")
    
    logger.debug("Анализ данных завершен")
    return analysis