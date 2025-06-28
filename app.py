import streamlit as st
import pandas as pd
import logging
from utils import analyze_data

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, filename="app.log", filemode="w")
logger = logging.getLogger(__name__)

# Проверка запуска Streamlit
logger.debug("Запуск приложения Streamlit")
st.set_page_config(page_title="ReGenerator Datasets", layout="wide")
st.title("ReGenerator Datasets")

# Тестовый текст для проверки рендеринга
st.write("Добро пожаловать в ReGenerator Datasets! Если вы видите это сообщение, интерфейс работает.")

# Шаг 1: Загрузка CSV-файла
st.header("Загрузка данных")
uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"], help="Поддерживаются файлы до 1 ГБ")
logger.debug("Отображен загрузчик файлов")

if uploaded_file:
    logger.debug("Файл загружен, начинаем чтение CSV")
    try:
        df = pd.read_csv(uploaded_file, dtype_backend="numpy_nullable", low_memory=False)
        logger.debug("CSV успешно прочитан")
        st.success("Файл успешно загружен!")
        st.subheader("Предпросмотр данных")
        st.dataframe(df.head(), use_container_width=True)
        
        # Сохранение датасета для дальнейшей работы
        st.session_state["dataframe"] = df
        logger.debug("Датасет сохранен в session_state")
        
        # Шаг 2: Анализ данных
        st.header("Анализ данных")
        with st.spinner("Анализируем данные..."):
            analysis = analyze_data(df)
            logger.debug("Анализ данных завершен")
        
        # Отображение результатов анализа
        if "error" not in analysis:
            st.subheader("Результаты анализа")
            st.write("**Типы данных столбцов:**")
            st.json(analysis["dtypes"])
            st.write("**Количество пропущенных значений (NULL):**")
            st.json(analysis["null_counts"])
            st.write("**Количество уникальных значений:**")
            st.json(analysis["unique_counts"])
            st.write("**Смысловой контекст столбцов:**")
            st.json(analysis["context"])
        else:
            st.error(analysis["error"])
            logger.error(analysis["error"])
            
    except Exception as e:
        st.error(f"Ошибка при загрузке или анализе файла: {str(e)}")
        logger.error(f"Ошибка при загрузке или анализе файла: {str(e)}")