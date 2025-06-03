# Prompt Generator

**Описание проекта**  
Репозиторий **promt_generator** — это генератор текстовых подсказок (prompt generator) для моделей генерации изображений и текста (Stable Diffusion, Midjourney, DALL·E и др.). Проект принимает на вход текст на русском языке, автоматически переводит его на английский, извлекает ключевые теги, подбирает стили и имена художников, и формирует «правильный» англоязычный промт.

Кроме использования в качестве библиотеки или через CLI, проект включает файл **app.py** (в корне), позволяющий запустить веб‑интерфейс (или локальный API) для интерактивного ввода текста и получения промта.

---

## Структура репозитория  
```text
promt_generator/
├── app.py                          # Точка входа: веб-приложение / API для генерации промтов
├── data/
│   ├── style_categories.json       # Словарь STYLE_CATEGORIES с категориями и тегами
│   ├── style_artists.json          # Словарь STYLE_ARTISTS с привязкой художников к стилям
│   └── popular_tags.json           # (опционально) Часто используемые теги для подсказок
│
├── examples/
│   ├── example_input.txt           # Пример входного текста (на русском)
│   └── example_output.txt          # Сгенерированный промт (на английском)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Константы, пути к файлам, параметры по умолчанию
│   ├── translator.py               # Логика перевода с русского на английский
│   ├── tag_extractor.py            # Алгоритмы извлечения ключевых слов из русского текста
│   ├── style_logic.py              # Обработка стилистики: добавление стилей, выбор художников
│   ├── prompt_builder.py           # Основная логика сборки финального промта
│   └── main.py                     # Точка входа для CLI (пример без веб)
│
├── tests/
│   ├── test_translator.py          # Модульные тесты для переводчика
│   ├── test_tag_extractor.py       # Модульные тесты для извлечения тегов
│   └── test_prompt_builder.py      # Модульные тесты для сборки промта
│
├── requirements.txt                # Список зависимостей (Python-библиотеки)
├── .env.example                    # Пример файла с переменными окружения (API-ключи, пути)
├── README.md                       # Этот файл
└── LICENSE                         # Лицензия проекта (MIT/BSD/GPL и т. п.)
```

---

## Требования  
1. **Python ≥ 3.8**  
2. Виртуальное окружение (рекомендуется `venv` или `conda`)  
3. Библиотеки, указанные в `requirements.txt`  
   ```text
   python-dotenv
   requests
   transformers          # если используется локальная модель перевода
   openai                # если перевод через OpenAI API
   fastapi               # если app.py на основе FastAPI
   uvicorn               # для запуска FastAPI
   streamlit             # если app.py реализован на Streamlit
   pytest                # для запуска тестов
   ```
4. (Опционально) Доступ к интернету для обращения к внешним API (OpenAI, Hugging Face и т. д.).

---

## Установка

1. **Клонируйте репозиторий**  
   ```bash
   git clone https://github.com/NocturnaExtreme/promt_generator.git
   cd promt_generator
   ```

2. **Создайте и активируйте виртуальное окружение**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate          # на macOS/Linux
   venv\Scripts\activate            # на Windows
   ```

3. **Установите зависимости**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Скопируйте файл окружения и заполните переменные**  
   ```bash
   cp .env.example .env
   ```
   Затем откройте `.env` и пропишите переменные, например:
   ```
   # API-ключ для перевода (если используете OpenAI)
   OPENAI_API_KEY=ваш_openai_api_key

   # Или, если используете Hugging Face Translation
   HUGGINGFACE_TOKEN=ваш_huggingface_token

   # Другие настройки (при необходимости)
   DEFAULT_LANGUAGE=ru
   OUTPUT_LANGUAGE=en
   ```

---

## Запуск приложения

В корне проекта находится файл **app.py**, который организует работу веб‑интерфейса или локального API. В зависимости от выбранной технологии, используйте одну из команд:

### 1. Если **app.py** реализован на FastAPI  
1. Убедитесь, что `fastapi` и `uvicorn` установлены (они указаны в `requirements.txt`).  
2. Запустите приложение командой:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
3. Перейдите в браузере по адресу [http://localhost:8000/docs](http://localhost:8000/docs), чтобы увидеть автоматически сгенерированную документацию Swagger UI.  
4. В разделе `/generate` передайте JSON вида:
   ```json
   {
     "text": "Ваш текст на русском языке"
   }
   ```
   В ответ получите сгенерированный англоязычный промт.

### 2. Если **app.py** реализован на Streamlit  
1. Убедитесь, что `streamlit` установлен.  
2. Запустите приложение командой:
   ```bash
   streamlit run app.py
   ```
3. В браузере автоматически откроется веб‑интерфейс. Введите текст на русском языке в поле ввода, нажмите «Submit» (или аналогичную кнопку), и внизу появится готовый промт на английском.

> **Важно**: Замените в коде `app.py` (или через переменные окружения) `YOUR_OPENAI_API_KEY` на ваш реальный ключ OpenAI, если приложение использует OpenAI API для перевода. Или настройте локальный перевод через Hugging Face.

---

## Пример работы

1. Запустите приложение (FastAPI или Streamlit) в режиме разработки.  
2. Введите в веб‑интерфейсе:  
   ```
   Тёмный фэнтезийный лес с туманом, руинами и мистическими существами.
   ```
3. В ответе получите что‑то вроде:  
   ```
   Dark fantasy forest with mist, ruins, and mystical creatures, fantasy, forest, dark, mist, ruins, by J. R. R. Tolkien, ultra-detailed, 8K, cinematic lighting
   ```

---

## Как библиотека (CLI)

Если вы предпочитаете CLI или интеграцию в свой код, используйте модуль **main.py**:

```bash
python src/main.py \
  --input examples/example_input.txt \
  --output examples/example_output.txt \
  --use-openai-translate
```

Или программно:

```python
from src.translator import Translator
from src.tag_extractor import TagExtractor
from src.style_logic import StyleLogic
from src.prompt_builder import PromptBuilder

translator = Translator(api_key="YOUR_OPENAI_KEY")
tag_extractor = TagExtractor()
style_logic = StyleLogic(
    style_categories_path="data/style_categories.json",
    style_artists_path="data/style_artists.json"
)
prompt_builder = PromptBuilder(
    translator=translator,
    tag_extractor=tag_extractor,
    style_logic=style_logic,
    max_tags=5
)

russian_input = "Яркий киберпанк-город ночью"
prompt = prompt_builder.build(russian_input)
print(prompt)
```

---

## Тестирование

Для запуска модульных тестов выполните:

```bash
pytest --verbose
```

Убедитесь, что виртуальное окружение активно и установлены зависимости.

---

## Развитие и вклад в проект

1. **Форк** репозитория и создание ветки `feature/your_feature`.  
2. Внесение изменений и добавление тестов.  
3. Создание Pull Request’а с описанием изменений и примерами работы.  

---

## Лицензия

Проект распространяется под лицензией **MIT License**. Полный текст лицензии находится в файле `LICENSE`.  
