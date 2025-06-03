# promt_generator

**Описание проекта**  
Репозиторий **promt_generator** представляет собой генератор текстовых подсказок (prompt generator) для моделей генерации изображений и текста (Stable Diffusion, Midjourney, DALL·E и др.). Задача проекта — на основе пользовательского ввода (на русском языке) автоматически формировать «правильный» англоязычный промт, объединяя основные теги, стилевые категории, имена художников и дополнительные ключевые слова.

---

## Структура репозитория  
```text
promt_generator/
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
│   ├── translator.py               # Логика перевода с русского на английский (через API или local-model)
│   ├── tag_extractor.py            # Алгоритмы извлечения ключевых слов из русского текста
│   ├── style_logic.py              # Обработка стилистики: добавление стилей, выбор художников
│   ├── prompt_builder.py           # Основная логика сборки финального промта
│   └── main.py                     # Точка входа: пример запуска из командной строки и/или API
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
   venv\Scripts\activate             # на Windows
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
   Откройте `.env` и пропишите необходимые ключи:
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

## Конфигурация

- В `config.py` задаются основные константы:
  - **Пути** к JSON-файлам со словарями стилей и художников
  - **Параметры** по умолчанию: максимальное число тегов, порядок включения элементов в промт
  - **API-энды**: URL для переводчика, параметры rate‐limit и т. д.

- Файлы `style_categories.json` и `style_artists.json` лежат в папке `data/`:
  ```jsonc
  // Пример style_categories.json
  {
    "Cyberpunk": ["cyberpunk", "neon", "futuristic", "sci-fi"],
    "Renaissance": ["Renaissance", "classical painting", "oil painting"],
    // ...
  }
  ```
  ```jsonc
  // Пример style_artists.json
  {
    "Impressionism": ["Claude Monet", "Pierre-Auguste Renoir"],
    "Surrealism": ["Salvador Dali", "René Magritte"],
    // ...
  }
  ```

- При необходимости добавьте собственные файлы в папку `data/`:
  - `popular_tags.json` (если хотите хранить локальную статистику наиболее часто применяемых тегов)
  - Любые другие вспомогательные справочники.

---

## Запуск и использование

### 1. Из командной строки (CLI)

Файл `src/main.py` выполняет следующие шаги:

1. Считывает текстовый файл / строку на русском языке
2. Извлекает ключевые слова и «сущности» (tag_extractor.py)
3. Переводит текст и/или отдельные теги на английский (translator.py)
4. Определяет стиль(и) и подбирает художников (style_logic.py)
5. Собирает финальный промт (prompt_builder.py) в формате:
   ```
   <translated_text>, <tags>, <style_tags>, by <artist>, high quality, 8k, cinematic lighting
   ```

#### Пример запуска:
```bash
python src/main.py \
  --input examples/example_input.txt \
  --output examples/example_output.txt \
  --use-openai-translate             # флаг: перевод через OpenAI API
```

**Параметры CLI** (описаны в `main.py`):
- `--input <путь>` — входной файл с русским текстом (UTF-8)
- `--output <путь>` — файл, куда будет записан результат (англоязычный промт)
- `--direct-text "<строка>"` — вместо `--input` можно передать текст напрямую
- `--use-openai-translate` — принудительно использовать OpenAI API для перевода (если есть ключ)
- `--use-local-translate` — перевод через локальную модель Hugging Face (если это настроено)
- `--max-tags <число>` — максимальное количество автоматически извлекаемых тегов (по умолчанию 5)
- `--verbose` — подробный режим, вывод в консоль всех промежуточных шагов

---

### 2. Как библиотека (импорт в свой проект)

```python
from src.translator import Translator
from src.tag_extractor import TagExtractor
from src.style_logic import StyleLogic
from src.prompt_builder import PromptBuilder

# 1) Настройка
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

# 2) Сборка промта
russian_input = "Светлый киберпанк-город ночью с неоновыми вывесками и футуристическими автомобилями"
prompt = prompt_builder.build(russian_input)

print("Сгенерированный промт:")
print(prompt)
# Выведет, например:
# "A bright cyberpunk city at night with neon signs and futuristic vehicles, neon, cityscape, night, futuristic, cyberpunk, by Syd Mead, ultra-detailed, 8K, cinematic lighting"
```

---

## Логика работы модулей

1. **translator.py**  
   - Абстракция над сервисом перевода с русского на английский.  
   - Поддерживаются два «режима»:  
     1. **OpenAI API** (ChatGPT или GPT-4-подобные модели для высококачественного перевода)  
     2. **Hugging Face Transformers** (локальный модельный стек)  
   - Возвращает строку на английском языке.

2. **tag_extractor.py**  
   - Использует простые NLP‐приёмы (разбиение на токены, стоп-слова) или внешние библиотеки (NLTK/spaCy) для извлечения ключевых слов.  
   - Опционально: использование готовых моделей Named Entity Recognition (NER) для выделения сущностей (юзер-имена, география, стилистика).  
   - На выходе — список «грубых» тегов на русском, которые потом переводятся.

3. **style_logic.py**  
   - Загружает файлы `style_categories.json` и `style_artists.json`.  
   - По ключевым словам пытается определить, к какому стилю они ближе (простое сравнение множества тегов) или позволяет вручную задать стиль через параметр.  
   - Выбирает 1–2 подходящих художников из категории (если совпадает стиль) и возвращает их имена.

4. **prompt_builder.py**  
   - Объединяет перевод оригинального текста, переведённые теги, стилевые теги, имена художников и стандартные «флаеры» (как «high quality», «8k», «ultra detailed», «cinematic lighting»).  
   - Поддерживает шаблонную вставку:  
     ```
     {translated_text}, {translated_tags}, {style_tags}, by {artists}, {suffix}
     ```  
   - Возвращает одну строку — готовый англоязычный промт.

5. **main.py**  
   - Обрабатывает аргументы командной строки, инициализирует все компоненты, запускает процедуру генерации и записывает результат в файл или выводит на экран.

---

## Примеры

1. **Пример входа** (`examples/example_input.txt`):
   ```
   Сделать атмосферу тёмного фэнтези: заброшенный замок, мрачное небо, призрачные тени.
   ```
2. **Пример выхода** (`examples/example_output.txt`):
   ```
   Dark fantasy atmosphere: abandoned castle, gloomy sky, ghostly shadows, fantasy, castle, dark, eerie, medieval, by H. R. Giger, ultra-detailed, 4K, dramatic lighting
   ```

3. **Запуск напрямую:**
   ```bash
   python src/main.py --direct-text "Яркий футуристический пейзаж с оживлёнными улицами в стиле киберпанк." --output result.txt
   ```
   В `result.txt` будет что-то вроде:
   ```
   Bright futuristic landscape with bustling streets in a cyberpunk style, bright, futuristic, city, neon, cyberpunk, by Syd Mead, ultra–detailed, 8K, neon glow
   ```

---

## Тестирование

В репозитории предусмотрена директория `tests/` с набором модульных тестов (pytest). Для запуска:

1. Убедитесь, что виртуальное окружение активировано.  
2. Установлены зависимости (в том числе `pytest`).  
3. Выполните в корне проекта:
   ```bash
   pytest --verbose
   ```
   Все тесты должны пройти без ошибок. Если какой-то тест завалился, проверьте логику соответствующего модуля или обновите тестовые данные.

---

## Развитие и вклад в проект

1. **Форк** репозитория и создание новой ветки (`git checkout -b feature/your_feature`).  
2. Внедрение изменений / правок / новых функций.  
3. Написание (или корректировка) модульных тестов в `tests/`.  
4. Открытие Pull Request’а в основной репозиторий.  
   - В описание PR приложите:  
     - Краткое описание сути изменений  
     - Скриншоты и/или примеры работы (если есть интерфейс)  
     - Обновлённые или новые тесты  

**Рекомендации по ведению веток**  
- `main` (или `master`) — всегда «чистая» ветка с рабочим кодом.  
- `develop` — основная ветка для слияния функций перед релизом (необязательно, но удобно для командной работы).  
- `feature/<описание>` — для разработки новых возможностей.  
- `bugfix/<описание>` — для исправления конкретных багов.  
- После проверки CI и прохождения всех тестов PR сливается в `main`.

---

## Чаще задаваемые вопросы (FAQ)

1. **Можно ли использовать локальную модель перевода вместо OpenAI?**  
   Да. В `translator.py` реализованы оба варианта. Установите `transformers` и скачайте нужную модель. В `.env` укажите `USE_LOCAL_TRANSLATOR=true`, и `Translator` автоматически переключится на локальную модель Hugging Face.

2. **Как добавить новый стиль или художника?**  
   - Откройте `data/style_categories.json` и добавьте новую категорию:  
     ```json
     "Vintage": ["vintage", "analogue", "film grain"]
     ```  
   - В `data/style_artists.json` под тем же ключом укажите список художников:  
     ```json
     "Vintage": ["Ansel Adams", "Henri Cartier-Bresson"]
     ```
   - Затем при анализе текста `style_logic.py` при нахождении слова из ключей категории «Vintage» будет подтягивать их в промт.

3. **Какие настройки параметра `max_tags` лучше использовать?**  
   По умолчанию выставлено 5, чтобы не перегружать промт слишком большим количеством ключевых слов. Но если хотите более «детализированные» подсказки, можно поднять до 7–10.

4. **Как модифицировать шаблон финального промта?**  
   Откройте `prompt_builder.py` и найдите метод `build()`. Там используется строковый шаблон с плейсхолдерами `{translated_text}`, `{tags}`, `{style_tags}`, `{artists}`, `{suffix}`. Подкорректируйте порядок или добавьте собственные приставки, например «vivid colors», «depth of field» и т. д.

---

## Лицензия

Проект распространяется под лицензией **MIT License**. Полный текст лицензии находится в файле `LICENSE`. Кратко: вы вправе использовать, копировать, изменять и распространять проект, указывая авторство.

---

## Контакты автора

- **GitHub**: [NocturnaExtreme](https://github.com/NocturnaExtreme)  
- **Email**: nocturna.extreme@example.com (замените на актуальный при необходимости)  
- **Telegram/Discord**: @NocturnaExtreme  

Буду рад вашим вкладкам (issues, PR) и предложениям по улучшению функционала!

---

> **Примечание**:  
> Данный документ подготовлен как шаблон «правильной» документации для Git-репозитория. При необходимости адаптируйте пути к файлам и конкретные описания модулей в соответствии с актуальным содержимым вашего кода.
