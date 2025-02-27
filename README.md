# README: Анализ базы данных с помощью Gemini API

## 📌 Описание проекта
Этот проект предназначен для автоматического анализа базы данных, формирования аналитических отчётов и выявления ключевых закономерностей в данных с помощью API Gemini. Он анализирует структуру таблиц, выявляет основные тренды и предлагает рекомендации для инженеров данных и аналитиков.

## 🚀 Основные возможности
- Подключение к базе данных и анализ структуры таблиц.
- Обнаружение ключевых статистических показателей и закономерностей.
- Генерация аналитического отчёта с помощью **Gemini API**.
- Обнаружение проблем в данных (пропуски, дубли, ошибки).
- Выдача рекомендаций по обработке данных и их применению.

## 📂 Структура проекта
```
├── main.py                  # Основной скрипт анализа
├── reports_db.docx          # Итоговый отчёт в формате Word
├── error_log.txt            # Лог ошибок
├── requirements.txt         # Список зависимостей
└── README.md                # Данный файл
```

## ⚙️ Установка и запуск
### 1. Установка зависимостей
Убедитесь, что у вас установлен Python 3.7+.

```bash
pip install -r requirements.txt
```

### 2. Настройка переменных
Создайте файл `.env` и добавьте API-ключ Gemini:
```
GEMINI_API_KEY=ваш_ключ
```

### 3. Запуск анализа
```bash
python main.py
```

## 🛠 Используемые технологии
- **Python 3.7+**
- **SQLAlchemy** — для взаимодействия с базой данных
- **pandas** — для обработки данных
- **docx (python-docx)** — для формирования отчётов
- **requests** — для работы с API Gemini

## 📖 Как работает проект?
1. Подключается к базе данных и извлекает список таблиц.
2. Перебирает таблицы пачками (по 2 таблицы за раз).
3. Определяет структуру и ключевые характеристики данных.
4. Отправляет запрос в API Gemini для генерации аналитического отчёта.
5. Сохраняет результат в формате Word (`reports_db.docx`).
6. В случае ошибок записывает их в `error_log.txt`.

## 🔍 Основные параметры
| Параметр | Описание |
|----------|------------|
| `TABLES_PER_BATCH` | Количество таблиц для анализа за один цикл (по умолчанию: 2) |
| `GEO_TYPES` | Список геометрических типов данных, которые обрабатываются отдельно |
| `report_filename` | Имя итогового файла отчёта |

## 📌 Возможные ошибки и их решения
| Ошибка | Возможное решение |
|--------|------------------|
| `Ошибка подключения к базе` | Проверьте параметры соединения в коде |
| `Ошибка в запросе к API` | Проверьте API-ключ и лимиты на запросы |
| `Документ не сохраняется` | Закройте `reports_db.docx` перед запуском |

## 📈 Будущие улучшения
- Улучшение анализа аномалий в данных.
- Визуализация отчётов (графики и диаграммы).
- Автоматическое определение зависимостей между таблицами.

---
**Авторы:** [Ваше имя]  
📅 Дата последнего обновления: *2025-02-27*

