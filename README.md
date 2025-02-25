# 📊 Автоматическое обновление кода с использованием Gemini API

## 📝 Описание проекта
Этот проект предназначен для автоматизации процесса обновления Python-кода с использованием **Gemini API** для улучшения и оптимизации существующих моделей и алгоритмов машинного обучения. Он включает в себя механизм для **загрузки данных**, их **предобработки**, **обучения моделей**, а также **автоматического обновления кода** с предложениями от **Gemini API**.

Проект интегрирован с платформой **OpenML** для загрузки различных наборов данных и использует **scikit-learn** для обучения моделей. **Каждый час или при изменении данных** запускается процесс, который автоматически обновляет код, генерирует сообщения о коммитах и применяет улучшения с помощью **Gemini API**.

---

## 🚀 Ключевые особенности
✅ **Автоматизация обновлений**: Полностью автоматизированный процесс обновления модели и кода без необходимости вручного редактирования.

✅ **Поддержка различных моделей**: Включает поддержку нескольких алгоритмов ML, таких как **RandomForest**, **LogisticRegression**, что позволяет гибко работать с задачами классификации и регрессии.

✅ **Гибкость работы с данными**: Интеграция с **OpenML** позволяет загружать и анализировать разные датасеты **без изменения кода**.

✅ **API-интеграция**: Использование **Gemini API** для автоматического улучшения кода, оптимизации алгоритмов и структур данных.

✅ **Автоматическое коммитирование**: После обновления кода проект **генерирует сообщения коммитов** и **автоматически пушит** их в репозиторий.

---

## 🛠 Используемые технологии и инструменты
- **Python 3.8+**: Язык программирования.
- **Gemini API**: Автоматические улучшения и оптимизация кода.
- **OpenML**: Источник наборов данных.
- **scikit-learn**: Обучение и тестирование ML-моделей.
- **GitHub Actions**: Автоматизация обновлений и развертывания кода.

---

## 📂 Структура проекта
```
├── .github/workflows/auto_commit.yml  # Автоматическое обновление кода через GitHub Actions
├── src/                               # Основные скрипты проекта
│   ├── main.py                        # Главный скрипт: загрузка данных, предобработка, обучение модели, обновление кода
│   ├── model_training.py               # Скрипт для обучения моделей и оценки результатов
│   ├── data_analysis.py                # Анализ данных и визуализация
│   ├── commit_message.py               # Генерация сообщений коммитов через Gemini API
│   ├── __init__.py                      # Позволяет использовать src как модуль
├── requirements.txt                     # Зависимости проекта
├── README.md                            # Описание проекта и инструкции по использованию
```

---

## 📦 Установка и запуск
### 🔹 1. Клонировать репозиторий
```bash
git clone git@github.com:alinachrks/ai-agent.git
cd ai-agent
```

### 🔹 2. Создать и активировать виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate  # Для macOS/Linux
venv\Scripts\activate    # Для Windows
```

### 🔹 3. Установить зависимости
```bash
pip install -r requirements.txt
```

### 🔹 4. Добавить API-ключ Gemini в переменные среды
```bash
export GEMINI_API_KEY="your-google-gemini-api-key"
```

### 🔹 5. Запустить анализ данных и обновление кода
```bash
python src/main.py
```

---

## ⚙️ Автоматизация через GitHub Actions
Этот проект автоматически обновляет код **каждый час** или при изменении данных с помощью **GitHub Actions**.

Workflow `.github/workflows/auto_commit.yml`:
- **Запускается раз в час** (`cron: '0 * * * *'`).
- **Автоматически обновляет код и пушит его в репозиторий**.
- **Генерирует осмысленные сообщения коммитов через Gemini API**.

### 🔑 **Настройка GitHub Secrets**
1. Перейти в **Settings → Secrets and variables → Actions**.
2. Добавить новый секрет:
   - **Имя:** `GEMINI_API_KEY`
   - **Значение:** ваш API-ключ Gemini.

После этого GitHub Actions автоматически обновит код.

---

## 🔥 Возможные улучшения
✅ Поддержка **других ML-фреймворков** (например, TensorFlow, PyTorch).  
✅ Оптимизация **подбора гиперпараметров моделей**.  
🔄 Добавление **AutoML-пайплайна** для более продвинутой автоматизации.  
📈 Интеграция с **Jupyter Notebook** для интерактивного анализа.  

---

## 🤝 Контрибьютинг
Хотите внести вклад?  
Форкайте репозиторий, создавайте **Pull Request**!  
Поддержите проект ⭐ и следите за обновлениями. 🚀
