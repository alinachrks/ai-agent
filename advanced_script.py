import requests
import numpy as np
import pandas as pd

# Запрос данных о погоде в Лондоне (пример API-запроса)
response = requests.get("https://wttr.in/London?format=%t")
temperature = response.text.strip()

# Создание тестового DataFrame
data = {
    "City": ["London", "New York", "Tokyo"],
    "Temperature": [temperature, np.random.randint(-5, 35), np.random.randint(-10, 40)]
}
df = pd.DataFrame(data)

# Сохранение в CSV
df.to_csv("weather_data.csv", index=False)

print("Weather data saved successfully:")
print(df)
