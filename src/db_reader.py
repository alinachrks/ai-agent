import psycopg2
import os
from dotenv import load_dotenv

# Загружаем секреты из .env (локально) или из GitHub Secrets (при CI/CD)
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

def get_table_info(table_name):
    """Читает структуру таблицы и количество записей."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        cur = conn.cursor()

        # Получаем количество строк в таблице
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cur.fetchone()[0]

        # Получаем названия и типы колонок
        cur.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = %s;
        """, (table_name,))
        columns = cur.fetchall()

        cur.close()
        conn.close()

        return row_count, columns

    except Exception as e:
        return f"Ошибка при подключении к базе: {e}"
