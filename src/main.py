import os
from update_code import update_code
from commit_message import generate_commit_message

def main():
    """Основной процесс: обновление кода и фиктивное изменение"""
    
    print("🔄 Начинаем процесс обновления...")

    # Запуск обновления кода (форсированное изменение)
    updated = update_code()
    
    if updated:
        print("✅ Код был обновлён!")
    else:
        print("⚠ Код не изменился, добавляем фиктивное изменение...")
        with open("src/dummy.txt", "w") as f:
            f.write("Auto-update marker\n")

    # Генерация сообщения коммита
    commit_msg = generate_commit_message("Проверка автообновления")
    print(f"📌 Коммит: {commit_msg}")

if __name__ == "__main__":
    main()



