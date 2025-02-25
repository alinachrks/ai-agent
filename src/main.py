from data_analysis import load_data, exploratory_data_analysis
from model_training import train_model
from code_updater import update_code
from commit_message import generate_commit_message


if __name__ == "__main__":
    df = load_data()
    eda_report = exploratory_data_analysis(df)
    model, metric_name, metric = train_model(df)

    with open("report.md", "w") as f:
        f.write(eda_report)
        f.write(f"\n\n**Метрика модели ({metric_name}): {metric:.4f}**\n\n")

    update_code()
    commit_msg = generate_commit_message()

    print(f"✅ Обновление завершено. Название коммита: {commit_msg}")
