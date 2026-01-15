import pandas as pd


class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads data from CSV with encoding fallback and separator auto-detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                self.df = pd.read_csv(self.file_path, encoding=enc, sep=None, engine='python')
                return True, f"Данные успешно загружены с кодировкой: {enc}"
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return False, f"Ошибка загрузки данных с кодировкой {enc}: {e}"
        return False, "Не удалось загрузить данные ни с одной из поддерживаемых кодировок."
