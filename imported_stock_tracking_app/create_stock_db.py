import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'stocks.db'
SCHEMA_PATH = BASE_DIR / 'schema.sql'


def load_schema():
    with SCHEMA_PATH.open('r', encoding='utf-8') as schema_file:
        return schema_file.read()


def initialize_database():
    schema_sql = load_schema()
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema_sql)
        conn.row_factory = sqlite3.Row
    conn.commit() 
    conn.close()

    

if __name__ == '__main__':
    initialize_database()