import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("DATABASE_URL")
print("Intentando conectar a:", url)
conn = psycopg2.connect(url)
print("¡Conexión exitosa!")
conn.close()