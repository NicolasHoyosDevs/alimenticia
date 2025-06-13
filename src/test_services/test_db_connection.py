# src/test_db_connection.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.database_service import get_db, UserProfile, init_db
from datetime import datetime

def test_database_connection():
    db = None
    try:
        # Inicializar la base de datos
        init_db()
        print("✅ Base de datos inicializada correctamente")

        # Obtener una sesión de base de datos
        db = next(get_db())
        
        # Crear un usuario de prueba
        test_user = UserProfile(
            user_id=2,
            password="123456",
            age=30,
            sex="male",
            height_cm=175.0,
            current_weight_kg=70.0,
            activity_level="moderate",
            primary_goal="weight_loss",
            target_weight_kg=65.0,
            weight_change_rate_kg_week=-0.5,
            secondary_goals=["increase protein intake", "reduce sugar"],
            dietary_restrictions=["vegetarian"],
            allergies=["peanuts"],
            intolerances=["lactose"],
            medical_conditions=["none"],
            target_calories_kcal=2000,
            target_protein_g=150,
            target_carbs_g=200,
            target_fat_g=65,
            target_fiber_g=30
        )

        # Intentar guardar el usuario
        db.add(test_user)
        db.commit()
        print("✅ Usuario de prueba creado correctamente")

        # Intentar recuperar el usuario
        retrieved_user = db.query(UserProfile).filter_by(user_id="2").first()
        if retrieved_user:
            print("✅ Usuario recuperado correctamente:")
            print(f"  - ID: {retrieved_user.user_id}")
            print(f"  - Edad: {retrieved_user.age}")
            print(f"  - Altura: {retrieved_user.height_cm} cm")
            print(f"  - Peso actual: {retrieved_user.current_weight_kg} kg")
            print(f"  - Nivel de actividad: {retrieved_user.activity_level}")
            print(f"  - Objetivo principal: {retrieved_user.primary_goal}")

        # Limpiar el usuario de prueba
        db.delete(retrieved_user)
        db.commit()
        print("✅ Usuario de prueba eliminado correctamente")

    except Exception as e:
        print(f"❌ Error durante la prueba: {str(e)}")
    finally:
        if db is not None:
            db.close()

if __name__ == "__main__":
    print("🧪 Iniciando prueba de conexión a la base de datos...")
    test_database_connection()