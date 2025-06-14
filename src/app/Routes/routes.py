from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
import uuid
from datetime import datetime
from typing import List
from services.database_service import get_connection

router = APIRouter()

class UserRegister(BaseModel):
    user_id: int
    password: str
    age: int
    sex: str
    height_cm: float
    current_weight_kg: float
    activity_level: str
    primary_goal: str
    target_weight_kg: float
    weight_change_rate_kg_week: float
    secondary_goals: List[str] = []
    dietary_restrictions: List[str] = []
    allergies: List[str] = []
    intolerances: List[str] = []
    medical_conditions: List[str] = []

    @validator('age')
    def age_range(cls, v):
        if v < 15 or v > 100:
            raise ValueError('La edad debe estar entre 15 y 100 años')
        return v

    @validator('sex')
    def sex_valid(cls, v):
        if v not in ['male', 'female']:
            raise ValueError('El sexo debe ser "male" o "female"')
        return v

    @validator('activity_level')
    def activity_valid(cls, v):
        valid = ['sedentary', 'light', 'moderate', 'active', 'very_active']
        if v not in valid:
            raise ValueError(f'El nivel de actividad debe ser uno de: {valid}')
        return v

@router.post("/register")
def register(user: UserRegister):
    # Cálculo de calorías y macros (ejemplo simple)
    if user.sex == 'male':
        tmb = (10 * user.current_weight_kg) + (6.25 * user.height_cm) - (5 * user.age) + 5
    else:
        tmb = (10 * user.current_weight_kg) + (6.25 * user.height_cm) - (5 * user.age) - 161

    activity_factors = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    tdee = tmb * activity_factors[user.activity_level]

    if user.primary_goal == 'weight_loss':
        target_calories = tdee - 500
    elif user.primary_goal == 'muscle_gain':
        target_calories = tdee + 300
    else:
        target_calories = tdee

    protein = int(user.current_weight_kg * 1.6)
    fat = int((target_calories * 0.25) / 9)
    carbs = int((target_calories - (protein * 4) - (fat * 9)) / 4)
    fiber = int((target_calories / 1000) * 14)

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM user_profiles WHERE user_id = %s", (user.user_id,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="El usuario ya existe")

        cur.execute("""
            INSERT INTO user_profiles (
                user_id, password, age, sex, height_cm, current_weight_kg, activity_level,
                primary_goal, target_weight_kg, weight_change_rate_kg_week,
                secondary_goals, dietary_restrictions, allergies, intolerances, medical_conditions,
                target_calories_kcal, target_protein_g, target_carbs_g, target_fat_g, target_fiber_g
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            user.user_id, user.password, user.age, user.sex, user.height_cm, user.current_weight_kg, user.activity_level,
            user.primary_goal, user.target_weight_kg, user.weight_change_rate_kg_week,
            user.secondary_goals, user.dietary_restrictions, user.allergies, user.intolerances, user.medical_conditions,
            int(target_calories), protein, carbs, fat, fiber
        ))
        conn.commit()
        cur.close()
        conn.close()
        return {
            "message": "Usuario registrado correctamente",
            "objetivos": {
                "calorias": int(target_calories),
                "proteinas": protein,
                "carbohidratos": carbs,
                "grasas": fat,
                "fibra": fiber
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor: {str(e)}"
        )

class UserLogin(BaseModel):
    user_id: str
    password: str

@router.post("/login")
def login(user: UserLogin):
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Consulta para obtener los datos del usuario
        cur.execute("""
            SELECT user_id, password, age, sex, height_cm, current_weight_kg, 
                   activity_level, primary_goal, target_weight_kg
            FROM user_profiles 
            WHERE user_id = %s
        """, (user.user_id,))
        
        result = cur.fetchone()
        cur.close()
        conn.close()

        if not result:
            raise HTTPException(
                status_code=401, 
                detail="Usuario no encontrado"
            )

        # Extraer datos de la consulta
        db_user_id, db_password, age, sex, height, weight, activity, goal, target = result

        # Verificar contraseña (NOTA: En producción, usar hash seguro)
        if user.password != db_password:
            raise HTTPException(
                status_code=401, 
                detail="Credenciales incorrectas"
            )

        # Devolver información básica del usuario
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_uuid = str(uuid.uuid4())[:8]
        session_id = f"ses_{db_user_id}_{timestamp}_{random_uuid}"
        return {
            "message": "Login exitoso",
            "session_id": session_id,
            "user": {
                "user_id": db_user_id,
                "age": age,
                "sex": sex,
                "height_cm": height,
                "current_weight_kg": weight,
                "activity_level": activity,
                "primary_goal": goal,
                "target_weight_kg": target
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor: {str(e)}"
        )

# Modelo para mensajes de chat
class ChatMessage(BaseModel):
    session_id: str
    user_id: str
    role: str  # 'human', 'ai', 'system'
    content: str

    @validator('role')
    def role_valid(cls, v):
        valid = ['human', 'ai', 'system']
        if v not in valid:
            raise ValueError(f'El rol debe ser uno de: {valid}')
        return v

@router.post("/chat/message")
def create_message(message: ChatMessage):
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Verificar que el usuario existe
        cur.execute("SELECT 1 FROM user_profiles WHERE user_id = %s", (message.user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        # Insertar el mensaje
        cur.execute("""
            INSERT INTO chat_messages (session_id, user_id, role, content)
            VALUES (%s, %s, %s, %s)
            RETURNING message_id, sent_at
        """, (message.session_id, message.user_id, message.role, message.content))
        message_id, sent_at = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return {
            "message": "Mensaje creado exitosamente",
            "data": {
                "message_id": message_id,
                "session_id": message.session_id,
                "user_id": message.user_id,
                "role": message.role,
                "content": message.content,
                "sent_at": sent_at.isoformat()
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@router.get("/chat/session/{session_id}")
def get_session_messages(session_id: str):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT message_id, user_id, role, content, sent_at
            FROM chat_messages 
            WHERE session_id = %s
            ORDER BY sent_at ASC
        """, (session_id,))
        messages = []
        for row in cur.fetchall():
            message_id, user_id, role, content, sent_at = row
            messages.append({
                "message_id": message_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "sent_at": sent_at.isoformat()
            })
        cur.close()
        conn.close()
        return {
            "session_id": session_id,
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@router.get("/chat/user/{user_id}")
def get_user_messages(user_id: str, limit: int = 50):
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Verificar que el usuario existe
        cur.execute("SELECT 1 FROM user_profiles WHERE user_id = %s", (user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        # Obtener los mensajes más recientes del usuario
        cur.execute("""
            SELECT message_id, session_id, role, content, sent_at
            FROM chat_messages 
            WHERE user_id = %s
            ORDER BY sent_at DESC
            LIMIT %s
        """, (user_id, limit))
        messages = []
        for row in cur.fetchall():
            message_id, session_id, role, content, sent_at = row
            messages.append({
                "message_id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "sent_at": sent_at.isoformat()
            })
        cur.close()
        conn.close()
        return {
            "user_id": user_id,
            "messages": messages
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )