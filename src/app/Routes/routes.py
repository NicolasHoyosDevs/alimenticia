from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
import uuid
from datetime import datetime
from typing import List
from src.services.database_service import get_connection
from src.schemas.schema import AgentState
import base64
from src.graph.workflow import app  # 'app' es el grafo compilado y listo para usar

router = APIRouter()

class UserRegister(BaseModel):
    user_id: str
    password: str

@router.post("/register")
def register(user: UserRegister):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM user_profiles WHERE user_id = %s", (user.user_id,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="El usuario ya existe")
        cur.execute("""
            INSERT INTO user_profiles (user_id, password)
            VALUES (%s, %s)
        """, (user.user_id, user.password))
        conn.commit()
        cur.close()
        conn.close()
        return {"message": "Usuario registrado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# Nuevo modelo y endpoint para completar/actualizar el perfil
class UserProfileUpdate(BaseModel):
    user_id: str
    age: int = None
    sex: str = None
    height_cm: float = None
    current_weight_kg: float = None
    activity_level: str = None
    primary_goal: str = None
    target_weight_kg: float = None
    weight_change_rate_kg_week: float = None
    secondary_goals: List[str] = []
    dietary_restrictions: List[str] = []
    allergies: List[str] = []
    intolerances: List[str] = []
    medical_conditions: List[str] = []

@router.put("/profile/update")
def update_profile(profile: UserProfileUpdate):
    try:
        conn = get_connection()
        cur = conn.cursor()
        # Solo actualiza los campos que no sean None
        update_fields = []
        values = []
        for field, value in profile.dict().items():
            if field != "user_id" and value is not None:
                update_fields.append(f"{field} = %s")
                values.append(value)
        # Verificar si ya se tienen todos los datos necesarios para el cálculo
        required_fields = [profile.age, profile.sex, profile.height_cm, profile.current_weight_kg, profile.activity_level, profile.primary_goal]
        if all(x is not None for x in required_fields):
            # Cálculo de calorías y macros
            if profile.sex == 'male':
                tmb = (10 * profile.current_weight_kg) + (6.25 * profile.height_cm) - (5 * profile.age) + 5
            else:
                tmb = (10 * profile.current_weight_kg) + (6.25 * profile.height_cm) - (5 * profile.age) - 161
            activity_factors = {
                'sedentary': 1.2,
                'light': 1.375,
                'moderate': 1.55,
                'active': 1.725,
                'very_active': 1.9
            }
            tdee = tmb * activity_factors.get(profile.activity_level, 1.2)
            if profile.primary_goal == 'weight_loss':
                target_calories = tdee - 500
            elif profile.primary_goal == 'muscle_gain':
                target_calories = tdee + 300
            else:
                target_calories = tdee
            protein = int(profile.current_weight_kg * 1.6)
            fat = int((target_calories * 0.25) / 9)
            carbs = int((target_calories - (protein * 4) - (fat * 9)) / 4)
            fiber = int((target_calories / 1000) * 14)
            # Añadir estos campos a la actualización
            update_fields += [
                "target_calories_kcal = %s",
                "target_protein_g = %s",
                "target_carbs_g = %s",
                "target_fat_g = %s",
                "target_fiber_g = %s"
            ]
            values += [int(target_calories), protein, carbs, fat, fiber]
        if not update_fields:
            raise HTTPException(status_code=400, detail="No hay campos para actualizar")
        values.append(profile.user_id)
        query = f"UPDATE user_profiles SET {', '.join(update_fields)} WHERE user_id = %s"
        cur.execute(query, tuple(values))
        conn.commit()
        cur.close()
        conn.close()
        return {"message": "Perfil actualizado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

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
    
class activate_workflow(BaseModel):
    user_id: str
    session_id: str
    input_image_bytes: str  # Imagen en base64
    meal_context_description: str = None

@router.post("/analizar_comida")
async def analizar_comida(request: activate_workflow):
    try:
        # Decodifica la imagen de base64 a bytes
        image_bytes = base64.b64decode(request.input_image_bytes)
        
        # Inicializa el AgentState
        state = AgentState(
            user_id=request.user_id,
            session_id=request.session_id,
            input_image_bytes=image_bytes,
            meal_context_description=request.meal_context_description
        )
        
        # Lanza el workflow de LangGraph (esto depende de tu integración)
        # Por ejemplo:
        # result = langgraph.run(state)
        # Aquí solo simulo la respuesta:

        result = app.invoke(state.model_dump())
        # Eliminar input_image_bytes y cualquier campo tipo bytes de la respuesta
        if isinstance(result, dict):
            result.pop("input_image_bytes", None)
            # Si hay otros campos tipo bytes, también puedes filtrarlos aquí
            # Solo retornar la respuesta final formateada
            return {"formatted_final_response": result.get("formatted_final_response")}
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")