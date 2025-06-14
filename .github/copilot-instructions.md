# Personalized Nutritional Assistance AI Agent - Project Instructions

## 1. Introduction

This project aims to develop an AI-powered agent for personalized nutritional assistance. Users will interact with a Progressive Web Application (PWA) to receive dietary recommendations based on visual analysis of their meals and their pre-defined nutritional profiles. The backend will leverage Google Gemini models orchestrated by LangGraph.

## 2. Core Technologies

*   **Programming Language:** Python 3.x
*   **AI Models:**
    *   **Gemini Gemini 2.5 Flash:** For image processing (food detection and segmentation).
    *   **Gemini 2.5 Pro:** For nutritional reasoning and generating personalized recommendations.
    *   **Gemini 2.0 flash lite:** For user interaction and Q&A.
*   **Orchestration:** LangGraph
*   **Data Validation & State Management:** Pydantic
*   **Frontend:** Progressive Web Application (PWA) - details to be defined.
*   **Database:** PostgreSQL - For user profiles, meal logs, and chat history.

## 3. Project Setup

1.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install Dependencies:**
    Create a `requirements.txt` file:
    ```txt
    langchain
    langchain-google-genai
    langgraph
    pydantic
    google-generativeai
    # Add other dependencies like Flask/FastAPI for PWA backend, DB connectors, etc.
    ```
    Install them:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Google Gemini API Key:**
    *   Obtain an API key from Google AI Studio.
    *   Set it as an environment variable:
        ```bash
        export GOOGLE_API_KEY="YOUR_API_KEY" # On Windows: set GOOGLE_API_KEY="YOUR_API_KEY"
        ```

## 3.1. Database Schema (PostgreSQL)

In the PostgreSQL database.

`CREATE TYPE sex_enum AS ENUM ('male', 'female');`
`CREATE TYPE activity_level_enum AS ENUM ('sedentary', 'light', 'moderate', 'active', 'very_active')`
`CREATE TYPE message_sender_enum AS ENUM ('human', 'ai', 'system');`


```sql
CREATE TABLE user_profiles (
    user_id VARCHAR(255) PRIMARY KEY, -- Could be a UUID or your authentication system's ID

    -- Basic Personal Information
    age INTEGER,
    sex sex_enum,
    height_cm NUMERIC(5,2), -- Ex: 175.50 cm
    current_weight_kg NUMERIC(5,2), -- Ex: 70.50 kg
    activity_level activity_level_enum,

    -- Health and Nutrition Goals
    primary_goal TEXT, -- Ex: "weight_loss", "muscle_gain"
    target_weight_kg NUMERIC(5,2) NULL,
    weight_change_rate_kg_week NUMERIC(4,2) NULL, -- Ex: -0.50 kg/week
    secondary_goals TEXT[] NULL, -- Array of texts, ex: ARRAY['increase fiber intake', 'reduce sugar']

    -- Dietary Restrictions and Preferences
    dietary_restrictions TEXT[] NULL, -- Ex: ARRAY['veganism', 'gluten-free']
    allergies TEXT[] NULL, -- Ex: ARRAY['peanuts', 'shellfish']
    intolerances TEXT[] NULL, -- Ex: ARRAY['lactose']

    -- Relevant Medical Conditions (handle with care due to privacy)
    medical_conditions TEXT[] NULL, -- Ex: ARRAY['type 2 diabetes', 'hypertension']

    -- Nutritional Targets
    target_calories_kcal INTEGER NULL,
    target_protein_g INTEGER NULL,
    target_carbs_g INTEGER NULL,
    target_fat_g INTEGER NULL,
    target_fiber_g INTEGER NULL
);

-- meal_logs table removed as per simplification

CREATE TABLE chat_messages (
    message_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL, -- Unique identifier for a conversation session
    user_id VARCHAR(255) NOT NULL REFERENCES user_profiles(user_id) ON DELETE CASCADE,
    sender_type message_sender_enum NOT NULL, -- 'human', 'ai', or 'system'
    content TEXT NOT NULL,
    -- Add timestamp for message ordering
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## 4. LangGraph Agent Workflow & State

The system will be architected as a multi-agent graph using LangGraph.

### 4.1. Pydantic State Definition:
```python
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class DetectedFoodItem(BaseModel):
    label: str = Field(description="Name of the detected food item.")
    bounding_box: List[float] = Field(description="Coordinates of the bounding box [x_min, y_min, x_max, y_max] relative to image size (0.0-1.0) or pixels.")
    mask_data: Optional[Any] = Field(None, description="Segmentation mask data, e.g., RLE, polygon, or raw mask.")

class UserProfile(BaseModel):
    user_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    height_cm: Optional[float] = None
    current_weight_kg: Optional[float] = None
    activity_level: Optional[str] = None
    primary_goal: Optional[str] = None
    target_weight_kg: Optional[float] = None
    weight_change_rate_kg_week: Optional[float] = None
    secondary_goals: Optional[List[str]] = Field(default_factory=list)
    dietary_restrictions: Optional[List[str]] = Field(default_factory=list)
    allergies: Optional[List[str]] = Field(default_factory=list)
    intolerances: Optional[List[str]] = Field(default_factory=list)
    disliked_foods: Optional[List[str]] = Field(default_factory=list)
    medical_conditions: Optional[List[str]] = Field(default_factory=list)
    target_calories_kcal: Optional[int] = None
    target_protein_g: Optional[int] = None
    target_carbs_g: Optional[int] = None
    target_fat_g: Optional[int] = None
    target_fiber_g: Optional[int] = None

class AgentState(BaseModel):
    user_id: str = Field(description="ID del usuario autenticado.")
    session_id: str = Field(description="ID único para la sesión de chat/análisis de esta comida específica.")
    input_image_bytes: Optional[bytes] = Field(None, description="Bytes de la imagen de la comida subida por el usuario.")
    meal_context_description: Optional[str] = Field(None, description="Descripción textual opcional de la comida proporcionada por el usuario (ej. \'\'\'mi desayuno\'\'\').") # NUEVO
    user_profile: Optional[UserProfile] = Field(None, description="Perfil nutricional del usuario obtenido de la BD.")
    detected_foods: Optional[List[DetectedFoodItem]] = Field(default_factory=list, description="Alimentos detectados en la imagen.")
    raw_gemini_image_analysis_response: Optional[Any] = Field(None, description="Respuesta cruda del análisis de imagen de Gemini (para depuración o datos más ricos).")
    nutritional_recommendation: Optional[str] = Field(None, description="Recomendación nutricional generada por el agente de razonamiento.")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description=\'\'\'Historial del chat para esta sesión. Ej: [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]\'\'\')
    formatted_final_response: Optional[str] = Field(None, description="La respuesta formateada final que se envía al usuario (podría ser la recomendación inicial o una respuesta de Q&A).")
    error_message: Optional[str] = Field(None, description="Mensaje de error si algo falla en el proceso.")
    # Flags de progreso
    profile_retrieved: bool = Field(False)
    image_analyzed: bool = Field(False)
    recommendation_generated: bool = Field(False)
```

### 4.2. Agent Nodes (Conceptual):

**Inicio del Grafo / Configuración del Estado Inicial (Pre-Nodos):**
*   **Input (desde el backend de la PWA al invocar el grafo):** `user_id` (del usuario autenticado), `image_bytes` (imagen de la comida), `meal_context_description` (texto opcional del usuario, ej., "mi desayuno").
*   **Acción:** Se genera un `session_id` único para esta interacción. El `AgentState` se inicializa con `user_id`, `session_id`, `input_image_bytes`, y `meal_context_description`.

**`UserProfileRetrievalNode`:**
*   **Input:** `user_id` del `AgentState`.
*   **Acción:** Consulta la base de datos (tabla `user_profiles`) para obtener el perfil del usuario. Popula `AgentState.user_profile`.
*   **Output:** Actualiza `user_profile`, `profile_retrieved` en el estado.

**`ImageAnalysisNode` (Utilizando Gemini 2.5 Flash):**

    Input: `input_image_bytes`, `meal_context_description` (opcional, para posible futuro uso o logging) del `AgentState`.

    Action:
        1.  **Preparación de la Imagen:**
            *   Convierte los `input_image_bytes` (que están en formato `bytes`) a un objeto de imagen compatible con la librería de Google GenAI, típicamente un objeto `PIL.Image`. Esto se puede hacer usando `io.BytesIO` y `PIL.Image.open()`:
                ```python
                # Ejemplo conceptual dentro del nodo
                import io
                from PIL import Image
                image_object = Image.open(io.BytesIO(state.input_image_bytes))
                ```

        2.  **Instanciación del Modelo Gemini:**
            *   Se instancia el modelo Gemini optimizado para visión y velocidad, como "Gemini 1.5 Flash" (o el nombre exacto del modelo que decidas, ej. `gemini-1.5-flash-latest`).
                ```python
                # Ejemplo conceptual
                import google.generativeai as genai
                # (Asumiendo que GOOGLE_API_KEY está configurado)
                model = genai.GenerativeModel('gemini-1.5-flash-latest') # o el modelo específico
                ```

        3.  **Construcción del Prompt Multimodal:**
            *   Se crea un prompt que es una lista conteniendo tanto instrucciones textuales como el objeto de imagen. El texto del prompt debe solicitar explícitamente la identificación de alimentos, sus bounding boxes, y sus máscaras de segmentación, pidiendo la salida en formato JSON.
            *   **Referencia Clave del Colab (Sección "Experimental: Segmentation"):** El Colab muestra cómo obtener `label`, `box_2d` (bounding box), y `mask` (como data URI de imagen PNG base64).
            *   **Prompt Ejemplo para el Nodo:**
                ```python
                prompt_parts = [
                    \"\"\"Por favor, analiza la siguiente imagen de un plato de comida.
                    Identifica cada alimento distinto presente.
                    Para cada alimento identificado, proporciona:
                    1. Una etiqueta (label) descriptiva para el alimento (ej., "plátano", "filete de pollo a la parrilla", "arroz blanco cocido").
                    2. Las coordenadas del bounding box (box_2d) como una lista de 4 números [y_min, x_min, y_max, x_max] en píxeles.
                    3. Una máscara de segmentación (mask) para el contorno preciso del alimento, como una cadena de datos URI de imagen PNG codificada en base64.

                    Devuelve esta información como una lista JSON, donde cada elemento del JSON sea un objeto representando un alimento con las claves "label", "box_2d", y "mask".
                    Asegúrate de que la salida sea únicamente la lista JSON válida.
                    \"\"\",
                    image_object # El objeto PIL.Image preparado en el paso 1
                ]
                ```

        4.  **Llamada al Modelo Gemini y Obtención de Respuesta:**
            *   Se envía el `prompt_parts` al modelo Gemini.
                ```python
                # Ejemplo conceptual
                response = model.generate_content(prompt_parts)
                # La respuesta cruda (incluyendo el JSON) usualmente está en response.text
                raw_response_text = response.text
                ```
            *   Se guarda esta respuesta cruda (el texto completo) en `AgentState.raw_gemini_image_analysis_response` para depuración o análisis posterior.

        5.  **Parseo de la Respuesta JSON:**
            *   El texto de la respuesta de Gemini a menudo contiene el JSON envuelto en triple comillas invertidas (Markdown para un bloque de código JSON). Se debe limpiar este formato para extraer el JSON puro.
                ```python
                # Ejemplo conceptual de limpieza y parseo
                import json
                cleaned_json_str = raw_response_text.strip().removeprefix("```json").removesuffix("```").strip()
                try:
                    parsed_food_data_list = json.loads(cleaned_json_str) # Debería ser una lista de dicts
                except json.JSONDecodeError as e:
                    # Manejar el error, quizás poblar AgentState.error_message
                    # y detener el flujo o marcar image_analyzed como False.
                    print(f"Error al decodificar JSON: {e}")
                    parsed_food_data_list = []
                ```

        6.  **Transformación a Objetos `DetectedFoodItem`:**
            *   Se itera sobre la `parsed_food_data_list` (que es una lista de diccionarios, donde cada diccionario representa un alimento).
            *   Para cada diccionario de alimento, se crea una instancia del Pydantic model `DetectedFoodItem`, mapeando las claves del JSON (`label`, `box_2d`, `mask`) a los campos correspondientes del modelo (`label`, `bounding_box`, `mask_data`).
                *   `item_json['label']` -> `DetectedFoodItem.label`
                *   `item_json['box_2d']` -> `DetectedFoodItem.bounding_box` (asegurar el orden correcto de coordenadas si es necesario, ej. [x_min, y_min, x_max, y_max] vs. [y_min, x_min, y_max, x_max])
                *   `item_json['mask']` -> `DetectedFoodItem.mask_data`
            *   Se añaden estos objetos `DetectedFoodItem` a la lista `AgentState.detected_foods`.

    Output: Actualiza `detected_foods` (con la lista de objetos `DetectedFoodItem`), `raw_gemini_image_analysis_response` (con el texto crudo de Gemini), y `image_analyzed` (a `True` si todo fue exitoso, `False` en caso de error) en el `AgentState`.

**`NutritionalReasoningNode` (Utilizando Gemini 2.5 Pro):**
*   **Input:** `detected_foods`, `user_profile`, `meal_context_description` del `AgentState`.
*   **Acción:**
    1.  Construye un prompt detallado para Gemini Pro. Este prompt debe incluir:
        *   La lista de `detected_foods`.
        *   El `user_profile` completo (objetivos, restricciones, etc.).
        *   El `meal_context_description` (ej., "El usuario indica que esta comida es su 'almuerzo'. Basado en su perfil y los siguientes alimentos identificados, provee un análisis y recomendaciones...").
    2.  Consulta a Gemini Pro para generar el `nutritional_recommendation`.
*   **Output:** Actualiza `nutritional_recommendation`, `recommendation_generated` en el estado.

**`PresentationAndQANode` (Utilizando Gemini para chat):**
*   **Input:** `nutritional_recommendation`, `chat_history`, `user_profile`, `detected_foods`, `meal_context_description` (para contexto en Q&A) del `AgentState`.
*   **Acción:**
    1.  **Para la respuesta inicial:** Formatea el `nutritional_recommendation` de manera amigable. Esta será la primera respuesta de la IA en la interfaz de chat. Se añade al `chat_history` (ej., `{"role": "model", "content": "Análisis de tu comida: ..."}`).
    2.  **Para Q&A subsiguiente:** Cuando el usuario envía un nuevo mensaje (pregunta/comentario), este se añade al `chat_history`. Se envía el `chat_history` actualizado, junto con el contexto relevante (perfil, comida analizada, recomendación inicial), al modelo de chat de Gemini para generar una respuesta. La respuesta del modelo también se añade al `chat_history`.
*   **Output:** Actualiza `formatted_final_response` (con la última respuesta de la IA) y `chat_history` en el estado.

**`DataPersistenceNode`:**
*   **Input:** `AgentState` (específicamente `user_id`, `session_id`, `chat_history`).
*   **Acción:** Guarda los mensajes nuevos en `chat_history` (tanto del usuario como de la IA) en la tabla `chat_messages` de la base de datos, asegurándose de que estén asociados con el `user_id` y `session_id` correctos.
*   **Output:** Principalmente efectos secundarios (escritura en BD).

### 4.3. LangGraph Edges:

*   **Define el flujo:**
    1.  **Invocación del Grafo y Configuración del Estado Inicial:** (Como se describió arriba, el backend de la PWA prepara los datos iniciales y llama al grafo LangGraph).
    2.  **Ejecución Paralela (o secuencial según se determine óptimo):**
        *   `UserProfileRetrievalNode`
        *   `ImageAnalysisNode`
        *   (Se necesita un punto de unión o condición para asegurar que ambos nodos completen antes de pasar al siguiente paso).
    3.  **`NutritionalReasoningNode`:** Se ejecuta después de que `UserProfileRetrievalNode` y `ImageAnalysisNode` hayan finalizado y sus resultados estén en el estado.
    4.  **`PresentationAndQANode`:** Se ejecuta después de `NutritionalReasoningNode`. Este nodo es responsable de la respuesta inicial de la IA y de manejar el ciclo de Q&A posterior.
    5.  **`DataPersistenceNode`:** Se ejecuta después de cada interacción dentro de `PresentationAndQANode` que modifique `chat_history` (es decir, después de que el usuario envía un mensaje y después de que la IA genera una respuesta) para mantener la base de datos actualizada.
*   Considerar condiciones de error o rutas alternativas si algún dato no está disponible o si ocurre un fallo en algún nodo.

## 5. General Guidance & Best Practices

*   **Modularity:** Keep each LangGraph node focused on a single responsibility.
*   **Error Handling:** Implement robust error handling within each node and potentially define error paths in the graph.
*   **Prompt Engineering:** Carefully craft the prompts for each Gemini model to elicit the most accurate and relevant responses.
*   **Configuration:** Manage API keys and other configurations securely (e.g., via environment variables).
*   **Simplicity:** Start with a core functional flow and iterate
*   **Documentation:** Maintain clear documentation for each node's purpose, inputs, and outputs to facilitate future development and debugging.
*   **Code:** Code in english