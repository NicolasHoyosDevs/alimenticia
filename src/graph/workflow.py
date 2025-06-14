from langgraph.graph import StateGraph, END
from ..schemas.schema import AgentState  # Assuming state.py is in src
from ..agents.agents import (  # Assuming agents.py is in src
    welcome_and_capture_node,
    user_profile_retrieval_node,
    image_analysis_node,
    nutritional_reasoning_node,
    presentation_and_qa_node,
    data_persistence_node
)

# --- LangGraph Workflow Definition ---

# Initialize the StateGraph with the AgentState Pydantic model
workflow = StateGraph(AgentState)

# Add nodes to the graph
# Each node corresponds to a function defined in agents.py
workflow.add_node("welcome_and_capture", welcome_and_capture_node)
workflow.add_node("user_profile_retrieval", user_profile_retrieval_node)
workflow.add_node("image_analysis", image_analysis_node)
workflow.add_node("nutritional_reasoning", nutritional_reasoning_node)
workflow.add_node("presentation_and_qa", presentation_and_qa_node)
workflow.add_node("data_persistence", data_persistence_node)

# Define the entry point of the graph
workflow.set_entry_point("welcome_and_capture")

# Define edges that determine the flow of execution
# After welcome_and_capture, user_profile_retrieval and image_analysis can run (potentially in parallel if supported and desired)
workflow.add_edge("welcome_and_capture", "user_profile_retrieval")
workflow.add_edge("welcome_and_capture", "image_analysis")

# Nutritional reasoning depends on both user profile and image analysis being complete.
# LangGraph handles this by ensuring that the state fields these nodes populate
# are available before 'nutritional_reasoning' is called, assuming it accesses them.
# For a more explicit join/synchronization, a conditional edge or a dedicated join node
# might be used in more complex scenarios, but for this linear dependency, direct edging works.
workflow.add_edge("user_profile_retrieval", "nutritional_reasoning")
workflow.add_edge("image_analysis", "nutritional_reasoning")

workflow.add_edge("nutritional_reasoning", "presentation_and_qa")
workflow.add_edge("presentation_and_qa", "data_persistence")

# Define the end point of the graph
workflow.add_edge("data_persistence", END)

# Compile the graph into a runnable application
app = workflow.compile()

# # --- Example of how to run the graph (for testing) ---
# # This would typically be part of your main application logic,
# # perhaps triggered by an API endpoint.

# if __name__ == "__main__":
#     print("Compiling and running LangGraph agent...")

#     # Simulate initial input that would come from the PWA/API
#     # For image_bytes, you would load an actual image file as bytes.
#     # Example:
#     with open(r"C:\\Users\\jhonh\\Downloads\\Imagen de WhatsApp 2025-06-11 a las 17.12.23_b3a72ca1.jpg", "rb") as f:
#         sample_image_bytes = f.read()
#     # sample_image_bytes = b"simulated_image_data_bytes"  # Placeholder

#     initial_state = {
#         "user_id": "test_user_123",
#         "input_image_bytes": sample_image_bytes,
#         "chat_history": []  # Start with an empty chat history
#     }
#     print(
#         f"Invoking graph with initial state: {{k: v if k != 'input_image_bytes' else 'image_bytes_present' for k, v in initial_state.items()}}")

#     # Invoke the graph with the initial state
#     # The `stream()` method can be used for observing state changes at each step.
#     # For a final result, `invoke()` is simpler.
#     final_state = None
#     try:
#         # Using stream to see intermediate states for debugging/logging
#         # step_output is a dict like {'node_name': AgentState_after_node}
#         for step_output in app.stream(initial_state):
#             # Get the name of the node that just ran
#             node_name = next(iter(step_output))
#             # Get the AgentState object
#             current_state_after_node = step_output[node_name]

#             print(f"--- Current State after node '{node_name}' ---")

#             if current_state_after_node is not None:
#                 state_dict = None
#                 if isinstance(current_state_after_node, AgentState):
#                     # If it's an AgentState Pydantic model instance
#                     state_dict = current_state_after_node.model_dump()
#                 elif isinstance(current_state_after_node, dict):
#                     # If it's already a dictionary
#                     state_dict = current_state_after_node
#                 else:
#                     print(
#                         f"Warning: State after node '{node_name}' is of unexpected type: {type(current_state_after_node)}")
#                     state_dict = {}  # Use empty dict to avoid further errors in printing

#                 printable_state = {
#                     k: v if k != 'input_image_bytes' else f'{len(v) if isinstance(v, bytes) else "0 or not bytes"} bytes'
#                     for k, v in state_dict.items()
#                 }
#                 print(printable_state)
#             else:
#                 # This case indicates a problem with the node's return value.
#                 print(
#                     f"Warning: State after node '{node_name}' is None. This might indicate an issue in the node's implementation (it might be returning None).")

#             print("---------------------------------------")
#             # This will be None if the node returned None
#             final_state = current_state_after_node

#         # Check type before accessing attributes
#         if final_state is not None and isinstance(final_state, AgentState):
#             print("\n---FINAL AGENT OUTPUT---")
#             print(
#                 f"Formatted Response: {final_state.formatted_final_response}")
#             if final_state.error_message:
#                 print(f"Error: {final_state.error_message}")
#         elif final_state is not None:
#             print("\n---FINAL AGENT OUTPUT (Unexpected format)---")
#             print(f"Final state: {final_state}")
#         else:
#             print("Graph execution did not produce a final state (it was None). This likely means the last node or an intermediate node returned None.")

#     except Exception as e:
#         print(f"An error occurred during graph execution: {e}")
#         import traceback
#         traceback.print_exc()

#     print("\nGraph execution finished.")
