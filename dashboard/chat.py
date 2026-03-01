"""Chat page for Streamlit dashboard.

Adds a conversational AI tab where users can query energy data
using natural language via the local LLM agent.

This file is imported and rendered as a tab in dashboard/app.py.
"""

import streamlit as st
from loguru import logger


def render_chat_tab():
    """Render the chat interface tab."""
    st.header("💬 Energy AI Assistant")
    st.caption(
        "Ask questions about your energy data. "
        "Powered by Llama 3.1 running locally via Ollama."
    )

    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = None
        st.session_state.chat_messages = []
        st.session_state.agent_error = None

    # Connect to agent
    if st.session_state.agent is None:
        try:
            from src.agent import Agent

            st.session_state.agent = Agent()
            st.session_state.agent_error = None
        except ConnectionError:
            st.session_state.agent_error = (
                "Cannot connect to Ollama. Make sure it's running: "
                "`ollama serve`"
            )
        except Exception as e:
            st.session_state.agent_error = f"Agent initialization failed: {e}"

    if st.session_state.agent_error:
        st.error(st.session_state.agent_error)
        if st.button("Retry Connection"):
            st.session_state.agent = None
            st.session_state.agent_error = None
            st.rerun()
        return

    # Suggested prompts
    if not st.session_state.chat_messages:
        st.markdown("**Try asking:**")
        suggestions = [
            "What date range of data do you have?",
            "What was the average power usage in January 2007?",
            "Find anomalies in the first week of 2007",
            "Tell me about the trained model",
            "What was peak consumption on 2007-01-15?",
        ]

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            if col.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state.chat_messages.append(
                    {"role": "user", "content": suggestion}
                )
                st.rerun()

    # Display conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your energy data..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(response)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Sidebar controls for chat
    with st.sidebar:
        st.markdown("---")
        st.subheader("Chat Controls")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            if st.session_state.agent:
                st.session_state.agent.reset()
            st.rerun()

        if st.button("Show Tool Log", use_container_width=True):
            if st.session_state.agent:
                log = st.session_state.agent.get_tool_log()
                if log:
                    st.sidebar.json(log[-5:])  # Show last 5 calls
                else:
                    st.sidebar.info("No tool calls yet")
