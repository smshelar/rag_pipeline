import streamlit as st
from streamlit_chat import message
from query import query_rag
from load_model import *
from time import sleep

# Configure Streamlit page layout and title
st.set_page_config(layout='centered', page_title=f'PDF Chatbot')
st.title("PDF Chatbot")


def main():
    st.sidebar.subheader("Load PDF")

    pdf_file_uploaded = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    st.markdown("<br>" * 5, unsafe_allow_html=True)
    st.sidebar.text("OR")
    st.markdown("<br>" * 5, unsafe_allow_html=True)

    try:
        file_chosen = st.sidebar.selectbox("Select an existing document:", get_list_of_documents())
    except NameError:
        file_chosen = None  # Ensure the code doesn't break if function is missing

    pdf_files = pdf_file_uploaded if pdf_file_uploaded else file_chosen

    # Determine button color based on CHROMA_PATH
    button_color = "black" if not os.path.exists(CHROMA_PATH) else "red"

    # Custom button styling for the chatbot UI
    st.markdown(
        f"""
        <style>
        .stButton>button {{
            background-color: {button_color} !important;
            color: white !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border-radius: 10px !important;
            padding: 10px !important;
            position: fixed;
            bottom: 35px;
            width: 275px;
        }}

         /* Custom avatar size */
        .streamlit-message .chat-avatar img {{
            width: 10px;  /* Set desired avatar width */
            height: 10px; /* Set desired avatar height */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar additional UI elements
    with st.sidebar:
        # Create a placeholder for spacing
        st.markdown("<br>" * 15, unsafe_allow_html=True)

        # Add a separator
        st.sidebar.markdown('<hr style="border:0.1px solid #D3D3D3;">', unsafe_allow_html=True)

        # Button in the sidebar
        if st.button("🗑️ Clear Database"):
            if button_color == "red":
                clear_database()
                button_color = "black"
                st.write("Cleared!!")
                sleep(0.5)
                st.rerun()
            else:
                st.write(f"Path: ./{CHROMA_PATH} doesn't exists.")

    if 'processed' not in st.session_state:
        st.session_state.processed = False  # Flag to track if preprocessing is done
    # Process uploaded or selected PDFs
    if pdf_files and not st.session_state.processed:
        st.toast("📄 Files Uploaded Successfully!", icon="✅")

        with st.spinner("Calculating Vectors and Updating Database... Please wait ⏳"):
            database_pipeline(pdf_files)

        st.toast("📄 Database updated Successfully!", icon="✅")

        st.session_state.processed = True  # Mark preprocessing as done

    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = []

    if 'rag_response' not in st.session_state:
        st.session_state['rag_response'] = []

    def get_text():
        input_text = st.chat_input("Say something")
        return input_text

    user_input = get_text()

    if user_input:
        output = query_rag(user_input)
        output = output.lstrip("\n")

        # Store the output
        st.session_state.user_input.append(output)
        st.session_state.rag_response.append(user_input)

    if st.session_state['user_input']:
        for i in range(len(st.session_state['user_input'])):
            # This function displays rag response
            message(st.session_state['rag_response'][i],
                    avatar_style="miniavs", is_user=True, key=str(i) + 'data_by_user')

            # This function displays user input
            message(st.session_state["user_input"][i], key=str(i), avatar_style="icons")


main()
