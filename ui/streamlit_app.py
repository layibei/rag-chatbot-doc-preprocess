import streamlit as st
import requests
import os
from datetime import datetime

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {
    "user-id": "test-user"  # For testing purposes
}

def upload_document():
    st.header("Upload Document")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'csv', 'json', 'docx'])
    
    # Source type selector
    source_type = st.selectbox(
        "Select document type",
        ['pdf', 'txt', 'csv', 'json', 'docx']
    )
    
    if uploaded_file and st.button("Upload"):
        files = {"file": uploaded_file}
        try:
            response = requests.post(
                f"{API_BASE_URL}/embedding/docs/upload",
                headers=HEADERS,
                files=files,
                params={"source_type": source_type}
            )
            
            if response.status_code == 200:
                st.success("Document uploaded successfully!")
                st.json(response.json())
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error uploading document: {str(e)}")

def view_documents():
    st.header("View/list Documents")
    
    # Search box
    search_query = st.text_input("Search/list documents", "")
    
    # Pagination controls
    col1, col2 = st.columns(2)
    with col1:
        page = st.number_input("Page", min_value=1, value=1)
    with col2:
        page_size = st.number_input("Page Size", min_value=1, value=10)
    
    if st.button("Load documents"):
        try:
            params = {
                "page": page,
                "page_size": page_size
            }
            if search_query:
                params["search"] = search_query
                
            response = requests.get(
                f"{API_BASE_URL}/embedding/docs",
                headers=HEADERS,
                params=params
            )
            
            if response.status_code == 200:
                documents = response.json()
                if not documents:
                    st.info("No documents found")
                else:
                    for doc in documents:
                        with st.expander(f"{doc['source']} ({doc['source_type']})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Status:", doc['status'])
                                st.write("Created by:", doc['created_by'])
                                st.write("Created at:", datetime.fromisoformat(doc['created_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S'))
                            with col2:
                                st.write("ID:", doc['id'])
                                st.write("Modified by:", doc['modified_by'])
                                st.write("Modified at:", datetime.fromisoformat(doc['modified_at'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S'))
                            if doc['error_message']:
                                st.error(f"Error: {doc['error_message']}")
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")

def check_document_status():
    st.header("Get document by id")
    
    doc_id = st.number_input("Document ID", min_value=1)
    
    if st.button("Load"):
        try:
            response = requests.get(
                f"{API_BASE_URL}/embedding/docs/{doc_id}",
                headers=HEADERS
            )
            
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error checking status: {str(e)}")

def main():
    st.title("Document Management System")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Upload", "Search/list Documents", "Get document by id"])
    
    with tab1:
        upload_document()
    
    with tab2:
        view_documents()
    
    with tab3:
        check_document_status()

if __name__ == "__main__":
    main() 