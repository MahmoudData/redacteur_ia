import streamlit as st
import tempfile
import os
from docx import Document
from io import BytesIO
import uuid
from pipeline import (
    parser_sommaire, generate_prompts_from_structure, load_documents_from_paths, group_docs_by_section,
    split_text, chunk_sections_with_context, embed_text, create_collection, upsert_embeddings,
    search_in_qdrant, scaleway_llm, generate_sections_from_prompts,
    delete_collection, generer_docx
)

st.set_page_config(page_title="Générateur documents", page_icon="📄")

# Vérifie si l'utilisateur est connecté
if not st.session_state.get("authentication_status"):
    st.warning("Veuillez vous connecter dans la page *Mon Compte* pour accéder à cette fonctionnalité.")
    st.stop()
    
# Initialisation des variables de session 
if 'document_buffer' not in st.session_state:
    st.session_state.document_buffer = None
if 'generated_document_name' not in st.session_state:
    st.session_state.generated_document_name = "document_ia.docx"

st.title("📄 Générateur de document IA")
st.markdown("Remplis le formulaire et récupère un document structuré généré par l'IA.")

# Formulaire de saisie
with st.form("formulaire_doc"):
    titre_doc = st.text_input("Titre du document")
    finalite_doc = st.text_input("Finalité du document")
    sommaire = st.text_area("Sommaire", height=150, help="1. Partie A\n2. Partie B")

    uploaded_files = st.file_uploader("Documents sources (PDF)", type=["pdf"], accept_multiple_files=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit = st.form_submit_button("Générer le document")

# --- TRAITEMENT DIRECT AVEC LA PIPELINE ---
if submit:
    if not titre_doc or not finalite_doc or not sommaire:
        st.warning("Merci de remplir tous les champs.")
    else:
        with st.spinner("🧠 Traitement en cours..."):
            try:
                api_key = st.secrets["api_key"]
                llmsherpa_api_url = st.secrets["llmsherpa_api_url"]
                qdrant_url = st.secrets["qdrant_url"]
                model_embed = "bge-multilingual-gemma2"  
                model_llm = "llama-3.3-70b-instruct"  
                
                pdf_paths = []

                for uploaded_file in uploaded_files:
                    if not uploaded_file.name.lower().endswith(".pdf"):
                        st.warning(f"{uploaded_file.name} n’est pas un PDF, ignoré.")
                        continue

                    fn = f"{uuid.uuid4().hex}.pdf"
                    rel_path = os.path.join("uploaded_pdfs", fn)   
                    os.makedirs("uploaded_pdfs", exist_ok=True)

                    with open(rel_path, "wb") as f:
                        f.write(uploaded_file.read())

                    pdf_paths.append(rel_path) 
                
                structure = parser_sommaire(sommaire)
                prompts = generate_prompts_from_structure(structure)
                
                collection_name = create_collection(host=qdrant_url, vector_size=3584)  
                
                with st.status("📄 Analyse des documents en cours.."):
                    docs = load_documents_from_paths(pdf_paths, llmsherpa_api_url=llmsherpa_api_url)
                    section_map = group_docs_by_section(docs)
                    chunks = chunk_sections_with_context(section_map, 4000)

                    st.write("✅ Texte extrait et découpé.")

                    embeddings = embed_text(chunks, model=model_embed, api_key=api_key)
                    upsert_embeddings(embeddings, collection_name, host=qdrant_url)
                    
                    st.write("✅ Embeddings générés et insérés.")
                
                with st.status("Rédaction du document..."):
                    results = generate_sections_from_prompts(
                        prompts=prompts,
                        titre_doc=titre_doc,
                        finalite_doc=finalite_doc,
                        collection_name=collection_name,
                        embedding_fn=embed_text,
                        llm_fn=scaleway_llm,
                        model_embed=model_embed,
                        model_llm=model_llm,
                        api_key=api_key,
                        host=qdrant_url,
                        top_k=3
                    )

                    st.write("✅ Rédaction terminée.")
                
                result_json = {
                    "titre_document": titre_doc,
                    "sommaire": sommaire.strip().split('\n'),
                    "sections": results
                }
                
                st.session_state.document_buffer = generer_docx(result_json)
                st.session_state.generated_document_name = f"{titre_doc}.docx"
                
                delete_collection(collection_name, host=qdrant_url)  
                
                for path in pdf_paths:
                    try:
                        os.unlink(path)
                    except:
                        pass
                
                st.success("📄 Le document est prêt à être téléchargé !")

            except Exception as e:
                st.error(f"Erreur lors du traitement : {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# bouton de téléchargement 
if st.session_state.document_buffer is not None:
    st.download_button(
        label="📥 Télécharger le document Word",
        data=st.session_state.document_buffer,
        file_name=st.session_state.generated_document_name,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key="docx-download"
    )
