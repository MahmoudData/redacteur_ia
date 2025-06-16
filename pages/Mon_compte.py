import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Mon Compte", page_icon="🔐")

# Chargement de la config d'auth
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# Affichage du formulaire de connexion
authenticator.login(location='main', key='login')

# Gestion des états d'authentification
if st.session_state.get("authentication_status"):
    st.sidebar.success(f"Connecté en tant que {st.session_state.get('name')}")
    authenticator.logout("Se déconnecter", "sidebar")
    st.success("Vous êtes connecté ! Vous pouvez maintenant accéder à la page principale.")
elif st.session_state.get("authentication_status") is False:
    st.error("Nom d'utilisateur ou mot de passe incorrect.")
elif st.session_state.get("authentication_status") is None:
    st.warning("Veuillez entrer vos identifiants.")
