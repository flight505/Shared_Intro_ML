mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"jvan@dtu.dk\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\