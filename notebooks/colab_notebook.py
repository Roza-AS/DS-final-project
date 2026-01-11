
def run_streamlit():
    !streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Keep the cell running a bit to let Streamlit start
time.sleep(5)
print("If the app doesn't open, re-run this cell or check logs above.")
