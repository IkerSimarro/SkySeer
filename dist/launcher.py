import subprocess
import webbrowser
import time
process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(5)
webbrowser.open("http://localhost:8501")
process.wait()