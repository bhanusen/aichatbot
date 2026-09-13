import phoenix as px
import time

print("Starting Arize Phoenix server...")
session = px.launch_app()
print(f"Phoenix server started at {session.url}")

while True:
    time.sleep(60)
