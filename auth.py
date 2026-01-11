import json
import os

USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(username, password):
    users = load_users()

    for user in users:
        if user["username"] == username:
            return False, "User already exists"

    users.append({
        "username": username,
        "password": password,
        "role": "user"
    })

    save_users(users)
    return True, "User registered"

def authenticate_user(username, password):
    users = load_users()

    for user in users:
        if user["username"] == username and user["password"] == password:
            return True, user["role"]

    return False, None
