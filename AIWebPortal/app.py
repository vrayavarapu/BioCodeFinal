import os
from flask import Flask

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

from routes import *
