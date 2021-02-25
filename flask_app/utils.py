from flask_app import init_app
from flask_sqlalchemy import SQLAlchemy
from flask_app.models import User
from werkzeug.security import generate_password_hash

app = init_app()

db = SQLAlchemy(app)
db.create_all()

app.app_context().push()
username = input("Username:")
email = input("Email:")
password = input("Password:")
user = User(
    username=username,
    email=email,
    password=generate_password_hash(
        password,
        method='sha256',
    )
)
db.session.add(user)
db.session.commit()
