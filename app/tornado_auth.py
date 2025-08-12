# auth.py
import tornado.web
from tornado.web import RequestHandler

# This is where to redirect unauthenticated users
login_url = "/login"

# A simple in-memory user database (replace with a real DB in production)
users = {
    "meouser": {"password": "me0ana1yz3r"}
}

# Login handler that processes the login form
class LoginHandler(RequestHandler):
    def get(self):
        self.write('<html><body><form action="/login" method="post">'
                   'Username: <input type="text" name="username"><br>'
                   'Password: <input type="password" name="password"><br>'
                   '<input type="submit" value="Sign in">'
                   '</form></body></html>')
    
    def post(self):
        username = self.get_argument("username", "")
        password = self.get_argument("password", "")
        
        if username in users and users[username]["password"] == password:
            # Set a secure cookie
            self.set_secure_cookie("user", username)
            # Redirect to the main page
            self.redirect("/")
        else:
            self.write('<html><body>Invalid login. <a href="/login">Try again</a></body></html>')

# Optional logout handler
logout_url = "/logout"

class LogoutHandler(RequestHandler):
    def get(self):
        self.clear_cookie("user")
        self.redirect(login_url)

# The function that returns the current user or None
def get_user(request_handler):
    user_cookie = request_handler.get_secure_cookie("user")
    if user_cookie:
        return user_cookie.decode('utf-8')
    return None