from flask import Flask

app = Flask(_name_)

@app.route('/')
def home():
	return "Flask heroku app"

if __name__ == '__main__'
	app.run()