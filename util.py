from flask import Flask,request, url_for, redirect, render_template
import requests
import pickle
model_msft_close=pickle.load(open('model_close.pkl','rb'))
model_msft_high=pickle.load(open('model_high.pkl','rb'))
model_msft_low=pickle.load(open('model_low.pkl','rb'))
model_time_close=pickle.load(open('dataframe_close.pkl','rb'))
model_time_high=pickle.load(open('dataframe_high.pkl','rb'))
model_time_low=pickle.load(open('dataframe_low.pkl','rb'))
app = Flask(__name__)

@app.route("/",methods=['POST','GET'])
def home():
  return render_template('index.html')
@app.route("/msft",methods=['POST','GET'])
def predict1():
  if request.method == 'POST':
      date = request.form["date"]
      open = int(request.form["open"])
      dt = date.split('-')
      year = int(dt[0])
      month = int(dt[1])
      day = int(dt[2]) 
      feat = [[day,month,year,open]]
      close = model_msft_close.predict(feat)
      high = model_msft_high.predict(feat)
      low = model_msft_low.predict(feat)
      return render_template('msft.html', close = str(close[0]),high = str(high[0]), low = str(low[0]))
  else:
      return render_template('msft.html')
@app.route("/dataframe",methods=['POST','GET'])
def predict2():
  if request.method == 'POST':
      date = request.form["date"]
      open = int(request.form["open"])
      dt = date.split('-')
      year = int(dt[0])
      month = int(dt[1])
      day = int(dt[2]) 
      time = request.form["time"].split(":")
      hour = int(time[0])
      min = int(time[1])
      feat = [[day,month,year,hour,min,open]]
      close = model_time_close.predict(feat)
      high = model_time_high.predict(feat)
      low = model_time_low.predict(feat)
      return render_template('dataframe.html', close = str(close[0]),high = str(high[0]), low = str(low[0]))
  else:
      return render_template('dataframe.html')


if __name__ == "__main__":
  app.run(debug=True)