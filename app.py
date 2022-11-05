#Seunfo video
#Minuto 10
from flask import (Flask, render_template, url_for, request, redirect, jsonify)
from flask import flash
from flask_mysqldb import MySQL
import os as os
import pandas as pd
from sklearn import datasets
from werkzeug import Request

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import seaborn as sns
import numpy  as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key ="mineriadedatos"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'crud'
mysql = MySQL(app)

@app.route('/')
def main():
    link = mysql.connection.cursor()
    link.execute("SELECT * FROM employes")
    data = link.fetchall()
    return render_template('index.html',empleados=data)

@app.route('/login')
def login():
    return "En construcción!!!"

@app.route('/viewemployes', methods=['POST', 'GET'])
def viewemployes():
    if request.method == 'POST':
        id = request.form['Documento']
        link = mysql.connection.cursor()
        link.execute("SELECT * FROM employes WHERE ID = %s", [id])
        data = link.fetchall()
    return jsonify({'htmlresponse': render_template('viewemployes.html', empleados=data)})

@app.route('/addemployes', methods=['POST', 'GET'])
def addemployes():
    if request.method == 'POST':
        edad = request.form["EDAD"]
        sexo = request.form["SEXO"]
        estrato = request.form["ESTRATO"]
        naturalezaColegio = request.form["NATURALEZA DE COLEGIO"]
        puntajeIcfes = request.form["PUNTAJE INSCRIPCIÓN ICFES (No matriculado)"]
        valorPagado = request.form["VALOR PAGADO"]
        calAcademica = request.form["CALIF_ACADEMICA"]
        calEconomica = request.form["CALIF_ECONOMICO"]
        calFamiliar = request.form["CALIF_FAMILIAR"]
        calPsicosocial = request.form["CALIF_PSICOSOCIAL"]
        depresion = request.form["DEPRESION"]
        ansiedad = request.form["ANSIEDAD"]
        punTabaco = request.form["PUNTAJETABACO"]
        punAlcohol = request.form["PUNTAJEALCOHOL"]
        punCannabis = request.form["PUNTAJECANNABIS"]
        punCocaina = request.form["PUNTAJECOCAINA"]
        punAnfetaminas = request.form["PUNTAJEANFETAMINA"]
        punInhalante = request.form["PUNTAJEINHALANTE"]
        punSedante = request.form["PUNTAJESEDANTE"]
        punAlucinogeno = request.form["PUNTAJEALUCINOGENO"]
        punOpiaceo = request.form["PUNTAJEOPIACEO"]
        punOtradroga = request.form["PUNTAJEOTRADROGA"]
        graduado = request.form["Graduado / No Graduado"]
        id = request.form["ID"]
        link = mysql.connection.cursor()
        link.execute("INSERT INTO `employes` (`EDAD`, `SEXO`, `ESTRATO`, `NATURALEZA DE COLEGIO`, `PUNTAJE INSCRIPCIÓN ICFES (No matriculado)`, `VALOR PAGADO`, `CALIF_ACADEMICA`, `CALIF_ECONOMICO`, `CALIF_FAMILIAR`, `CALIF_PSICOSOCIAL`, `DEPRESIÓN`, `ANSIEDAD`, `PUNTAJETABACO`, `PUNTAJEALCOHOL`, `PUNTAJECANNABIS`, `PUNTAJECOCAINA`, `PUNTAJEANFETAMINA`, `PUNTAJEINHALANTE`, `PUNTAJESEDANTE`, `PUNTAJEALUCINOGENO`, `PUNTAJEOPIACEO`, `PUNTAJEOTRADROGA`, `Graduado / No Graduado`, `ID`) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(edad, sexo, estrato, naturalezaColegio, puntajeIcfes, valorPagado, calAcademica, calEconomica, calFamiliar, calPsicosocial, depresion, ansiedad, punTabaco, punAlcohol, punCannabis, punCocaina, punAnfetaminas, punInhalante, punSedante, punAlucinogeno, punOpiaceo, punOtradroga, graduado, id))
        mysql.connection.commit()
        link.close()
        flash("Estudiante registrado correctamente")
        return redirect(url_for('main'))

@app.route('/updateemployes', methods=['POST', 'GET'])
def updateemployes():
    if request.method == 'POST':
        edad = request.form["EDAD"]
        sexo = request.form["SEXO"]
        estrato = request.form["ESTRATO"]
        naturalezaColegio = request.form["NATURALEZA DE COLEGIO"]
        puntajeIcfes = request.form["PUNTAJE INSCRIPCIÓN ICFES (No matriculado)"]
        valorPagado = request.form["VALOR PAGADO"]
        calAcademica = request.form["CALIF_ACADEMICA"]
        calEconomica = request.form["CALIF_ECONOMICO"]
        calFamiliar = request.form["CALIF_FAMILIAR"]
        calPsicosocial = request.form["CALIF_PSICOSOCIAL"]
        depresion = request.form["DEPRESION"]
        ansiedad = request.form["ANSIEDAD"]
        punTabaco = request.form["PUNTAJETABACO"]
        punAlcohol = request.form["PUNTAJEALCOHOL"]
        punCannabis = request.form["PUNTAJECANNABIS"]
        punCocaina = request.form["PUNTAJECOCAINA"]
        punAnfetaminas = request.form["PUNTAJEANFETAMINA"]
        punInhalante = request.form["PUNTAJEINHALANTE"]
        punSedante = request.form["PUNTAJESEDANTE"]
        punAlucinogeno = request.form["	PUNTAJEALUCINOGENO"]
        punOpiaceo = request.form["PUNTAJEOPIACEO"]
        punOtradroga = request.form["PUNTAJEOTRADROGA"]
        graduado = request.form["Graduado / No Graduado"]
        id = request.form["ID"]
        link = mysql.connection.cursor()
        link.execute("UPDATE employes SET EDAD= %s,SEXO=%s,ESTRATO=%s,NATURALEZA DE COLEGIO=%s,PUNTAJE INSCRIPCIÓN ICFES (No matriculado)=%s,VALOR PAGADO=%s,CALIF_ACADEMICA=%s,CALIF_ECONOMICO=%s,	CALIF_FAMILIAR=%s,CALIF_PSICOSOCIAL=%s,DEPRESIÓN=%s,ANSIEDAD=%s,PUNTAJETABACO=%s,PUNTAJEALCOHOL=%s,PUNTAJECANNABIS=%s,PUNTAJECOCAINA=%s,PUNTAJEANFETAMINA=%s,PUNTAJEINHALANTE=%s,PUNTAJESEDANTE=%s,PUNTAJEALUCINOGENO=%s,PUNTAJEOPIACEO=%s,PUNTAJEOTRADROGA=%s,Graduado / No Graduado=%s WHERE ID=%s",(edad, sexo, estrato, naturalezaColegio, puntajeIcfes, valorPagado, calAcademica, calEconomica, calFamiliar, calPsicosocial, depresion, ansiedad, punTabaco, punAlcohol, punCannabis, punCocaina, punAnfetaminas, punInhalante, punSedante, punAlucinogeno, punOpiaceo, punOtradroga, graduado, id))
        mysql.connection.commit()
        link.close()
        flash("Estudiante actualizado correctamente")
        return redirect(url_for('main'))

@app.route('/deleteemployes/<string:Documento>', methods=['POST', 'GET'])
def deleteemployes(Documento):
    if request.method == 'GET':
        link = mysql.connection.cursor()
        link.execute("DELETE FROM `employes` WHERE ID=%s",[Documento])
        mysql.connection.commit()
        link.close()
        flash("Estudiante eliminado correctamente")
        return redirect(url_for('main'))

@app.route('/cargarcsv')
def cargarcsv():
    return render_template('cargarcsv.html')

@app.route('/uploadcsv', methods=['POST','GET'])
def uploadcsv():
    if request.method == 'POST':
        upload_file = request.files['csvfile']
        if upload_file.filename != '':

            file_path = os.path.join(app.config['UPLOAD_FOLDER'],
             upload_file.filename)
            upload_file.save(file_path)
            grabarCSV(file_path)
        flash("DataSet Cargado correctamente")
        return redirect(url_for('cargarcsv'))

def grabarCSV(filepath):
    columnas = ['EDAD','SEXO','ESTRATO','NATURALEZA DE COLEGIO','PUNTAJE INSCRIPCIÓN ICFES (No matriculado)','VALOR PAGADO','CALIF_ACADEMICA','CALIF_ECONOMICO','CALIF_FAMILIAR','CALIF_PSICOSOCIAL','DEPRESIÓN','ANSIEDAD','PUNTAJETABACO','PUNTAJEALCOHOL','PUNTAJECANNABIS','PUNTAJECOCAINA','PUNTAJEANFETAMINA','PUNTAJEINHALANTE','PUNTAJESEDANTE','PUNTAJEALUCINOGENO','PUNTAJEOPIACEO','PUNTAJEOTRADROGA','Graduado / No Graduado','ID']
    csvData = pd.read_csv(filepath)
    link = mysql.connection.cursor()
    for i, row in csvData.iterrows():
        sql = "INSERT INTO employes (`EDAD`, `SEXO`, `ESTRATO`, `NATURALEZA DE COLEGIO`, `PUNTAJE INSCRIPCIÓN ICFES (No matriculado)`, `VALOR PAGADO`, `CALIF_ACADEMICA`, `CALIF_ECONOMICO`, `CALIF_FAMILIAR`, `CALIF_PSICOSOCIAL`, `DEPRESIÓN`, `ANSIEDAD`, `PUNTAJETABACO`, `PUNTAJEALCOHOL`, `PUNTAJECANNABIS`, `PUNTAJECOCAINA`, `PUNTAJEANFETAMINA`, `PUNTAJEINHALANTE`, `PUNTAJESEDANTE`, `PUNTAJEALUCINOGENO`, `PUNTAJEOPIACEO`, `PUNTAJEOTRADROGA`, `Graduado / No Graduado`, `ID`)VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        valores = (row['EDAD'],row['SEXO'],row['ESTRATO'],row['NATURALEZA DE COLEGIO'],row['PUNTAJE INSCRIPCIÓN ICFES (No matriculado)'],row['VALOR PAGADO'],row['CALIF_ACADEMICA'],row['CALIF_ECONOMICO'],row['CALIF_FAMILIAR'],row['CALIF_PSICOSOCIAL'],row['DEPRESIÓN'],row['ANSIEDAD'],row['PUNTAJETABACO'],row['PUNTAJEALCOHOL'],row['PUNTAJECANNABIS'],row['PUNTAJECOCAINA'],row['PUNTAJEANFETAMINA'],row['PUNTAJEINHALANTE'],row['PUNTAJESEDANTE'],row['PUNTAJEALUCINOGENO'],row['PUNTAJEOPIACEO'],row['PUNTAJEOTRADROGA'],row['Graduado / No Graduado'],row['ID'])
        link.execute(sql,valores)
        mysql.connection.commit()

@app.route ("/Kmeans",methods=['GET'])
def Kmeans():
    return render_template('/Kmeans.html')

@app.route('/kmeans1',methods=['POST'])
def kmeans1():
    dataset = pd.read_csv("static/files/DataFinal(1).csv")
    x = request.form.get("columna1")
    y = request.form.get("columna2")
    X = np.array(dataset[[x, y]])
    k_means = KMeans(n_clusters=4).fit(X)
    centroides = k_means.cluster_centers_
    etiquetas = k_means.labels_
    fig = plt.figure(figsize=(6,6))
    plt.scatter(X[:,0], X[:,1], c=etiquetas, cmap='rainbow')
    plt.scatter(centroides[:,0], centroides[:,1], color="black", marker="*",s=100)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Kmeans')
    path = 'static/kmeans_images/kmeans{}{}.png'.format(x, y)
    plt.savefig(path)
    Flask_Logo = os.path.join(path)
    return render_template("Kmeans.html", user_image=Flask_Logo)

@app.route ("/DBscan",methods=['GET'])
def DBscan():
    return render_template('/DBscan.html')

@app.route('/DBscan1',methods=['POST'])
def DBscan1():
    dataset = pd.read_csv("static/files/DataFinal(1).csv")
    x = request.form.get("columna1")
    y = request.form.get("columna2")
    X = np.array(dataset[[x, y]])
    model = DBSCAN(eps = 0.2).fit(X)
    clusters = model.fit_predict(X)
    fig = sns.scatterplot(X[:, 0], X[:, 1], hue = clusters)
    path = 'static/dbscan_images/DBscan{}{}.png'.format(x, y)
    fig.figure.savefig(path)
    Flask_Logo = os.path.join(path)
    return render_template("DBscan.html", user_image=Flask_Logo)    

if __name__=='__main__':
    app.run(port=5000, debug=True)