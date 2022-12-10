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

#Prediccion CON VALORES ENTEROS
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#clasificacion CON PUNTO FLOTANTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

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
    kmeansIMG = os.path.join(path)
    return render_template("Kmeans.html", user_image=kmeansIMG)

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
    dbscanIMG = os.path.join(path)
    return render_template("DBscan.html", user_image=dbscanIMG)    

@app.route ("/prediccion",methods=['GET'])
def prediccion():
    return render_template('/prediccion.html')

#prediccion
@app.route ("/prediccion1",methods=['POST'])
def prediccion1():
    dataset = pd.read_csv("static/files/DataFinal(1).csv")
    x = request.form.get("columna1")
    y = request.form.get("columna2")
    path = 'static/prediccion_images/pred{}{}.png'.format(x, y)
    X = np.array(dataset[[x, y]])
    x = X[:,0]
    y = X[:,1]
    x = x.reshape(1150,1)
    y = y.reshape(1150,1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)
    #DecisionTreeRegressor
    model1 = DecisionTreeRegressor()
    model1.fit(X_train, y_train)
    predictions1 = model1.predict(X_test)
    errors1 = abs(predictions1 - y_test)
    aux1 = 100*(errors1 / y_test)
    precision1 = np.mean(aux1)
    precision1 = round(precision1, 2)
    #SVR
    regressor = SVR(kernel = "rbf")
    regressor.fit(X_train,y_train)
    y_predSVR = regressor.predict(X_test)
    errors2 = abs(y_predSVR - y_test)
    aux2 = 100*(errors2 / y_test)
    precision2 = 100 - np.mean(aux2)
    precision2 = round(precision2, 2)
    #RandomForestRegressor
    randomf = RandomForestRegressor(n_estimators = 100, random_state = 0)
    randomf.fit(X_train,y_train)
    y_predRF = randomf.predict(X_test)
    errors3 = abs(y_predRF - y_test)
    aux3 = 100*(errors3 / y_test)
    precision3 = 100 - np.mean(aux3)
    precision3 = round(precision3, 2)
    #Grafica
    plt.bar(["DecisionTreeRegressor", "SVR", "RandomForestRegressor"], [precision1, precision2, precision3])
    plt.savefig(path)
    prediccionIMG = os.path.join(path)
    return render_template("prediccion.html", pred1=precision1, pred2=precision2, pred3=precision3, user_image=prediccionIMG)

#clasificacion
@app.route ("/clasificacion",methods=['GET'])
def clasificacion():
    return render_template('/clasificacion.html')

@app.route ("/clasificacion1",methods=['GET','POST'])
def clasificacion1():
    dataset = pd.read_csv("static/files/DataFinal(1).csv")
    nombreCol = request.form.get("columna1")
    if nombreCol == 'Graduado / No Graduado':
        path = 'static/clasificacion_images/predGraduado.png'
    else:
        path = 'static/clasificacion_images/pred{}.png'.format(nombreCol)
    x = dataset.drop([nombreCol, "ID"], axis = 1)
    y = dataset[nombreCol]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)
    #KNeighborsClassifier
    clasificador1 = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    clasificador1 = clasificador1.fit(X_train, y_train)
    y_pred = clasificador1.predict(X_test)
    precision1 = precision_score(y_test, y_pred)
    precision1 = precision1*100
    mc1 = confusion_matrix(y_test, y_pred)
    #GaussianNB
    clasificador2 = GaussianNB()
    clasificador2 = clasificador2.fit(X_train, y_train)
    y_pred = clasificador2.predict(X_test)
    precision2 = precision_score(y_test, y_pred)
    precision2 = precision2*100
    mc2 = confusion_matrix(y_test, y_pred)
    #DecisionTreeClassifier
    clasificador3 = DecisionTreeClassifier()
    clasificador3 = clasificador3.fit(X_train, y_train)
    y_pred = clasificador3.predict(X_test)
    precision3 = precision_score(y_test, y_pred)
    precision3 = precision3*100
    mc3 = confusion_matrix(y_test, y_pred)
    #Grafica
    plt.bar(["DecisionTreeRegressor", "SVR", "RandomForestRegressor"], [precision1, precision2, precision3])
    plt.savefig(path)
    clasificacionIMG = os.path.join(path)
    return render_template("clasificacion.html", pred1=precision1, pred2=precision2, pred3=precision3, user_image=clasificacionIMG,
                                                mc01=mc1[0][0], mc02=mc1[0][1], mc03=mc1[1][0], mc04=mc1[1][1],
                                                mc11=mc2[0][0], mc12=mc2[0][1], mc13=mc2[1][0], mc14=mc2[1][1],
                                                mc21=mc3[0][0], mc22=mc3[0][1], mc23=mc3[1][0], mc24=mc3[1][1])

if __name__=='__main__':
    app.run(port=5000, debug=True)