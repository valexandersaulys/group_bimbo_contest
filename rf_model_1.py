"""
First RandomForest Model I built
"""
from sklearn.ensemble import RandomForestClassifier
import random, time
from readin_data import read_data, get_random_data
from sklearn.externals import joblib

model = RandomForestClassifier(n_estimators=500,n_jobs=6,verbose=1)

features = [u'Semana', u'Agencia_ID', u'Canal_ID', u'Ruta_SAK', u'Cliente_ID',
            u'Producto_ID', u'Venta_uni_hoy', u'Venta_hoy', u'Dev_uni_proxima',
            u'Dev_proxima']
target =  u'Demanda_uni_equil'

start_time = time.time()

i = 0; rl = [];
for x in read_data(10000):
    # on iteration #3052/52
    print "Going on Iteration #%d" % int(i+1);
    random_indice = random.randint(0,10000)
    raa = x.iloc[random_indice,:]
    rl.append(raa)
    rd = range(10000)
    rd.remove(random_indice)
    x_tr = x.iloc[rd]
    # Do I need to return it like this or is it self-acting? not sure
    model = model.fit(x_tr[features],x_tr[target])  
    if (i % 10)==0:
        # no overwrite needed apparently
        print "# # # # # # Model Saved!"
        joblib.dump(model,"rf_model_1/first_rf.pkl")  # use .load('first_rf.pkl') to get back
    i += 1;

print "Training took %f seconds" % (time.time() - start_time);
    
#random_data = get_random_data(10)
#random_data = pd.read_csv("data/valid_set.csv")
random_data = pd.concat(rl)
random_data.to_csv("data/valid_set_rf_model_1.csv")

preds = model.predict(random_data[features])
truths = random_data[target].values  # return a NumPy array


from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(truths, preds)
cm = confusion_matrix(truths, preds)

print "Accuracy Score: %f" % acc;
print cm;
