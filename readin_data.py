# Helper function to read in the data
import pandas as pd
from random import randint

def read_data(n,i=0):           # pass how many lines to read at once
    data_length = 74180464; # length of the data we're reading
    i = i * n;
    while i < data_length:
          df = pd.read_csv("data/train.csv",skiprows=i,nrows=n)
          df.columns = [u'Semana', u'Agencia_ID', u'Canal_ID', u'Ruta_SAK', u'Cliente_ID',
                        u'Producto_ID', u'Venta_uni_hoy', u'Venta_hoy', u'Dev_uni_proxima',
                        u'Dev_proxima', u'Demanda_uni_equil']
          yield df;
          i += n;


def get_random_data(n):
    # This is really memory intensive, might need to chomp down on this a bit
    data_length = 74180464;
    ls = [randint(0,data_length) for i in range(n)]
    a = 0;
    b = [];
    for l in ls:
        #print l;
        df = pd.read_csv("data/train.csv",skiprows=l,nrows=1)
        df.columns = [u'Semana', u'Agencia_ID', u'Canal_ID', u'Ruta_SAK', u'Cliente_ID',
                        u'Producto_ID', u'Venta_uni_hoy', u'Venta_hoy', u'Dev_uni_proxima',
                        u'Dev_proxima', u'Demanda_uni_equil']
        b.append(df)
    return pd.concat(b)


        
