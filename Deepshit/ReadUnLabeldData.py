# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 14:39:35 2017

@author: awagner
"""

def ReadUnLabeldData(sec,freq,session_ids):

    X_table_ses = []
    Y_table_ses = []
    Z_table_ses = []
    for ses in session_ids:
        #ids_str = ','.join(['%d' % x for x in session_ids])
        print(ses)
    
        text = ("select  acc.SessionId, acc.DeviceID, acc.X, acc.Y, acc.Z "
                "FROM [dbo].[ProcessedAccelerometerData] as acc "
                "where (acc.[SessionId] in (%s)) and (acc.DeviceId = 14) "
                "ORDER BY [TS]")
        query = text % ses
    
        conn = pyodbc.connect("DSN=" + dsn, autocommit=True)
        res_home = pd.read_sql(query, conn)
        close_connection(conn)
    
    
        raw_home_x = np.ones((res_home.shape[0]/(sec*freq),sec*freq))
        raw_home_y = np.zeros((res_home.shape[0]/(sec*freq),sec*freq))
        raw_home_z = np.zeros((res_home.shape[0]/(sec*freq),sec*freq))
        for i in range(int(np.floor(res_home.shape[0]/(sec*freq)))):
            #print(i)
            raw_home_x[i] = np.asarray(res_home['X'][i*sec*freq:(i+1)*sec*freq])
            raw_home_y[i] = np.asarray(res_home['Y'][i*sec*freq:(i+1)*sec*freq])
            raw_home_z[i] = np.asarray(res_home['Z'][i*sec*freq:(i+1)*sec*freq])
        X_table_ses.append(raw_home_x)    
        Y_table_ses.append(raw_home_y)
        Z_table_ses.append(raw_home_z)    
        
    X_table = np.vstack(X_table_ses)
    Y_table = np.vstack(Y_table_ses)
    Z_table = np.vstack(Z_table_ses)
        
    return X_table, Y_table, Z_table
    
    
