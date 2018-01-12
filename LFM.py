def LFM_model(R, userID, itemID, K, steps, a, b):
    # 初始化参数q,p矩阵, 随机
    arrayp = np.random.rand(len(userID), K)
    arrayq = np.random.rand(K, len(itemID))
    P = pd.DataFrame(arrayp, columns=range(0,K), index=userID)
    Q = pd.DataFrame(arrayq, columns=itemID, index=range(0,K))
    # LFM
    E_in_append = []
    for step in xrange(steps):
        eij_hat = 0; E_in = 0
        for i in xrange(len(R)):
            for j in xrange(len(R.columns)):
                # i = 0; j = 1
                if R.iloc[i,j] > 0:
                    # eij
                    eij = R.iloc[i,j] - np.dot(P.iloc[i,:], Q.iloc[:,j])
                    P.iloc[i,:] += a*(2*eij*Q.iloc[:,j]-b*P.iloc[i,:])
                    Q.iloc[:,j] += a*(2*eij*P.iloc[i,:]-b*Q.iloc[:,j])
                    # eij_hat
                    eij_hat = R.iloc[i,j] - np.dot(P.iloc[i,:], Q.iloc[:,j])
                    E_in += eij_hat**2 + (b/2)*P.iloc[i,:].dot(P.iloc[i,:]) + (b/2)*Q.iloc[:,j].dot(Q.iloc[:,j])
        E_in_append.append(E_in)
        R_hat = pd.DataFrame(np.dot(P, Q), index=userID, columns=itemID)
        if round(E_in,1) <= 1: break
    return R_hat, E_in, E_in_append
    
def LFM_model(R, userID, itemID, K, steps, a, b):
    # 初始化参数q,p矩阵, 随机
    arrayp = np.random.rand(len(userID), K)
    arrayq = np.random.rand(K, len(itemID))
    P = pd.DataFrame(arrayp, columns=range(0,K), index=userID)
    Q = pd.DataFrame(arrayq, columns=itemID, index=range(0,K))
    # LFM
    E_in_append = []
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R.columns)):
                # i = 0; j = 3
                if R.iloc[i,j] > 0:
                    eij = R.iloc[i,j] - np.dot(P.iloc[i,:], Q.iloc[:,j])
                    P.iloc[i,:] += a*(2*eij*Q.iloc[:,j]-b*P.iloc[i,:])
                    Q.iloc[:,j] += a*(2*eij*P.iloc[i,:]-b*Q.iloc[:,j])
        R_hat = pd.DataFrame(np.dot(P, Q), index=userID, columns=itemID)
        eij_hat = 0; E_in = 0
        for i in range(len(R)):
            for j in range(len(R.columns)):
                if R.iloc[i,j] > 0:
                    eij_hat = R.iloc[i,j] - R_hat.iloc[i,j]
                    E_in += eij_hat**2 + (b/2)*P.iloc[i,:].dot(P.iloc[i,:]) + (b/2)*Q.iloc[:,j].dot(Q.iloc[:,j])
        E_in_append.append(E_in)
        if round(E_in,1) <= 0.8:
            break
    return R_hat, E_in, E_in_append
