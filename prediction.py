import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cp = pd.read_csv("C:\\Users\\sv\\Desktop\\python\\Country-Population.csv")

cp.columns = ['Country_Name', 'Country_Code', 'Indicator_Name', 'Indicator_Code',
       '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
       '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
       '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
       '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
       '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013']


def population():
    cname = input("Country Name:")
    cname = cname.capitalize()
    cname1 = cp[cp.Country_Name == cname]
    for i in cp.Country_Name:
        if cname  == i:
            X1 = cname1.iloc[:, 4:].values
            y1 = cname1.columns[4:]
        
            l1 = y1
            l1 = list(map(int, l1))
            #break
    for j in X1:
        n1 = list(j)
        n1 = list(map(int, n1))
    df = pd.DataFrame({
            "Year": np.array(l1),
            "Population":np.array(n1)})
    X = df.iloc[:, 1].values
    Y = df.iloc[:, 0].values
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1,1))
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train,Y_train)
    
    pred = regressor.predict(X_test)
    
    plt.scatter(X_test,Y_test,color = 'red')
    plt.plot(X_test,pred,color = 'purple')
    plt.title('Population Prediction for ' + cname)
    plt.show()

population()