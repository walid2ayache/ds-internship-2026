import numpy as np
areas=np.array([1000,2000,3000,4000,5000])
bedrooms=np.array([1,2,3,4,5])
mean1=np.mean(areas)
mean2=np.mean(bedrooms)
std1=np.std(areas)
std2=np.std(bedrooms)
scaled_areas=(areas-mean1)/std1
scaled_bedrooms=(bedrooms-mean2)/std2
print("scaled areas",scaled_areas)
print("scaled bedrooms",scaled_bedrooms)


# scaler=StandardScaler()
# scaler.fit(areas)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)