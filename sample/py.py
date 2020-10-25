import csv
import numpy as np
import matplotlib.pyplot as plt
optimal_weight = 0
optimal_xaxis = []
optimal_yaxis = []

with open("sample_data.csv", encoding = 'utf-8-sig') as data: # 밑에줄까지 파일읽는거
    reader = csv.DictReader(data)
    prices = []
    quantities = []#판매량
    for row in reader:
        price = int(row['price'])
        sale_qty = int(row['sale_qty'])
        prices.append(price)
        quantities.append(sale_qty)
        plt.scatter(price, sale_qty)#scatter 점찍어주는거

x = np.array(prices) #np.array 데이터형태 바꿔줌
y = np.array(quantities)

fit = np.polyfit(x, y, 2)# polyfit(x y 차수)
print(fit)

for price in range(10000,100000,1000):
    optimal_xaxis.append(price)
    optimal_yaxis.append(fit[0] * (price **2) + fit[1] * price + fit[2])#2차식

print(optimal_weight)
plt.plot(optimal_xaxis, optimal_yaxis)
plt.show()
