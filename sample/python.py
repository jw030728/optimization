import csv
import matplotlib.pyplot as plt
sale_data=[]
optimal_weight=0
min_difference=-1
#n=분모 m=분자 weight=기울기

with open("sample_data.csv", encoding = 'utf-8-sig') as data: #밑에줄까지 파일읽는거
    reader = csv.DictReader(data)
    for row in reader:
        price = int(row['price'])
        sale_qty= int(row['sale_qty'])
        sale_data.append({'price':price,'qty':sale_qty})
        plt.scatter(price,sale_qty)
          

    for n in range(-100,101):
        if (n==0):
            continue    
        for m in range(1,101):
            weight = m / (n * 1000)

            sum_difference=0
            for sale in sale_data:
                estimate_qty = abs(weight * sale.get('price'))
                difference = abs(estimate_qty - sale.get('qty'))
                sum_difference += difference

            if min_difference < 0 or min_difference > sum_difference: #초기값 음수라고생각
                min_difference = sum_difference
                optimal_weight = weight

    optimal_xaxis = [] #x축 
    optimal_yaxis = [] #y축
    for price in range(10000,100000,1000):
        optimal_xaxis.append(price)
        optimal_yaxis.append(optimal_weight*price)
    print(optimal_weight)

    plt.plot(optimal_xaxis, optimal_yaxis)
    plt.show() 