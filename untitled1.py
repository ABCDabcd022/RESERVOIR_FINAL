ticket = input("Введите строку: ")

def function(ticket):
    sum1 = 0
    sum2 = 0
    for i in range(0, 3):
        sum1 = sum1 + int(ticket[i])
        
    for i in range(3, 6):
        sum2 = sum2 + int(ticket[i])
        
    if (sum1 == sum2):
        return "Lucky!"
    else: 
        return "Unlucky!"

print(function(ticket))

