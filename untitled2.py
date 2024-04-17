list = [
        [2, 3, 4],
        [1, 2, 10]
        ]


def square(list):
    s = 0
    for i in range (len(list)):
        s = s + 2*(list[i][0]*list[i][1] + list[i][0]*list[i][2] + list[i][1]*list[i][2])
    return s

print(square(list))