data = (
        '00-50-07',
        '00-53-10',
        '00-56-09',
        '00-59-17',
        '01-02-40',
        '01-05-58',
)

seconds = []

for index, i in enumerate(data):
    splited = i.split('-')
    seconds.append(int(splited[0])*60*60 + int(splited[1])*60 + int(splited[2]))
    if index > 0:
        print(seconds[index] - seconds[index-1])
