import fileinput

def SearchReplaceStr(searchExp,replaceExp,file='example.txt'):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            print(replaceExp, end ='\n')
        else:
            print(line, end ='')
