def print_multiplication_table(n: int):
    for i in range(1, 11):
        print(f'{n} * {i} = {n*i}')

if __name__ == '__main__':
    #take a number
    n = int(input('Enter a number: '))
    print_multiplication_table(n)