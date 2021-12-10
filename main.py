import math

def main():
    bin_capacity = 10
    items = [1,4,5,3,2,7,3,4,6,2,3,5,6,2,4,7,2,1,4,6,7,2,3,5,7,2,3,3,7,2,9,3,5,7,3,4,6]

    print(f"Lower bound: {math.ceil(sum(items)/bin_capacity)}")
    
    nf = nextfit(items, bin_capacity)
    print(f"Next fit: {nf}")

def nextfit(items, bin_capacity):
    res = 0
    rem = bin_capacity
    for i in range(len(items)):
        if rem >= items[i]:
            rem = rem - items[i]
        else:
            res += 1
            rem = bin_capacity - items[i]
    return res

if __name__ == '__main__':
    main()