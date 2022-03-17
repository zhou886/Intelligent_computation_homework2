import numpy as np

def main():
    with open("./dataNoisy.txt","r+") as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        f.writelines(lines)
    
if __name__ == "__main__":
    main()