import os

cwd = os.getcwd()
if os.name == 'nt':
    cmd = f'explorer "{cwd}"'
    os.system(cmd)
    cmd = f'wt -d "{cwd}"'
    os.system(cmd)
    
    print("Setup done.")

    os.system("cls")
