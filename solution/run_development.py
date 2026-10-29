"""Run the predeclared full comparisons once the verified source subset is ready."""
from pathlib import Path
import subprocess,sys,time
BASE=Path(__file__).resolve().parent

def main():
    while not (BASE/'cache/reference_manifest.csv').exists() or not (BASE/'runs/expanded_pilot/validation_metrics.json').exists():
        time.sleep(5)
    commands=[
        ['train.py','--name','development','--n-train','100000','--n-val','2500','--iterations','2000'],
        ['neural.py','--name','neural_development','--templates','development','--n-train','80000','--n-val','2500','--epochs','60'],
    ]
    for command in commands:
        name=command[command.index('--name')+1]
        log=BASE/'reports'/f'{name}.log'
        print('Starting',name,'log:',log,flush=True)
        with log.open('w') as output:
            subprocess.run([sys.executable,'-u',str(BASE/command[0])]+command[1:],stdout=output,stderr=subprocess.STDOUT,check=True)
        print('Completed',name,flush=True)

if __name__=='__main__':main()
