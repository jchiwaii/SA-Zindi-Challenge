"""Refit chosen models on all permitted training partitions, with no test reads."""
from pathlib import Path
import json,subprocess,sys
import joblib
BASE=Path(__file__).resolve().parent

def main():
    assert (BASE/'reports/audit_metrics.json').exists(), 'Complete the independent audit before final refitting.'
    tree=json.loads((BASE/'runs/development/training_summary.json').read_text())
    # Earlier runs may predate expanded summaries; recover exact counts from saved weights.
    if 'conservative_iterations' not in tree:
        tree['conservative_iterations']=joblib.load(BASE/'runs/development/conservative.joblib').n_estimators_
    neural=json.loads((BASE/'runs/neural_development/training_summary.json').read_text())
    commands=[
        ['train.py','--name','final','--n-train','160000','--iterations',str(tree['lightgbm_iterations']),'--conservative-iterations',str(tree['conservative_iterations']),'--catboost-iterations',str(tree['catboost_iterations']),'--final'],
        ['neural.py','--name','neural_final','--templates','final','--n-train','100000','--epochs',str(neural['chosen_epoch']),'--schedule-epochs','60','--final'],
    ]
    for command in commands:
        name=command[command.index('--name')+1];log=BASE/'reports'/f'{name}.log'
        print('Starting final fit:',name,'log:',log,flush=True)
        with log.open('w') as output:
            subprocess.run([sys.executable,'-u',str(BASE/command[0])]+command[1:],stdout=output,stderr=subprocess.STDOUT,check=True)
        print('Completed',name,flush=True)
    print('All models fitted. Competition test inputs remain unread by this process.',flush=True)

if __name__=='__main__':main()
