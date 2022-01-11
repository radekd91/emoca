import pandas as pd
import wandb
api = wandb.Api()
run1 = api.run("rdanecek/EmotionalDeca/2021_11_09_21-34-35_4654975036132116438")
# run2 = api.run("<entity>/<project>/<run_id>")
df = pd.DataFrame([run1.config]).transpose()
# df = pd.DataFrame([run1.config, run2.config]).transpose()
df.columns = [run1.name, run2.name]
print(df[df[run1.name] != df[run2.name]])