import numpy as np
import pandas as pd

from pathlib import Path


mturk_root = Path("/ps/scratch/ps_shared/rdanecek/mturk_study")

# right
catch_sample_corrent_answer = 1


def get_experiment_score(experiment_name, image_filenames, answers, correct_answers):
    answers = answers.split(",")
    if answers[-1] == '':
        del answers[-1]
    clicked = np.array([int(a) for a in answers])



    assert clicked.size == len(image_filenames)
    discard_first_n = 15
    answers = clicked[discard_first_n:]
    image_filenames = image_filenames[discard_first_n:]

    # find indices where image_filenames is catch_sample.png
    catch_indices = [i for i, x in enumerate(image_filenames) if x == "catch_sample.png"]

    catch_incorrect = 0
    for ci in catch_indices:
        if answers[ci] != catch_sample_corrent_answer:
            catch_incorrect += 1
            # print("Filterign out a catch!")

    catch_correct = True
    if catch_incorrect > 0:
        catch_correct = False

    catch_indices.sort(reverse=True)
    answers = answers.tolist()
    for ci in catch_indices:
        del answers[ci]
        del image_filenames[ci]
    answers = np.array(answers)

    # recover indices from image filenames
    image_indices = []
    for im in image_filenames:
        idx = int(Path(im).stem)
        image_indices += [idx]

    image_indices = np.array(image_indices, dtype=np.int32)

    correct_image_answers = correct_answers[image_indices]

    num_correct_answers = correct_image_answers == answers
    if experiment_name == "EmocaDetail-Deep3DFace":
        print("Correct", catch_correct)
        print( num_correct_answers.mean())
        print(" ")
    return catch_correct, num_correct_answers.mean()


def main():
    # results_file = mturk_root / "EMOCA_PartialResults_43of48.csv"
    results_file = mturk_root / "EMOCA_FinalResults.csv"
    table = pd.read_csv(results_file)

    experiment_input_table = pd.read_csv(mturk_root / "mturk_images_final_rel.csv")


    # find all directories in mturk_root using pathlib
    # experiments = sorted(p for p in list(mturk_root.glob("*")) if p.is_dir())

    experiment2scores = {}


    for i in range(len(experiment_input_table)):
        # get the experiment row
        exp_row = experiment_input_table.iloc[i]
        experiment_name = str(Path(exp_row[1].split(';')[0]).parent)
        image_filenames = exp_row[1].split(';')
        # input_file = experiment / "mturk_images_selected.csv"


        correct_answers_table = pd.read_csv(mturk_root / experiment_name / "data.csv")
        correct_answers = np.array(correct_answers_table["was_swapped"]).astype(np.int32)


        for j in range(len(table)):
            result_row = table.iloc[j]
            image_list = result_row["Input.images"].split(";")
            exp_name = str(Path(image_list[0]).parent.name)
            # print (exp_name)
            if Path(image_list[0]).parent.name != experiment_name:
                continue
            answers = result_row["Answer.submitValues"]
            if experiment_name not in experiment2scores.keys():
                experiment2scores[experiment_name] = []

            experiment2scores[experiment_name] += [get_experiment_score(experiment_name, image_filenames, answers, correct_answers)]

    experiment_average_scores = {}
    experiments_num_discarded = {}
    experiments_num_correct = {}
    for experiment_name in experiment2scores.keys():
        avg_score = 0
        participants_discarded = 0
        num_correct_participants = 0
        for result in experiment2scores[experiment_name]:
            if result[0]:
                avg_score += result[1]
                num_correct_participants += 1
            else:
                participants_discarded += 1
        avg_score /= num_correct_participants
        experiment_average_scores[experiment_name] = avg_score
        experiments_num_discarded[experiment_name] = participants_discarded
        experiments_num_correct[experiment_name] = num_correct_participants

    for key, value in experiment_average_scores.items():
        print(key, value)
        print("Discarded", experiments_num_discarded[key])
        print("Correct", experiments_num_correct[key])
        print(" ")

    # create a table from experiment_average_scores and experiments_num_correct
    experiment_table = pd.DataFrame( columns=["experiment", "score", "participants"])
    # for each key in experiment_table, add the entry to the table
    i = 0
    for key, value in experiment_average_scores.items():
        experiment_table.loc[i, "experiment"] = key
        experiment_table.loc[i, "score"] = experiment_average_scores[key]
        experiment_table.loc[i, "participants"] = experiments_num_correct[key]
        # experiment_table.loc[i, "experiment"] = experiments_num_discarded[key]
        i += 1
    experiment_table.to_csv(mturk_root / "final_experiment_scores.csv")

    with open(mturk_root / "final_experiment_scores.tex", "w") as f:
        experiment_table.to_latex(f,
                       index_names=False,
                       # float_format="{:0.2f}",
                       float_format=format,
                       # column_formatstr=len(table.colums) * "l" + "r",
                       )


def format(num):
    return f"{num:.02f}"

if __name__ == "__main__":
    main()



