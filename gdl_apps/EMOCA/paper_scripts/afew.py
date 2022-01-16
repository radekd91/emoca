from collections import OrderedDict
from pathlib import Path
import json
from tqdm import auto
from PIL import Image
import subprocess
from io import BytesIO


def main():
    """
    Open /ps/project/EmotionalFacialAnimation/data/afew/Train/Train_6.xml and extract actors and movies from it
    """
    # Open the file in question
    with open('/ps/project/EmotionalFacialAnimation/data/afew/Train/Train_6.xml', 'r') as f:
        # extract actors and movies from the xml elements
        # the xml attribute of movies is "MovieTitle"
        # the xml attribute of actors is "NameOfActor"

        # create a list of actors
        actors = []
        # create a list of movies
        movies = []

        # create a list of movies' lengths
        movie_lengths = []

        # loop through the xml attributes and extract actors and movies from them
        for line in f:
            # check if the line has an XML attribute "NameOfActor" and extract its value. The value follows the attribute and is enclosed in quotes
            if 'NameOfActor' in line:
                # extract the value of the attribute
                index = line.find("NameOfActor")
                actor = line[index:].split('"')[1]
                # add the actor to the list of actors
                actors.append(actor)
            # do the same for "MovieTitle"
            if 'MovieTitle' in line:
                index = line.find("MovieTitle")
                movie = line[index:].split('"')[1]
                movies.append(movie)
            # extract the length from the xml tag "<Length>"
            if 'Length' in line:
                index = line.find("<Length>") + len("<Length>")
                length = line[index:].split('<')[0]
                movie_lengths.append(int(length))



    # do the same for the file /ps/project/EmotionalFacialAnimation/data/afew/Val/Val_6.xml'
    with open('/ps/project/EmotionalFacialAnimation/data/afew/Val/Val_6.xml', 'r') as f:
        # extract actors and movies from the xml elements
        # the xml attribute of movies is "MovieTitle"
        # the xml attribute of actors is "NameOfActor"

        # loop through the xml attributes and extract actors and movies from them
        for line in f:
            # check if the line has an XML attribute "NameOfActor" and extract its value. The value follows the attribute and is enclosed in quotes
            if 'NameOfActor' in line:
                # extract the value of the attribute
                index = line.find("NameOfActor")
                actor = line[index:].split('"')[1]
                # add the actor to the list of actors
                actors.append(actor)
            # do the same for "MovieTitle"
            if 'MovieTitle' in line:
                index = line.find("MovieTitle")
                movie = line[index:].split('"')[1]
                movies.append(movie)
            # extract the length from the xml tag "<Length>"
            if 'Length' in line:
                index = line.find("<Length>") + len("<Length>")
                length = line[index:].split('<')[0]
                movie_lengths.append(int(length))


    # print the number of actors in a nice way
    print('There are {} actors in the dataset'.format(len(actors)))
    # print the number of movies in a nice way
    print('There are {} movies in the dataset'.format(len(movies)))

    # print the number of unique actors in a nice way
    print('There are {} unique actors in the dataset'.format(len(set(actors))))
    # print the number of unique movies in a nice way
    print('There are {} unique movies in the dataset'.format(len(set(movies))))

    # create a mapping from actors to list of movies they've been in. Make it an OrderedDict
    # so that the order of the actors is preserved
    actor_to_movies = OrderedDict()
    # loop through the actors
    for actor in actors:
        # create an empty set for the actor if there isn't one already
        if actor not in actor_to_movies:
            actor_to_movies[actor] = set()
        # add the movie to the set of movies for the actor
        actor_to_movies[actor].add(movie)

    # check if an actor has been in a movie more than once
    for actor, actor_movies in actor_to_movies.items():
        if len(actor_movies) > 1:
            print(actor, actor_movies)
            # print a message that this actor has been in a movie more than once
            print(actor, "has been in", len(actor_movies), "movies")
            # exit if an actor has been in a movie more than once with
            # print a nice message that one to many mapping is not supported
            # and exit
            exit('One to many mapping is not supported')


        # else:
            # #print a message that actor has only been in one movie
            # print(actor, 'has only been in one movie')

    # create a mapping from movies to list of actors who have been in it by inverting the actor_to_movies mapping
    movie_to_actors = OrderedDict()
    # loop through the movies
    for movie in movies:
        # create an empty set for the movie if there isn't one already
        if movie not in movie_to_actors:
            movie_to_actors[movie] = set()
        # add the actor to the set of actors for the movie
        movie_to_actors[movie].add(actor)


    # print unique number of actors in a nice way
    print('There are {} unique actors in the original dataset'.format(len(set(actors))))
    # print unique number of movies in a nice way
    print('There are {} unique movies in the original dataset'.format(len(set(movies))))

    # now look through this folder and find all json files in it recursively: /ps/project/EmotionalFacialAnimation/data/afew-va
    # create a list of all the json files
    json_files = []
    #  find all json files in the folder using pathlib.Path and put them into a list and sort it in one line
    json_files = sorted(list(Path('/ps/project/EmotionalFacialAnimation/data/afew-va').rglob('*.json')))
    # for each json file, extract the "actor" entries
    # and add them to a new list of actors
    # create a list of new actors
    new_actors = []
    actors_to_jsons = OrderedDict()
    # loop through the json files, use tqdm to dispay the progress
    for json_file in auto.tqdm(json_files):
        # open the json file using the json module
        with open(json_file, 'r') as f:
            # load the json file into a dictionary
            data = json.load(f)
            # extract the "actor" entries from the dictionary
            # and add them to the list of actors
            new_actors.append(data['actor'])
            # add the actor to the mapping from actors to json files
            actors_to_jsons[data['actor']] = json_file

    # print the number of actors
    print(len(set(new_actors)))

    # for each actor, loop through actor_to_movies and find the movies they have been in
    # and add them to a new list of movies
    # create a list of movies
    movies = []
    missing_actors = []
    not_missing_actors = []
    # loop through the actors
    for new_actor in new_actors:
        # check if new_actor is not in actor_to_movies keys
        if new_actor not in actor_to_movies.keys():
            # if not, add him to a new list of missing actors
            # and print a message
            # print(new_actor, 'is missing')
            missing_actors.append(new_actor)
            continue

        # loop through the actor_to_movies mapping
        for movie in actor_to_movies[new_actor]:
            # add the movie to the list of movies
            movies.append(movie)
        # print a warning if the actor has not been in any movies
        if len(actor_to_movies[new_actor]) == 0:
            print(new_actor, 'has not been in any movies')
        # print a warning if the actor has been in more than one movie
        if len(actor_to_movies[new_actor]) > 1:
            print(new_actor, 'has been in', len(actor_to_movies[new_actor]), 'movies')
        # add the actor to the list of not missing actors
        not_missing_actors.append(new_actor)


    # print the number of missing actors and their names
    print('There are {} missing actors'.format(len(set(missing_actors))))
    print(missing_actors)

    for actor in not_missing_actors:
        # find an image in the same folder as the json file, open it and recover its width and height
        # find the image in the smae folder
        image_file = actors_to_jsons[actor].parent / ("00000.png")
        # open the image
        image = Image.open(image_file)
        # get the width and height of the image
        width, height = image.size

        # get the first movie the actor has been in by accessing the first element of the set
        # of movies in the actor_to_movies mapping
        movie = list(actor_to_movies[actor])[0]


    # get the list of all avi files from /ps/project/EmotionalFacialAnimation/data/afew/ recursively in one line
    avi_files = sorted(list(Path('/ps/project/EmotionalFacialAnimation/data/afew/').rglob('*.avi')))
    # for each of the avi files, extract the number of frames of the video
    # and add it to a new list of frames
    # create a list of frames
    num_frames = []
    first_frames = []
    video_dimensions = []
    # loop through the avi files
    for avi_file in auto.tqdm(avi_files):
        # use python ffmpeg  package to extract the number of frames from the avi file
        # and add it to the list of frames
        # frames.append(int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1', avi_file])))
        num_frames.append(int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(avi_file)])))
        # use ffmpeg to extrach the width and height of the video
        # and add it to the list of video dimensions
        dims = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(avi_file)])
        dims = str(dims,'utf-8').split('\n')[0].split(',')
        video_dimensions += [[int(dims[0]), int(dims[1])]]

        import cv2
        vidcap = cv2.VideoCapture(str(avi_file))
        vidcap2 = cv2.VideoCapture(str(avi_file), cv2.CAP_DSHOW)
        success, image = vidcap.read()
        success2, image2 = vidcap2.read()
        # extract the first frame from each video file
        # and load it with PIL
        # and add it to a list of first frames

        # get the first frame from the video using cv2.VideoCapture
        # and add it to the list of first frames

        im_resized = cv2.resize(image, (int(1066.6666666666665), 576))
        im_resized = cv2.resize(image, (int(1066.6666666666665), 576))
        im_resized3 = cv2.resize(image, (720, 432))
        im_resized4 = cv2.resize(image, (960, 576))
        im_resized2 = cv2.resize(image, (1080, 576))


    # print the number of frames

    print("yo")








if __name__ == '__main__':
    main()